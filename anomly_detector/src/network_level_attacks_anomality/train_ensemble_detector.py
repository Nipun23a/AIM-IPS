"""
train_ensemble_detector.py
───────────────────────────
Multi-Model Anomaly Detector for Network-Level Attack Detection.

ALL models trained on BENIGN traffic only — no attack labels needed.
Ensemble catches attacks the single autoencoder missed.

Models:
  1. Deep Autoencoder (AE)
       Score = reconstruction MSE
       Strength: catches structural outliers, zero-days

  2. Variational Autoencoder (VAE)
       Score = ELBO reconstruction loss + KL divergence
       Strength: models the full benign probability distribution,
                 not just a point reconstruction — better calibrated

  3. Deep SVDD (Support Vector Data Description)
       Score = distance from learned hypersphere centre
       Strength: compact representation — benign collapses to a ball,
                 attacks land far outside regardless of reconstruction quality

  4. Isolation Forest
       Score = anomaly score from sklearn
       Strength: fast, non-parametric, excellent on tabular data,
                 catches port scans (low-volume, simple feature vectors
                 that autoencoders accidentally reconstruct well)

Ensemble fusion:
  final_score = w_ae * ae_score
              + w_vae * vae_score
              + w_svdd * svdd_score
              + w_iso * iso_score

  Weights learned by maximising AUC on a labelled hold-out set
  (benign val + all attack rows).

Threshold:
  Set at p95 of benign ensemble scores on the validation set.

Outputs:
  models/anomly_detector/network_level_attacks_anomality/ensemble/
      models/
          ae_fp32.tflite
          vae_fp32.tflite
          svdd_fp32.tflite
          iso_forest.pkl
          ensemble_weights.pkl      ← learned fusion weights
      logs/ plots/
  models/anomly_detector/network_level_attacks_anomality/
      ensemble_scaler.pkl           ← scaler + clip bounds
      ensemble_threshold.pkl        ← final decision threshold
      feature_info.pkl
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score
)

# ── GPU ───────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"✅ GPU: {gpus}")
    except RuntimeError as e:
        print(f"⚠ GPU: {e}")
else:
    print("⚠ CPU only")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[3] / "models" / "anomly_detector" / "network_level_attacks_anomality"
DATASET_DIR = Path(__file__).resolve().parents[3] / "data_collector" / "data_sets" / "cicids"
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")

ENS_DIR    = BASE_DIR / "ensemble"
MODELS_DIR = ENS_DIR  / "models"
PLOTS_DIR  = ENS_DIR  / "plots"
LOGS_DIR   = ENS_DIR  / "logs"

SCALER_PATH    = BASE_DIR / "ensemble_scaler.pkl"
THRESHOLD_PATH = BASE_DIR / "ensemble_threshold.pkl"
FEAT_INFO_PATH = BASE_DIR / "feature_info.pkl"
WEIGHTS_PATH   = MODELS_DIR / "ensemble_weights.pkl"

AE_TFLITE_PATH   = MODELS_DIR / "ae_fp32.tflite"
VAE_TFLITE_PATH  = MODELS_DIR / "vae_fp32.tflite"
SVDD_TFLITE_PATH = MODELS_DIR / "svdd_fp32.tflite"
ISO_PATH         = MODELS_DIR / "iso_forest.pkl"

for d in [BASE_DIR, ENS_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Features ──────────────────────────────────────────────────────────────────
THREAT_FEATURES = [
    "flow duration",            "total fwd packets",
    "total backward packets",   "total length of fwd packets",
    "total length of bwd packets", "fwd packet length mean",
    "bwd packet length mean",   "flow bytes/s",
    "flow packets/s",           "syn flag count",
    "ack flag count",           "psh flag count",
    "packet length mean",       "packet length std",
    "idle mean",                "idle std",
]
N_FEATURES = len(THREAT_FEATURES)


# =============================================================================
# DATA LOADING
# =============================================================================

def _norm_cols(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

def _map_label(raw):
    s = raw.strip().lower()
    if s == "benign":           return "benign"
    if "ddos" in s or "dos" in s: return "ddos"
    if "portscan" in s:         return "portscan"
    if s == "bot":              return "bot"
    return None                 # skip app-layer

def load_data(dataset_dir: Path):
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {dataset_dir}")
    print(f"[data] {len(csv_files)} CSV file(s)")

    LABEL_CANDS = ["label","attack_cat","attack","class"]
    benign, attack = [], []

    for p in csv_files:
        try:
            df = _norm_cols(pd.read_csv(p, low_memory=False))
        except Exception as e:
            print(f"[data] ⚠ {p.name}: {e}"); continue

        lc = next((c for c in LABEL_CANDS if c in df.columns), None)
        if lc is None:
            print(f"[data] ⚠ {p.name}: no label col"); continue
        if any(f not in df.columns for f in THREAT_FEATURES):
            print(f"[data] ⚠ {p.name}: missing features"); continue

        feat = df[THREAT_FEATURES].copy()
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        mask  = ~feat.isnull().any(axis=1)
        feat  = feat[mask].astype(float)
        lbls  = df[lc][mask].astype(str)

        stats = {}
        for raw in lbls.unique():
            cls = _map_label(raw)
            if cls is None: continue
            rows = feat[lbls == raw]
            stats[cls] = len(rows)
            (benign if cls == "benign" else attack).append(rows)
        print(f"[data] {p.name}: {stats}")

    if not benign:
        raise RuntimeError("No BENIGN flows found.")

    X_b = pd.concat(benign, ignore_index=True).values.astype(np.float32)
    X_a = (pd.concat(attack, ignore_index=True).values.astype(np.float32)
           if attack else np.empty((0, N_FEATURES), dtype=np.float32))

    print(f"\n[data] BENIGN: {len(X_b):,}   ATTACK: {len(X_a):,}")
    return X_b, X_a


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess(X_b, X_a):
    print("\n" + "="*70)
    print("PREPROCESSING")
    print("="*70)

    # Clip outliers on benign [p1, p99] per feature
    clip_lo = np.percentile(X_b, 1,  axis=0).astype(np.float32)
    clip_hi = np.percentile(X_b, 99, axis=0).astype(np.float32)
    X_b_cl  = np.clip(X_b, clip_lo, clip_hi).astype(np.float32)
    print(f"  Outlier clipping [p1,p99] on {N_FEATURES} features")

    scaler = StandardScaler()
    X_b_sc = scaler.fit_transform(X_b_cl).astype(np.float32)

    X_a_sc = np.empty((0, N_FEATURES), dtype=np.float32)
    if len(X_a) > 0:
        X_a_cl = np.clip(X_a, clip_lo, clip_hi).astype(np.float32)
        X_a_sc = scaler.transform(X_a_cl).astype(np.float32)

    joblib.dump({"scaler": scaler, "clip_lo": clip_lo, "clip_hi": clip_hi}, SCALER_PATH)
    print(f"  Scaler + clip bounds → {SCALER_PATH}")

    # Splits: 70% train / 15% val / 15% test  (all benign)
    X_tr, X_tmp = train_test_split(X_b_sc, test_size=0.30, random_state=42)
    X_val, X_te = train_test_split(X_tmp, test_size=0.50, random_state=42)

    print(f"  Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_te.shape}")
    print(f"  Attack eval: {X_a_sc.shape}")

    return X_tr, X_val, X_te, X_a_sc, scaler


# =============================================================================
# MODEL 1 — DEEP AUTOENCODER
# =============================================================================

def build_ae(dim: int) -> tf.keras.Model:
    reg = regularizers.l2(1e-4)
    inp = layers.Input(shape=(dim,), name="ae_input")

    # Encoder
    x = layers.Dense(128, kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64,  kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32,  kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    bottleneck = layers.Dense(8, activation="relu", name="ae_bottleneck")(x)

    # Decoder
    x = layers.Dense(32, kernel_regularizer=reg)(bottleneck)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    out = layers.Dense(dim, activation="linear", name="ae_output")(x)

    return models.Model(inp, out, name="autoencoder")


def train_ae(X_tr, X_val, dim, epochs=100, batch=256):
    print("\n" + "="*70)
    print("MODEL 1 — DEEP AUTOENCODER")
    print("="*70)
    m = build_ae(dim)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    print(f"  Parameters: {m.count_params():,}")

    ckpt = MODELS_DIR / f"ae_{TIMESTAMP}.keras"
    cb = [
        EarlyStopping("val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau("val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        CSVLogger(str(LOGS_DIR / f"ae_log_{TIMESTAMP}.csv")),
    ]
    m.fit(X_tr, X_tr, validation_data=(X_val, X_val),
          epochs=epochs, batch_size=batch, callbacks=cb, verbose=1)
    return m


def ae_scores(model, X):
    """Per-sample MSE reconstruction error."""
    recon = model.predict(X, verbose=0)
    return np.mean((X - recon) ** 2, axis=1).astype(np.float32)


# =============================================================================
# MODEL 2 — VARIATIONAL AUTOENCODER (VAE)
# =============================================================================

# =============================================================================
# MODEL 2 — VARIATIONAL AUTOENCODER  (Keras Model subclass)
#
# Why subclass instead of Functional API?
# ────────────────────────────────────────
# The ELBO loss (reconstruction + KL) references both the input tensor
# and the encoder outputs (z_mean, z_log_var).  In Keras 3 / TF2 eager
# mode you cannot call bare tf.* ops on KerasTensors inside build_vae()
# at graph-construction time — they are symbolic placeholders, not real
# tensors.  The fix is to compute the loss inside train_step() where all
# tensors are concrete.
# =============================================================================

class VAE(tf.keras.Model):
    """
    Beta-VAE subclass.
    Encoder: input → z_mean, z_log_var
    Decoder: z_sample → reconstruction
    Loss   : MSE reconstruction + beta * KL divergence
    """

    def __init__(self, input_dim: int, latent_dim: int = 8, beta: float = 0.001):
        super().__init__(name="vae")
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.beta       = beta
        reg = regularizers.l2(1e-4)

        # Encoder layers
        self.enc_d1  = layers.Dense(128, activation="relu", kernel_regularizer=reg)
        self.enc_bn1 = layers.BatchNormalization()
        self.enc_d2  = layers.Dense(64,  activation="relu", kernel_regularizer=reg)
        self.enc_bn2 = layers.BatchNormalization()
        self.enc_d3  = layers.Dense(32,  activation="relu", kernel_regularizer=reg)
        self.z_mean_layer    = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var_layer = layers.Dense(latent_dim, name="z_log_var")

        # Decoder layers
        self.dec_d1  = layers.Dense(32,  activation="relu", kernel_regularizer=reg)
        self.dec_bn1 = layers.BatchNormalization()
        self.dec_d2  = layers.Dense(64,  activation="relu", kernel_regularizer=reg)
        self.dec_bn2 = layers.BatchNormalization()
        self.dec_d3  = layers.Dense(128, activation="relu", kernel_regularizer=reg)
        self.dec_out = layers.Dense(input_dim, activation="linear")

    def encode(self, x, training=False):
        h = self.enc_bn1(self.enc_d1(x), training=training)
        h = self.enc_bn2(self.enc_d2(h), training=training)
        h = self.enc_d3(h)
        return self.z_mean_layer(h), self.z_log_var_layer(h)

    def reparameterise(self, z_mean, z_log_var):
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def decode(self, z, training=False):
        h = self.dec_bn1(self.dec_d1(z), training=training)
        h = self.dec_bn2(self.dec_d2(h), training=training)
        h = self.dec_d3(h)
        return self.dec_out(h)

    def call(self, x, training=False):
        z_mean, z_log_var = self.encode(x, training=training)
        z = self.reparameterise(z_mean, z_log_var)
        return self.decode(z, training=training)

    def train_step(self, data):
        # data is (X, X) because we pass X as both input and target
        x, _ = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(x, training=True)
            z     = self.reparameterise(z_mean, z_log_var)
            recon = self.decode(z, training=True)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(x - recon), axis=1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1,
                )
            )
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def test_step(self, data):
        x, _ = data
        z_mean, z_log_var = self.encode(x, training=False)
        z     = self.reparameterise(z_mean, z_log_var)
        recon = self.decode(z, training=False)
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - recon), axis=1))
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            )
        )
        total_loss = recon_loss + self.beta * kl_loss
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    # Make the model exportable to TFLite via a concrete function
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="vae_input")
    ])
    def serve(self, x):
        return self(x, training=False)


def train_vae(X_tr, X_val, dim, epochs=30, batch=256):
    print("\n" + "="*70)
    print("MODEL 2 — VARIATIONAL AUTOENCODER (VAE)")
    print("="*70)

    vae = VAE(input_dim=dim, latent_dim=8, beta=0.001)
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    # Build the model by passing a dummy batch
    _ = vae(tf.zeros((1, dim)))
    print(f"  Parameters: {vae.count_params():,}")

    ckpt_path = MODELS_DIR / f"vae_{TIMESTAMP}.weights.h5"
    cb = [
        EarlyStopping("val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau("val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        CSVLogger(str(LOGS_DIR / f"vae_log_{TIMESTAMP}.csv")),
        # Note: ModelCheckpoint on subclassed models requires save_weights_only=True
        ModelCheckpoint(str(ckpt_path), monitor="val_loss",
                        save_best_only=True, save_weights_only=True),
    ]

    vae.fit(
        X_tr, X_tr,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=cb,
        verbose=1,
    )
    return vae


def vae_scores(vae_model, X):
    """Reconstruction MSE as anomaly score (ELBO proxy)."""
    recon = vae_model.predict(X, verbose=0)
    return np.mean((X - recon) ** 2, axis=1).astype(np.float32)


# =============================================================================
# MODEL 3 — DEEP SVDD (Support Vector Data Description)
# =============================================================================

def build_svdd_network(dim: int, rep_dim: int = 16) -> tf.keras.Model:
    """
    Network maps input → compact representation space.
    SVDD loss pushes all benign samples toward a fixed hypersphere centre.
    Attacks fall outside the sphere.
    
    No bias in the last layer (standard Deep SVDD requirement — prevents
    the network collapsing to a constant mapping).
    """
    reg = regularizers.l2(1e-5)
    inp = layers.Input(shape=(dim,), name="svdd_input")

    x = layers.Dense(128, activation="relu", kernel_regularizer=reg, use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64,  activation="relu", kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32,  activation="relu", kernel_regularizer=reg, use_bias=False)(x)
    out = layers.Dense(rep_dim, activation="linear",
                       kernel_regularizer=reg, use_bias=False,
                       name="svdd_representation")(x)

    return models.Model(inp, out, name="svdd_net")


class SVDDTrainer:
    """
    One-class Deep SVDD.
    
    Training:
      1. Pretrain as autoencoder to initialise weights meaningfully
         (random init causes hypersphere collapse).
      2. Set sphere centre c = mean of all benign representations
         after pretraining (kept fixed during SVDD training).
      3. Train SVDD loss = mean distance² from c.
    """

    def __init__(self, dim: int, rep_dim: int = 16):
        self.dim     = dim
        self.rep_dim = rep_dim
        self.net     = None
        self.centre  = None

    def _build_pretrain_ae(self):
        """Temporary AE for weight initialisation."""
        reg = regularizers.l2(1e-5)
        inp = layers.Input(shape=(self.dim,))
        x   = layers.Dense(128, activation="relu", kernel_regularizer=reg)(inp)
        x   = layers.Dense(64,  activation="relu", kernel_regularizer=reg)(x)
        x   = layers.Dense(32,  activation="relu", kernel_regularizer=reg)(x)
        enc = layers.Dense(self.rep_dim, activation="linear", name="svdd_representation")(x)
        x   = layers.Dense(32,  activation="relu")(enc)
        x   = layers.Dense(64,  activation="relu")(x)
        x   = layers.Dense(128, activation="relu")(x)
        out = layers.Dense(self.dim, activation="linear")(x)
        return models.Model(inp, out, name="svdd_pretrain_ae")

    def pretrain(self, X_tr, X_val, epochs=10, batch=256):
        print("  [SVDD] Pre-training encoder via AE reconstruction ...")
        ae = self._build_pretrain_ae()
        ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        ae.fit(X_tr, X_tr, validation_data=(X_val, X_val),
               epochs=epochs, batch_size=batch,
               callbacks=[EarlyStopping("val_loss", patience=8,
                                        restore_best_weights=True)],
               verbose=0)

        # Build SVDD net and copy weights from pretrained encoder layers
        self.net = build_svdd_network(self.dim, self.rep_dim)
        for svdd_layer in self.net.layers:
            try:
                ae_layer = ae.get_layer(svdd_layer.name)
                svdd_layer.set_weights(ae_layer.get_weights())
            except (ValueError, AttributeError):
                pass  # layer not in AE (e.g. BatchNorm) — skip
        print("  [SVDD] Encoder weights initialised from AE pretraining")

    def set_centre(self, X_tr, batch=4096):
        """Set sphere centre = mean of encoder outputs on all training data."""
        reps = []
        for i in range(0, len(X_tr), batch):
            reps.append(self.net.predict(X_tr[i:i+batch], verbose=0))
        reps = np.vstack(reps)
        self.centre = np.mean(reps, axis=0).astype(np.float32)
        print(f"  [SVDD] Centre set — shape={self.centre.shape}  "
              f"norm={np.linalg.norm(self.centre):.4f}")

    def train(self, X_tr, X_val, epochs=15, batch=256):
        """Train with SVDD loss = mean squared distance from centre."""
        print("  [SVDD] Training SVDD objective ...")
        c = tf.constant(self.centre, dtype=tf.float32)

        @tf.function
        def svdd_loss(y_true, y_pred):
            # y_pred = encoder representations, shape (batch, rep_dim)
            # loss = mean Euclidean distance² from centre
            diff = y_pred - c
            return tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1))

        self.net.compile(
            optimizer=tf.keras.optimizers.Adam(5e-4),
            loss=svdd_loss,
        )

        ckpt = MODELS_DIR / f"svdd_{TIMESTAMP}.keras"
        cb = [
            EarlyStopping("val_loss", patience=15,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau("val_loss", factor=0.5, patience=7,
                              min_lr=1e-6, verbose=1),
            ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
            CSVLogger(str(LOGS_DIR / f"svdd_log_{TIMESTAMP}.csv")),
        ]

        # For SVDD loss, y_target doesn't matter — use centre as dummy target
        c_tr  = np.tile(self.centre, (len(X_tr),  1))
        c_val = np.tile(self.centre, (len(X_val), 1))

        self.net.fit(X_tr, c_tr, validation_data=(X_val, c_val),
                     epochs=epochs, batch_size=batch, callbacks=cb, verbose=1)
        return self.net

    def scores(self, X):
        """Distance² from sphere centre = anomaly score."""
        reps  = self.net.predict(X, verbose=0)
        dists = np.sum((reps - self.centre) ** 2, axis=1)
        return dists.astype(np.float32)


# =============================================================================
# MODEL 4 — ISOLATION FOREST
# =============================================================================

def train_iso(X_tr, contamination=0.05):
    print("\n" + "="*70)
    print("MODEL 4 — ISOLATION FOREST")
    print("="*70)
    iso = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=contamination,  # expected fraction of anomalies
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    iso.fit(X_tr)
    print(f"  Trained on {len(X_tr):,} benign samples")
    joblib.dump(iso, ISO_PATH)
    print(f"  Saved → {ISO_PATH}")
    return iso


def iso_scores(iso, X):
    """
    IsolationForest returns negative anomaly scores (lower = more anomalous).
    Invert and shift to [0, +inf) so higher = more anomalous, consistent
    with the other models.
    """
    raw = iso.score_samples(X)          # range roughly [-0.5, 0.5]
    return (-raw).astype(np.float32)    # higher = more anomalous


# =============================================================================
# SCORE NORMALISATION
# =============================================================================

def normalise_scores(scores_dict: dict, fit_on: np.ndarray = None) -> dict:
    """
    Min-max normalise each model's scores to [0, 1] so they can be
    combined.  If fit_on is provided, use those samples to set the scale
    (should be benign validation scores so 0 = typical benign).
    """
    normed = {}
    scalers = {}
    for name, s in scores_dict.items():
        mn = fit_on[name].min() if fit_on else s.min()
        mx = fit_on[name].max() if fit_on else s.max()
        normed[name]  = np.clip((s - mn) / (mx - mn + 1e-9), 0, 1)
        scalers[name] = (mn, mx)
    return normed, scalers


# =============================================================================
# ENSEMBLE WEIGHT LEARNING
# =============================================================================

def learn_weights(
    val_scores:    dict,    # {model_name: np.array of scores on val benign}
    attack_scores: dict,    # {model_name: np.array of scores on attacks}
) -> np.ndarray:
    """
    Learn fusion weights by maximising AUC on the labelled hold-out
    (val benign + attack).

    Uses constrained optimisation: weights ∈ [0,1], sum = 1.
    Objective: maximise AUC-ROC of weighted sum score.
    """
    print("\n" + "="*70)
    print("LEARNING ENSEMBLE WEIGHTS (maximise AUC)")
    print("="*70)

    names = list(val_scores.keys())
    n     = len(names)

    # Build evaluation arrays
    S_ben = np.column_stack([val_scores[k]    for k in names])  # (n_ben, n_models)
    S_att = np.column_stack([attack_scores[k] for k in names])  # (n_att, n_models)

    S_all = np.vstack([S_ben, S_att])
    y_all = np.concatenate([
        np.zeros(len(S_ben), dtype=int),
        np.ones( len(S_att), dtype=int),
    ])

    def neg_auc(w):
        w = np.array(w)
        w = w / (w.sum() + 1e-9)
        fused = S_all @ w
        try:
            return -roc_auc_score(y_all, fused)
        except Exception:
            return 0.0

    # Multiple random restarts to avoid local minima
    best_w    = np.ones(n) / n
    best_auc  = -neg_auc(best_w)
    rng = np.random.default_rng(42)

    for trial in range(30):
        w0 = rng.dirichlet(np.ones(n))
        res = minimize(
            neg_auc, w0,
            method="SLSQP",
            bounds=[(0.05, 0.8)] * n,          # each model contributes at least 5%
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if res.success and (-res.fun) > best_auc:
            best_auc = -res.fun
            best_w   = res.x

    best_w = best_w / best_w.sum()

    print(f"\n  Learned weights (AUC = {best_auc:.4f}):")
    for name, w in zip(names, best_w):
        bar = "█" * int(w * 40)
        print(f"    {name:<20} {w:.4f}  {bar}")

    # Also print equal-weight baseline
    eq_w  = np.ones(n) / n
    eq_auc = -neg_auc(eq_w)
    print(f"\n  Equal-weight baseline AUC = {eq_auc:.4f}")
    print(f"  Improvement from learning  = {best_auc - eq_auc:+.4f}")

    return best_w, names


# =============================================================================
# TFLITE CONVERSION  (AE, VAE, SVDD encoder)
# =============================================================================

def _to_tflite(keras_model, out_path: Path, X_calib: np.ndarray):
    """Convert a Keras model to FP32 TFLite."""
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = []
    tflite = conv.convert()
    out_path.write_bytes(tflite)
    print(f"  TFLite → {out_path}  ({len(tflite)/1024:.1f} KB)")
    return out_path


def _tflite_infer(path: Path, X: np.ndarray) -> np.ndarray:
    """Run TFLite model, return output array."""
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    results = []
    for i in range(len(X)):
        interp.set_tensor(inp[0]["index"], X[i:i+1].astype(np.float32))
        interp.invoke()
        results.append(interp.get_tensor(out[0]["index"])[0])
    return np.array(results, dtype=np.float32)


# =============================================================================
# THRESHOLD CALIBRATION
# =============================================================================

def calibrate_threshold(
    ensemble_scores_val: np.ndarray,
    percentile: float = 95.0,
) -> float:
    print("\n" + "="*70)
    print(f"THRESHOLD CALIBRATION (p{percentile} of benign ensemble scores)")
    print("="*70)

    print("  Benign ensemble score statistics:")
    for p in [50, 75, 90, 92, 95, 97, 99]:
        marker = " ← selected" if p == percentile else ""
        print(f"    p{p:<3} = {np.percentile(ensemble_scores_val, p):.6f}{marker}")

    threshold = float(np.percentile(ensemble_scores_val, percentile))
    print(f"\n  ✅ Threshold = {threshold:.6f}  (~{100-percentile:.0f}% FPR)")

    joblib.dump({
        "threshold": threshold, "percentile": percentile,
        "benign_mean": float(ensemble_scores_val.mean()),
        "benign_std":  float(ensemble_scores_val.std()),
        "timestamp":   TIMESTAMP,
    }, THRESHOLD_PATH)
    print(f"  💾 Saved → {THRESHOLD_PATH}")
    return threshold


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_ensemble(
    ensemble_val:    np.ndarray,   # scores for benign val
    ensemble_attack: np.ndarray,   # scores for attacks
    threshold:       float,
    label:           str = "Ensemble",
) -> dict:
    X_all = np.concatenate([ensemble_val, ensemble_attack])
    y_all = np.concatenate([
        np.zeros(len(ensemble_val),    dtype=int),
        np.ones( len(ensemble_attack), dtype=int),
    ])
    y_pred = (X_all > threshold).astype(int)

    acc  = accuracy_score(y_all, y_pred)
    prec = precision_score(y_all, y_pred, zero_division=0)
    rec  = recall_score(y_all, y_pred, zero_division=0)
    f1   = f1_score(y_all, y_pred, zero_division=0)
    auc  = roc_auc_score(y_all, X_all)
    cm   = confusion_matrix(y_all, y_pred)
    TN, FP, FN, TP = cm.ravel()
    fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr  = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    det  = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    print(f"\n── {label} ──")
    print(f"   Accuracy:            {acc*100:.2f}%")
    print(f"   Precision:           {prec*100:.2f}%")
    print(f"   Recall / Detection:  {rec*100:.2f}%   ← want this HIGH")
    print(f"   F1-Score:            {f1*100:.2f}%")
    print(f"   AUC-ROC:             {auc*100:.2f}%")
    print(f"   False Positive Rate: {fpr*100:.2f}%   ← want this LOW")
    print(f"   False Negative Rate: {fnr*100:.2f}%   ← want this LOW")
    print(f"   Detection Rate:      {det*100:.2f}%")
    print(classification_report(y_all, y_pred,
                                target_names=["Benign","Attack"], digits=4))

    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                fpr=fpr, fnr=fnr, det=det,
                confusion_matrix=cm, scores=X_all, y_true=y_all)


def evaluate_per_attack_type(
    model_scores_val:    dict,    # {model: benign_val_scores}
    model_scores_attack: dict,    # {model: attack_scores}
    weights:             np.ndarray,
    weight_names:        list,
    threshold:           float,
    X_a_raw:             np.ndarray,
    attack_labels:       np.ndarray,
):
    """Break down detection rate per attack type (DDoS / PortScan / Bot)."""
    print("\n── Per-Attack-Type Detection ──")

    w_map = dict(zip(weight_names, weights))
    for atk_type in np.unique(attack_labels):
        mask = attack_labels == atk_type
        fused = sum(
            w_map[k] * model_scores_attack[k][mask]
            for k in weight_names
        )
        detected = np.sum(fused > threshold)
        total    = np.sum(mask)
        print(f"   {atk_type:<12}  detected {detected:>6,}/{total:>6,}  "
              f"({detected/total*100:.1f}%)")


# =============================================================================
# PLOTS
# =============================================================================

def plot_score_distributions(
    model_scores_val:    dict,
    model_scores_attack: dict,
    ensemble_val:        np.ndarray,
    ensemble_attack:     np.ndarray,
    threshold:           float,
):
    names    = list(model_scores_val.keys()) + ["ENSEMBLE"]
    val_s    = list(model_scores_val.values())    + [ensemble_val]
    att_s    = list(model_scores_attack.values()) + [ensemble_attack]

    ncols = 3
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle("Anomaly Score Distributions — Benign vs Attack", fontsize=14)
    axes = axes.flatten()

    for i, (name, vs, ats) in enumerate(zip(names, val_s, att_s)):
        ax = axes[i]
        clip = np.percentile(np.concatenate([vs, ats]), 99) * 1.1
        bins = np.linspace(0, clip, 100)
        ax.hist(np.clip(vs,  0, clip), bins=bins, alpha=0.6,
                color="steelblue", density=True, label=f"Benign (n={len(vs):,})")
        ax.hist(np.clip(ats, 0, clip), bins=bins, alpha=0.6,
                color="tomato",    density=True, label=f"Attack (n={len(ats):,})")
        if name == "ENSEMBLE":
            ax.axvline(threshold, color="black", lw=2, ls="--",
                       label=f"Threshold={threshold:.3f}")
        ax.set_title(name); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out = PLOTS_DIR / f"distributions_{TIMESTAMP}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close()
    print(f"📊 Distributions → {out}")


def plot_roc_comparison(
    model_scores_val:    dict,
    model_scores_attack: dict,
    ensemble_val:        np.ndarray,
    ensemble_attack:     np.ndarray,
):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("ROC Curves — All Models vs Ensemble", fontsize=13)

    all_scores = {**model_scores_val, "ENSEMBLE": ensemble_val}
    all_att    = {**model_scores_attack, "ENSEMBLE": ensemble_attack}

    for name in all_scores:
        vs  = all_scores[name]
        ats = all_att[name]
        y   = np.concatenate([np.zeros(len(vs)), np.ones(len(ats))])
        s   = np.concatenate([vs, ats])
        fpr_arr, tpr_arr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        lw  = 3 if name == "ENSEMBLE" else 1.5
        ls  = "-" if name == "ENSEMBLE" else "--"
        ax.plot(fpr_arr, tpr_arr, lw=lw, ls=ls, label=f"{name}  AUC={auc:.4f}")

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(); ax.grid(alpha=0.3)

    out = PLOTS_DIR / f"roc_{TIMESTAMP}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close()
    print(f"📊 ROC → {out}")


def plot_weight_bar(weights, names):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["steelblue", "seagreen", "darkorange", "mediumpurple"]
    ax.bar(names, weights, color=colors[:len(names)], edgecolor="black")
    ax.set_ylabel("Ensemble Weight"); ax.set_title("Learned Ensemble Weights")
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.4)
    for i, (n, w) in enumerate(zip(names, weights)):
        ax.text(i, w + 0.01, f"{w:.3f}", ha="center", fontsize=11, fontweight="bold")
    out = PLOTS_DIR / f"weights_{TIMESTAMP}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close()
    print(f"📊 Weights → {out}")


# =============================================================================
# MAIN
# =============================================================================

def main(
    dataset_dir:  Path  = DATASET_DIR,
    epochs:       int   = 30,
    batch_size:   int   = 256,
    threshold_p:  float = 95.0,
):
    print("\n" + "="*70)
    print("MULTI-MODEL ENSEMBLE ANOMALY DETECTOR")
    print("AE + VAE + Deep SVDD + Isolation Forest")
    print("Trained on BENIGN only — zero-day capable")
    print("="*70)

    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    X_b, X_a = load_data(dataset_dir)

    # Keep attack labels for per-type evaluation
    # Re-load labels (we need them here, not in load_data which drops them)
    attack_label_list = []
    LABEL_CANDS = ["label","attack_cat","attack","class"]
    for p in sorted(dataset_dir.glob("*.csv")):
        try:
            df = _norm_cols(pd.read_csv(p, low_memory=False))
        except Exception:
            continue
        lc = next((c for c in LABEL_CANDS if c in df.columns), None)
        if lc is None or any(f not in df.columns for f in THREAT_FEATURES):
            continue
        feat = df[THREAT_FEATURES].copy()
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        mask = ~feat.isnull().any(axis=1)
        lbls = df[lc][mask].astype(str)
        for raw in lbls.unique():
            cls = _map_label(raw)
            if cls and cls != "benign":
                attack_label_list.extend([cls] * int((lbls == raw).sum()))
    attack_labels = np.array(attack_label_list)

    X_tr, X_val, X_te, X_a_sc, scaler = preprocess(X_b, X_a)
    dim = X_tr.shape[1]

    # Save feature info
    joblib.dump({"features": THREAT_FEATURES, "n_features": dim, "timestamp": TIMESTAMP},
                FEAT_INFO_PATH)

    # ── 2. Train all 4 models ─────────────────────────────────────────────────
    # AE
    ae_model  = train_ae(X_tr, X_val, dim, epochs, batch_size)
    _to_tflite(ae_model, AE_TFLITE_PATH, X_tr[:500])

    # VAE
    vae_model = train_vae(X_tr, X_val, dim, epochs, batch_size)
    # Export VAE for TFLite using the concrete serve() function
    print("  [VAE] Converting to TFLite ...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [vae_model.serve.get_concrete_function()], vae_model
    )
    converter.optimizations = []
    tflite_model = converter.convert()
    VAE_TFLITE_PATH.write_bytes(tflite_model)
    print(f"  TFLite → {VAE_TFLITE_PATH}  ({len(tflite_model)/1024:.1f} KB)")

    # SVDD
    print("\n" + "="*70)
    print("MODEL 3 — DEEP SVDD")
    print("="*70)
    svdd = SVDDTrainer(dim, rep_dim=16)
    svdd.pretrain(X_tr, X_val, epochs=30, batch=batch_size)
    svdd.set_centre(X_tr)
    svdd.train(X_tr, X_val, epochs=min(epochs, 30), batch=batch_size)
    _to_tflite(svdd.net, SVDD_TFLITE_PATH, X_tr[:500])
    # Save SVDD centre
    joblib.dump({"centre": svdd.centre, "rep_dim": svdd.rep_dim}, 
                MODELS_DIR / "svdd_centre.pkl")

    # Isolation Forest
    iso_model = train_iso(X_tr, contamination=0.05)

    # ── 3. Compute raw scores on val (benign) + attacks ───────────────────────
    print("\n[scores] Computing raw scores on val + attack sets ...")
    raw_val = {
        "AE":         ae_scores(ae_model,  X_val),
        "VAE":        vae_scores(vae_model, X_val),
        "SVDD":       svdd.scores(X_val),
        "IsoForest":  iso_scores(iso_model, X_val),
    }
    raw_att = {
        "AE":         ae_scores(ae_model,  X_a_sc),
        "VAE":        vae_scores(vae_model, X_a_sc),
        "SVDD":       svdd.scores(X_a_sc),
        "IsoForest":  iso_scores(iso_model, X_a_sc),
    } if len(X_a_sc) > 0 else {k: np.array([]) for k in raw_val}

    # Per-model AUC (before ensemble)
    print("\n── Individual model AUCs ──")
    for name in raw_val:
        if len(raw_att[name]) > 0:
            y = np.concatenate([np.zeros(len(raw_val[name])),
                                 np.ones(len(raw_att[name]))])
            s = np.concatenate([raw_val[name], raw_att[name]])
            print(f"   {name:<14}  AUC={roc_auc_score(y, s):.4f}")

    # ── 4. Normalise scores to [0,1] ─────────────────────────────────────────
    # Fit normalisation on val benign so 0 = typical normal traffic
    norm_val, score_scalers = normalise_scores(raw_val)
    norm_att, _             = normalise_scores(raw_att, fit_on=raw_val)

    joblib.dump(score_scalers, MODELS_DIR / "score_scalers.pkl")

    # ── 5. Learn ensemble weights ─────────────────────────────────────────────
    if len(X_a_sc) > 0:
        weights, w_names = learn_weights(norm_val, norm_att)
    else:
        # No attacks in dataset — equal weights
        w_names = list(norm_val.keys())
        weights = np.ones(len(w_names)) / len(w_names)
        print("⚠ No attack data — using equal weights")

    joblib.dump({"weights": weights, "names": w_names,
                 "score_scalers": score_scalers, "timestamp": TIMESTAMP},
                WEIGHTS_PATH)
    print(f"\n💾 Weights saved → {WEIGHTS_PATH}")

    # ── 6. Compute ensemble scores ────────────────────────────────────────────
    ens_val = sum(weights[i] * norm_val[n] for i, n in enumerate(w_names))
    ens_att = (sum(weights[i] * norm_att[n] for i, n in enumerate(w_names))
               if len(X_a_sc) > 0 else np.array([]))

    # ── 7. Calibrate threshold ────────────────────────────────────────────────
    threshold = calibrate_threshold(ens_val, threshold_p)

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)

    for name in w_names:
        evaluate_ensemble(
            norm_val[name], norm_att[name] if len(norm_att[name]) > 0 else np.array([]),
            threshold=float(np.percentile(norm_val[name], threshold_p)),
            label=f"Solo: {name}",
        )

    final_results = evaluate_ensemble(ens_val, ens_att, threshold, "ENSEMBLE (final)")

    if len(attack_labels) > 0 and len(X_a_sc) > 0:
        evaluate_per_attack_type(
            norm_val, norm_att, weights, w_names, threshold,
            X_a_sc, attack_labels[:len(X_a_sc)],
        )

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    plot_score_distributions(norm_val, norm_att, ens_val, ens_att, threshold)
    plot_roc_comparison(norm_val, norm_att, ens_val, ens_att)
    plot_weight_bar(weights, w_names)

    # ── 10. Summary ───────────────────────────────────────────────────────────
    r = final_results
    summary = LOGS_DIR / f"summary_{TIMESTAMP}.txt"
    with open(summary, "w") as f:
        f.write("="*70 + "\n")
        f.write("MULTI-MODEL ENSEMBLE — TRAINING SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp      : {TIMESTAMP}\n")
        f.write(f"Models         : AE + VAE + SVDD + IsolationForest\n")
        f.write(f"Train (benign) : {len(X_tr):,}\n")
        f.write(f"Val   (benign) : {len(X_val):,}\n")
        f.write(f"Attack (eval)  : {len(X_a_sc):,}\n")
        f.write(f"Threshold      : {threshold:.6f}  (p{threshold_p})\n\n")
        f.write("Ensemble weights:\n")
        for n, w in zip(w_names, weights):
            f.write(f"  {n:<14} : {w:.4f}\n")
        f.write("\nFinal Ensemble Performance:\n")
        for k, label in [("acc","Accuracy"), ("prec","Precision"),
                          ("rec","Detection Rate"), ("f1","F1"),
                          ("auc","AUC-ROC"), ("fpr","False Positive Rate"),
                          ("fnr","False Negative Rate")]:
            f.write(f"  {label:<22}: {r.get(k,0)*100:.2f}%\n")

    print(f"\n💾 Summary → {summary}")
    print("\n" + "="*70)
    print("✅ DONE")
    print("="*70)
    print(f"  AE TFLite    : {AE_TFLITE_PATH}")
    print(f"  VAE TFLite   : {VAE_TFLITE_PATH}")
    print(f"  SVDD TFLite  : {SVDD_TFLITE_PATH}")
    print(f"  IsoForest    : {ISO_PATH}")
    print(f"  Weights      : {WEIGHTS_PATH}")
    print(f"  Scaler       : {SCALER_PATH}")
    print(f"  Threshold    : {THRESHOLD_PATH}")
    print("="*70 + "\n")

    return {"ae": ae_model, "vae": vae_model, "svdd": svdd,
            "iso": iso_model, "weights": weights, "threshold": threshold}


if __name__ == "__main__":
    if not DATASET_DIR.exists():
        print(f"⚠ Dataset not found: {DATASET_DIR.resolve()}")
        sys.exit(1)
    main()