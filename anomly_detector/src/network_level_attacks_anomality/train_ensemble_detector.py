"""
train_ensemble_detector.py
───────────────────────────
Multi-Model Anomaly Detector for Network-Level Attack Detection.

Semi-supervised ensemble:
  AE  + VAE  → trained on BENIGN only  (preserve zero-day detection)
  OCC-Net    → trained on BENIGN + REAL attacks as negatives
  IsoForest  → contamination set from real attack ratio in dataset
  Threshold  → optimal F1 point on labelled hold-out (not fixed p95)

Models:
  1. Deep Autoencoder (AE)
       Score = reconstruction MSE
       Strength: catches structural outliers, zero-days

  2. Variational Autoencoder (VAE)
       Score = ELBO reconstruction loss + KL divergence
       Strength: models the full benign probability distribution,
                 not just a point reconstruction — better calibrated

  3. OCC-Net  (One-Class Classifier Network)
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
              + w_occ  * occ_score 
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
          occ_fp32.tflite 
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
OCC_TFLITE_PATH  = MODELS_DIR / "occ_fp32.tflite"
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
    """
    Returns
    -------
    X_b           : (N_benign, 16)  benign feature matrix
    X_a           : (N_attack, 16)  attack feature matrix
    attack_labels : (N_attack,)     string label per attack row
                    e.g. "ddos", "portscan", "bot"
    """
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {dataset_dir}")
    print(f"[data] {len(csv_files)} CSV file(s)")

    LABEL_CANDS = ["label","attack_cat","attack","class"]
    benign, attack_rows, attack_lbls = [], [], []

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
            if cls == "benign":
                benign.append(rows)
            else:
                attack_rows.append(rows)
                attack_lbls.extend([cls] * len(rows))
        print(f"[data] {p.name}: {stats}")

    if not benign:
        raise RuntimeError("No BENIGN flows found.")

    X_b = pd.concat(benign, ignore_index=True).values.astype(np.float32)
    if attack_rows:
        X_a  = pd.concat(attack_rows, ignore_index=True).values.astype(np.float32)
        a_lb = np.array(attack_lbls, dtype=str)
    else:
        X_a  = np.empty((0, N_FEATURES), dtype=np.float32)
        a_lb = np.array([], dtype=str)

    print(f"\n[data] BENIGN: {len(X_b):,}   ATTACK: {len(X_a):,}")
    if len(X_a) > 0:
        for cls in np.unique(a_lb):
            print(f"         {cls:<12}: {(a_lb==cls).sum():,}")
    return X_b, X_a, a_lb


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
# MODEL 3 — OCC-Net  (One-Class Classifier Network)
#
# Replaces Deep SVDD which required two training stages (AE pretraining
# then SVDD objective), risked hypersphere collapse, and was slow.
#
# OCC-Net design:
#   Architecture : 16 → 32 → 16 → 8 → 1 (sigmoid)
#   Size         : ~5 KB TFLite  (vs ~80 KB for SVDD)
#   Latency      : ~0.001 ms per flow (single forward pass)
#   Training     : One stage — soft boundary loss with pseudo-negatives
#
# How it works:
#   Pseudo-negatives are synthetic "fake attack" samples generated by
#   adding Gaussian noise scaled to 3× the standard deviation of each
#   benign feature.  This pushes the decision boundary outward from the
#   benign cluster without needing real attack labels.
#
#   Loss = BCE(benign→1, pseudo_neg→0)
#          + λ · L2 regularisation
#
#   At inference:  score = 1 - model(x)
#   (model outputs "normality probability"; we invert for anomaly score)
#
# Why this beats SVDD for this task:
#   • Single training stage  — no collapse risk, simpler
#   • Sigmoid output          — well-calibrated probabilities
#   • Pseudo-negatives        — shapes a realistic boundary, not just
#                               "collapse everything to a point"
#   • 10× smaller TFLite     — faster cold-start, lower memory
# =============================================================================

def build_occ_net(dim: int) -> tf.keras.Model:
    """
    Compact one-class MLP: 16 → 32 → 16 → 8 → 1
    Output: P(normal) in [0, 1]
    Anomaly score at inference = 1 - output
    """
    reg = regularizers.l2(1e-4)
    inp = layers.Input(shape=(dim,), name="occ_input")

    x = layers.Dense(32, kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dense(16, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Dense(8, kernel_regularizer=reg)(x)
    x = layers.Activation("relu")(x)

    out = layers.Dense(1, activation="sigmoid", name="normality_prob")(x)
    return models.Model(inp, out, name="occ_net")


def _make_pseudo_negatives(X_benign: np.ndarray, n: int, noise_scale: float = 3.0,
                            rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate synthetic pseudo-negative samples by perturbing benign data
    with large Gaussian noise.  noise_scale=3 places samples ~3σ outside
    the normal distribution — far enough to be anomalous, not so far that
    the network trivially separates them without learning a useful boundary.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    std = X_benign.std(axis=0) + 1e-6
    idx = rng.choice(len(X_benign), size=n, replace=True)
    noise = rng.normal(0, noise_scale, size=(n, X_benign.shape[1])).astype(np.float32)
    return (X_benign[idx] + noise * std).astype(np.float32)


def train_occ(X_tr: np.ndarray, X_val: np.ndarray,
              dim: int, epochs: int = 30, batch: int = 256,
              X_attack: np.ndarray = None) -> tf.keras.Model:
    """
    Semi-supervised OCC-Net.

    Negative samples priority:
      1. Real attack flows (X_attack) — if provided, split 80/20 train/val
         These are the actual CICIDS attack flows (scaled), giving the model
         a true picture of what attacks look like.
      2. Pseudo-negatives (Gaussian noise) — fallback if no attacks available.
         Only used when the dataset has no labelled attacks at all.

    AE and VAE remain trained on benign only so the ensemble retains
    zero-day detection capability.
    """
    print("\n" + "="*70)
    print("MODEL 3 — OCC-Net (Semi-supervised: benign=1, attacks=0)")
    print("="*70)

    model = build_occ_net(dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print(f"  Parameters: {model.count_params():,}  "
          f"(≈{model.count_params()*4/1024:.1f} KB weights)")

    rng = np.random.default_rng(42)

    use_real = X_attack is not None and len(X_attack) > 0

    if use_real:
        # ── Semi-supervised: use real attack flows as negatives ───────────────
        # Split attacks 80/20 to get a validation set that mirrors training
        n_att_tr  = int(len(X_attack) * 0.8)
        perm_att  = rng.permutation(len(X_attack))
        X_att_tr  = X_attack[perm_att[:n_att_tr]]
        X_att_val = X_attack[perm_att[n_att_tr:]]

        # Balance: downsample whichever class is larger so BCE is unbiased
        n_tr  = min(len(X_tr),  len(X_att_tr))
        n_val = min(len(X_val), len(X_att_val))

        X_occ_tr  = np.vstack([X_tr[:n_tr],   X_att_tr[:n_tr]  ]).astype(np.float32)
        y_occ_tr  = np.concatenate([np.ones(n_tr),  np.zeros(n_tr)  ]).astype(np.float32)
        X_occ_val = np.vstack([X_val[:n_val],  X_att_val[:n_val]]).astype(np.float32)
        y_occ_val = np.concatenate([np.ones(n_val), np.zeros(n_val) ]).astype(np.float32)

        print(f"  Negatives  : REAL ATTACK FLOWS ({len(X_att_tr):,} train, "
              f"{len(X_att_val):,} val)")
        print(f"  Train      : {len(X_occ_tr):,} balanced samples "
              f"({n_tr:,} benign + {n_tr:,} attack)")
        print(f"  Val        : {len(X_occ_val):,} balanced samples")
    else:
        # ── Fallback: pseudo-negatives (Gaussian noise) ───────────────────────
        print("  ⚠ No attack data — falling back to Gaussian pseudo-negatives")
        X_pn_tr  = _make_pseudo_negatives(X_tr,  n=len(X_tr),  rng=rng)
        X_pn_val = _make_pseudo_negatives(X_val, n=len(X_val), rng=rng)
        X_occ_tr  = np.vstack([X_tr,  X_pn_tr ]).astype(np.float32)
        y_occ_tr  = np.concatenate([np.ones(len(X_tr)),  np.zeros(len(X_pn_tr)) ]).astype(np.float32)
        X_occ_val = np.vstack([X_val, X_pn_val]).astype(np.float32)
        y_occ_val = np.concatenate([np.ones(len(X_val)), np.zeros(len(X_pn_val))]).astype(np.float32)
        print(f"  Train: {len(X_occ_tr):,} ({len(X_tr):,} benign + {len(X_pn_tr):,} pseudo-neg)")

    ckpt = MODELS_DIR / f"occ_{TIMESTAMP}.keras"
    cb = [
        EarlyStopping("val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau("val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        CSVLogger(str(LOGS_DIR / f"occ_log_{TIMESTAMP}.csv")),
    ]

    model.fit(
        X_occ_tr,  y_occ_tr,
        validation_data=(X_occ_val, y_occ_val),
        epochs=epochs, batch_size=batch,
        callbacks=cb, verbose=1,
    )
    return model


def occ_scores(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Anomaly score = 1 - P(normal).
    High score → anomalous.  Low score → normal.
    """
    prob_normal = model.predict(X, verbose=0).flatten()
    return (1.0 - prob_normal).astype(np.float32)


# =============================================================================
# MODEL 4 — ISOLATION FOREST
# =============================================================================

def train_iso(X_tr, contamination=0.05):
    """
    Semi-supervised IsolationForest.

    contamination: fraction of anomalies expected at inference.
    Pass the real attack ratio from the dataset (n_attack / n_total)
    instead of the default 0.05 assumption.  This sets the internal
    decision threshold of IsolationForest correctly, improving recall
    on the actual attack distribution without retraining the trees.
    """
    print("\n" + "="*70)
    print("MODEL 4 — ISOLATION FOREST")
    print("="*70)
    # Clip to valid sklearn range (0, 0.5]
    contamination = float(np.clip(contamination, 0.001, 0.499))
    print(f"  Contamination = {contamination:.4f}  "
          f"({'dataset ratio' if contamination != 0.05 else 'default'})")
    iso = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=contamination,
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
        W_MIN = 0.05
        W_MAX = 0.40

        res = minimize(
            neg_auc, w0,
            method="SLSQP",
            bounds=[(W_MIN, W_MAX)] * n,
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
# TFLITE CONVERSION  (AE, VAE, OCC-Net)
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
    ensemble_scores_val:    np.ndarray,   # scores on benign validation set
    ensemble_scores_attack: np.ndarray,   # scores on attack set
    percentile: float = 95.0,            # fallback if no attack data
) -> float:
    """
    Semi-supervised threshold calibration.

    When attack scores are available:
      Sweep candidate thresholds and pick the one that maximises F1
      on the combined (benign val + attack) labelled hold-out.
      This directly optimises the detection/precision trade-off instead
      of guessing based on the benign score distribution alone.

    Fallback (no attack data):
      Use p{percentile} of benign scores — same as before.
    """
    print("\n" + "="*70)
    print("THRESHOLD CALIBRATION")
    print("="*70)

    use_labelled = len(ensemble_scores_attack) > 0

    if use_labelled:
        # Build combined score + label arrays
        scores_all = np.concatenate([ensemble_scores_val, ensemble_scores_attack])
        y_all      = np.concatenate([
            np.zeros(len(ensemble_scores_val),    dtype=int),
            np.ones( len(ensemble_scores_attack), dtype=int),
        ])

        # Sweep 200 candidate thresholds between p5 and p99 of all scores
        candidates = np.linspace(
            np.percentile(scores_all, 5),
            np.percentile(scores_all, 99),
            200,
        )

        best_f1, best_thr, best_rec, best_prec = 0.0, candidates[0], 0.0, 0.0
        results_table = []
        for thr in candidates:
            y_pred = (scores_all >= thr).astype(int)
            tp = int(((y_pred == 1) & (y_all == 1)).sum())
            fp = int(((y_pred == 1) & (y_all == 0)).sum())
            fn = int(((y_pred == 0) & (y_all == 1)).sum())
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            results_table.append((thr, f1, rec, prec))
            if f1 > best_f1:
                best_f1, best_thr, best_rec, best_prec = f1, thr, rec, prec

        threshold = float(best_thr)

        print(f"  Method     : Optimal F1 on labelled hold-out")
        print(f"  Candidates : 200 thresholds swept")
        print(f"  ✅ Best F1 = {best_f1:.4f}  at threshold = {threshold:.6f}")
        print(f"     Detection Rate = {best_rec*100:.2f}%   "
              f"Precision = {best_prec*100:.2f}%")

        # Also show what p95 would have given for comparison
        p95_thr = float(np.percentile(ensemble_scores_val, percentile))
        y_p95   = (scores_all >= p95_thr).astype(int)
        tp95    = int(((y_p95==1)&(y_all==1)).sum())
        fn95    = int(((y_p95==0)&(y_all==1)).sum())
        fp95    = int(((y_p95==1)&(y_all==0)).sum())
        rec95   = tp95/(tp95+fn95+1e-9)
        pr95    = tp95/(tp95+fp95+1e-9)
        f95     = 2*pr95*rec95/(pr95+rec95+1e-9)
        print(f"\n  p95 baseline: threshold={p95_thr:.6f}  "
              f"F1={f95:.4f}  DR={rec95*100:.2f}%  Prec={pr95*100:.2f}%")
        print(f"  F1 improvement from optimal threshold: "
              f"{(best_f1-f95)*100:+.2f} pp")

        method = "optimal_f1"
    else:
        # Fallback: percentile of benign scores
        threshold = float(np.percentile(ensemble_scores_val, percentile))
        method    = f"p{percentile}_benign"
        print(f"  ⚠ No attack data — using p{percentile} fallback")
        print(f"  Benign score statistics:")
        for p in [50, 75, 90, 92, 95, 97, 99]:
            marker = " ← selected" if p == percentile else ""
            print(f"    p{p:<3} = {np.percentile(ensemble_scores_val, p):.6f}{marker}")
        print(f"\n  ✅ Threshold = {threshold:.6f}  (~{100-percentile:.0f}% FPR)")

    joblib.dump({
        "threshold":    threshold,
        "method":       method,
        "percentile":   percentile,
        "benign_mean":  float(ensemble_scores_val.mean()),
        "benign_std":   float(ensemble_scores_val.std()),
        "timestamp":    TIMESTAMP,
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
    print("MULTI-MODEL ENSEMBLE ANOMALY DETECTOR (Semi-supervised)")
    print("AE + VAE : trained on BENIGN only  (zero-day capable)")
    print("OCC-Net  : trained on BENIGN + real attacks")
    print("IsoForest: contamination = dataset attack ratio")
    print("Threshold: optimal F1 on labelled hold-out")
    print("="*70)

    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    X_b, X_a, attack_labels = load_data(dataset_dir)

    # Real contamination ratio for IsolationForest
    n_total = len(X_b) + len(X_a)
    real_contamination = len(X_a) / n_total if n_total > 0 else 0.05
    print(f"[data] Real contamination ratio: {real_contamination:.4f} "
          f"({len(X_a):,} attacks / {n_total:,} total)")

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

    # OCC-Net  (semi-supervised: real attacks as negatives)
    occ_model = train_occ(X_tr, X_val, dim, epochs, batch_size,
                          X_attack=X_a_sc if len(X_a_sc) > 0 else None)
    _to_tflite(occ_model, OCC_TFLITE_PATH, X_tr[:500])

    # Isolation Forest (real contamination from dataset)
    iso_model = train_iso(X_tr, contamination=real_contamination)

    # ── 3. Compute raw scores on val (benign) + attacks ───────────────────────
    print("\n[scores] Computing raw scores on val + attack sets ...")
    raw_val = {
        "AE":         ae_scores(ae_model,  X_val),
        "VAE":        vae_scores(vae_model, X_val),
        "OCC":        occ_scores(occ_model, X_val),
        "IsoForest":  iso_scores(iso_model, X_val),
    }
    raw_att = {
        "AE":         ae_scores(ae_model,  X_a_sc),
        "VAE":        vae_scores(vae_model, X_a_sc),
        "OCC":        occ_scores(occ_model, X_a_sc),
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

    # ── 7. Calibrate threshold (optimal F1 using labelled hold-out) ─────────
    threshold = calibrate_threshold(ens_val, ens_att, threshold_p)

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
        f.write(f"Models         : AE + VAE + OCC-Net + IsolationForest\n")
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
    print(f"  OCC TFLite   : {OCC_TFLITE_PATH}")
    print(f"  IsoForest    : {ISO_PATH}")
    print(f"  Weights      : {WEIGHTS_PATH}")
    print(f"  Scaler       : {SCALER_PATH}")
    print(f"  Threshold    : {THRESHOLD_PATH}")
    print("="*70 + "\n")

    return {"ae": ae_model, "vae": vae_model, "occ": occ_model,
            "iso": iso_model, "weights": weights, "threshold": threshold}


if __name__ == "__main__":
    if not DATASET_DIR.exists():
        print(f"⚠ Dataset not found: {DATASET_DIR.resolve()}")
        sys.exit(1)
    main()