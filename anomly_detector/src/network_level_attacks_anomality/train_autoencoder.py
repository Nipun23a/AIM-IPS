"""
train_autoencoder.py
─────────────────────
Deep-Learning Anomaly Detector for Network-Level Attack Detection.

Strategy — train on BENIGN traffic ONLY.
  The autoencoder learns to reconstruct normal flows perfectly.
  Attack flows look "foreign" → high reconstruction error → anomaly score.

  This is a pure unsupervised / one-class approach:
    • No attack labels needed during training
    • Generalises to zero-day attacks (never seen during training)
    • Replaces the TCN binary classifier

Architecture:
    Input (16 features)
      → Encoder: Dense(64) → Dense(32) → Dense(16) → bottleneck Dense(8)
      → Decoder: Dense(16) → Dense(32) → Dense(64) → Output(16)
    Loss: MSE reconstruction error
    Anomaly score: per-sample MSE (higher = more anomalous)

Threshold:
    Set at the 99th percentile of reconstruction errors on the
    BENIGN validation set, so ~1% false-positive rate on clean traffic.

Output layout (mirrors TCN trainer):
    models/anomly_detector/network_level_attacks_anomality/autoencoder/
        models/
            autoencoder_fp32.tflite     ← main inference file
            autoencoder_int8.tflite     ← quantised
            best_autoencoder_<ts>.keras ← Keras checkpoint
        logs/
            training_log_<ts>.csv
            training_summary_<ts>.txt
        plots/
            training_history_<ts>.png
            threshold_distribution_<ts>.png
            roc_curve_<ts>.png
    models/anomly_detector/network_level_attacks_anomality/
        scaler.pkl          ← StandardScaler (shared with other detectors)
        feature_info.pkl    ← feature list + metadata
        ae_threshold.pkl    ← anomaly threshold (MSE cutoff)

Run from project root:
    python train_autoencoder.py
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)

# ── GPU setup ─────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU enabled: {gpus}")
    except RuntimeError as e:
        print(f"⚠️  GPU setup error: {e}")
else:
    print("⚠️  No GPU found — using CPU")

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[0] / "models" / "anomly_detector" / "network_level_attacks_anomality"
DATASET_DIR = Path(__file__).resolve().parents[0] / "data_collector" / "data_sets" / "cicids"
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")

AE_DIR      = BASE_DIR / "autoencoder"
PLOTS_DIR   = AE_DIR   / "plots"
MODELS_DIR  = AE_DIR   / "models"
LOGS_DIR    = AE_DIR   / "logs"

SCALER_PATH    = BASE_DIR / "scaler.pkl"
FEAT_INFO_PATH = BASE_DIR / "feature_info.pkl"
THRESHOLD_PATH = BASE_DIR / "ae_threshold.pkl"

TFLITE_FP32_PATH = MODELS_DIR / "autoencoder_fp32.tflite"
TFLITE_INT8_PATH = MODELS_DIR / "autoencoder_int8.tflite"

for d in [BASE_DIR, AE_DIR, PLOTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Autoencoder outputs → {AE_DIR}")


# ── Feature list (must match pipeline/network_level/feature.py) ───────────────
THREAT_FEATURES = [
    "flow duration",
    "total fwd packets",
    "total backward packets",
    "total length of fwd packets",
    "total length of bwd packets",
    "fwd packet length mean",
    "bwd packet length mean",
    "flow bytes/s",
    "flow packets/s",
    "syn flag count",
    "ack flag count",
    "psh flag count",
    "packet length mean",
    "packet length std",
    "idle mean",
    "idle std",
]
N_FEATURES = len(THREAT_FEATURES)

# Labels that count as "attack" when evaluating the anomaly detector
# (everything that is NOT benign at network layer)
ATTACK_LABELS = {"ddos", "portscan", "bot"}


# =============================================================================
# DATA LOADING
# =============================================================================

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df


def _raw_label_to_class(raw: str) -> str:
    """Map raw CICIDS label → 'benign' | 'ddos' | 'portscan' | 'bot' | None."""
    s = raw.strip().lower()
    if s == "benign":
        return "benign"
    if "ddos" in s or "dos" in s:
        return "ddos"
    if "portscan" in s or "port scan" in s:
        return "portscan"
    if s == "bot":
        return "bot"
    # application-layer attacks — skip
    return None


def load_network_flows(dataset_dir: Path):
    """
    Load all CICIDS CSVs.  Returns:
        X_benign  : np.ndarray  — BENIGN flows only (for training)
        X_attack  : np.ndarray  — attack flows (for threshold evaluation)
        y_attack  : np.ndarray  — 1 per row (all are attacks)
        features  : list[str]
    """
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {dataset_dir.resolve()}")

    print(f"[data] Found {len(csv_files)} CSV file(s)")

    LABEL_CANDIDATES = ["label", "attack_cat", "attack_category", "attack", "class"]

    benign_rows = []
    attack_rows = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"[data] ⚠ Could not read {csv_path.name}: {e}")
            continue

        df = _normalise_cols(df)

        # Find label column
        label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
        if label_col is None:
            print(f"[data] ⚠ No label column in {csv_path.name} — skipping")
            continue

        # Check features present
        missing = [f for f in THREAT_FEATURES if f not in df.columns]
        if missing:
            print(f"[data] ⚠ {csv_path.name} missing {len(missing)} features — skipping")
            continue

        # Clean features
        feat_df = df[THREAT_FEATURES].copy()
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        mask = ~feat_df.isnull().any(axis=1)
        feat_df = feat_df[mask].astype(float)
        raw_labels = df[label_col][mask].astype(str)

        counts = {}
        for raw_lbl in raw_labels.unique():
            cls = _raw_label_to_class(raw_lbl)
            if cls is None:
                continue   # skip application-layer attacks
            rows = feat_df[raw_labels == raw_lbl]
            counts[cls] = len(rows)
            if cls == "benign":
                benign_rows.append(rows)
            else:
                attack_rows.append(rows)

        print(f"[data] {csv_path.name}: {counts}")

    if not benign_rows:
        raise RuntimeError("No BENIGN flows found in dataset.")

    X_benign = pd.concat(benign_rows, ignore_index=True).values.astype(np.float32)
    X_attack = (
        pd.concat(attack_rows, ignore_index=True).values.astype(np.float32)
        if attack_rows else np.empty((0, N_FEATURES), dtype=np.float32)
    )
    y_attack = np.ones(len(X_attack), dtype=int)

    print(f"\n[data] BENIGN rows : {len(X_benign):,}")
    print(f"[data] ATTACK rows : {len(X_attack):,}")

    return X_benign, X_attack, y_attack, THREAT_FEATURES


# =============================================================================
# PREPROCESSING  (scale on BENIGN only — autoencoder sees only normal traffic)
# =============================================================================

def _clip_outliers(X: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """
    Per-feature percentile clipping to remove extreme benign outliers
    that inflate the threshold.  Fit ONLY on benign data passed in.
    """
    lo = np.percentile(X, lower, axis=0)
    hi = np.percentile(X, upper, axis=0)
    return np.clip(X, lo, hi)


def preprocess(X_benign: np.ndarray, X_attack: np.ndarray):
    """
    Fit StandardScaler on BENIGN only.

    Outlier clipping is applied BEFORE scaling:
      - The CICIDS benign set contains a small number of extreme flows
        (e.g. flow duration or byte count in the millions) that push
        StandardScaler means/stds way up, causing the autoencoder to
        reconstruct those outlier directions poorly even for normal flows.
      - Clipping to [p1, p99] per feature removes these without losing
        the bulk of the benign distribution.
      - The same clip bounds are saved alongside the scaler so the
        detector applies identical preprocessing at inference time.
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING  (fit scaler on BENIGN only + outlier clipping)")
    print("=" * 70)

    # Compute clip bounds from benign data ONLY
    clip_lo = np.percentile(X_benign, 1.0, axis=0).astype(np.float32)
    clip_hi = np.percentile(X_benign, 99.0, axis=0).astype(np.float32)

    X_benign_clipped = np.clip(X_benign, clip_lo, clip_hi).astype(np.float32)
    print(f"  Outlier clipping  [p1, p99] applied to {X_benign.shape[1]} features")

    scaler = StandardScaler()
    X_benign_sc = scaler.fit_transform(X_benign_clipped).astype(np.float32)

    # Attack: apply same clip then scale (attack values can exceed bounds — that's fine,
    # they will just be far from the benign distribution after scaling)
    if len(X_attack) > 0:
        X_attack_clipped = np.clip(X_attack, clip_lo, clip_hi).astype(np.float32)
        X_attack_sc = scaler.transform(X_attack_clipped).astype(np.float32)
    else:
        X_attack_sc = X_attack

    # Save scaler + clip bounds together so the detector can reproduce preprocessing
    joblib.dump({
        "scaler":   scaler,
        "clip_lo":  clip_lo,
        "clip_hi":  clip_hi,
    }, SCALER_PATH)
    print(f"  Scaler + clip bounds saved → {SCALER_PATH}")

    # Train/val split on BENIGN
    X_tr, X_val = train_test_split(X_benign_sc, test_size=0.15, random_state=42)

    print(f"  Train (benign): {X_tr.shape}")
    print(f"  Val   (benign): {X_val.shape}")
    print(f"  Attack (eval) : {X_attack_sc.shape}")

    return X_tr, X_val, X_attack_sc, scaler


# =============================================================================
# MODEL  — Deep Autoencoder
# =============================================================================

def build_autoencoder(input_dim: int) -> tf.keras.Model:
    """
    Deep autoencoder:
      Encoder: 16 → 64 → 32 → 16 → 8  (bottleneck)
      Decoder: 8  → 16 → 32 → 64 → 16

    BatchNorm + Dropout prevent trivial identity mapping.
    L2 regularisation on all Dense weights.
    """
    reg = regularizers.l2(1e-4)

    # ── Encoder ───────────────────────────────────────────────
    inp = layers.Input(shape=(input_dim,), name="input")

    x = layers.Dense(64, kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(32, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(16, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    bottleneck = layers.Dense(8, activation="relu",
                               kernel_regularizer=reg, name="bottleneck")(x)

    # ── Decoder ───────────────────────────────────────────────
    x = layers.Dense(16, kernel_regularizer=reg)(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(32, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(64, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    output = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    model = models.Model(inp, output, name="network_autoencoder")
    print(f"\n✅ Autoencoder built  params={model.count_params():,}")
    model.summary()
    return model


# =============================================================================
# TRAINING
# =============================================================================

def train(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    input_dim: int,
    epochs:    int = 100,
    batch_size: int = 256,
):
    print("\n" + "=" * 70)
    print("TRAINING AUTOENCODER  (BENIGN only — reconstruction objective)")
    print("=" * 70)

    model = build_autoencoder(input_dim)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    ckpt_path = MODELS_DIR / f"best_autoencoder_{TIMESTAMP}.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            str(ckpt_path), monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        CSVLogger(str(LOGS_DIR / f"training_log_{TIMESTAMP}.csv")),
    ]

    # Autoencoder: input == target (reconstruct the input)
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    hist_path = LOGS_DIR / f"history_{TIMESTAMP}.pkl"
    joblib.dump(history.history, hist_path)
    print(f"\n💾 History saved → {hist_path}")

    return model, history


# =============================================================================
# THRESHOLD CALIBRATION
# =============================================================================

def calibrate_threshold(
    model:        tf.keras.Model,
    X_val_benign: np.ndarray,
    percentile:   float = 95.0,
) -> float:
    """
    Compute reconstruction MSE on the BENIGN validation set.
    Set the anomaly threshold at `percentile`-th percentile.

    Why p95 instead of p99?
    ─────────────────────────────────────────────────────────
    Your BENIGN CICIDS data contains a handful of extreme flows
    (max MSE = 473 vs mean = 0.04).  At p99 those outliers drag
    the threshold up to ~0.51, which is higher than most attack
    MSEs, so attacks slip through (43% detection rate).

    p95 cuts the threshold dramatically while the FPR stays low
    because 95% of benign traffic still falls below it.

    Percentile guide:
        p99 → ~1%  FPR  (very conservative — use if FP cost is high)
        p95 → ~5%  FPR  (balanced — recommended default)
        p90 → ~10% FPR  (aggressive — use if missing attacks is costly)

    The table below is printed so you can choose manually.
    """
    print("\n" + "=" * 70)
    print(f"THRESHOLD CALIBRATION  (p{percentile} of BENIGN MSE)")
    print("=" * 70)

    recon = model.predict(X_val_benign, verbose=0)
    mse   = np.mean((X_val_benign - recon) ** 2, axis=1)

    # Print the full percentile table so you can compare options
    print(f"  BENIGN MSE statistics:")
    print(f"    min    = {mse.min():.6f}")
    print(f"    mean   = {mse.mean():.6f}")
    print(f"    median = {np.median(mse):.6f}")
    print(f"    std    = {mse.std():.6f}")
    print(f"    max    = {mse.max():.6f}")
    print()
    print(f"  Percentile table (choose threshold that balances FPR vs detection):")
    print(f"  {'Percentile':>12}  {'Threshold':>12}  {'Expected FPR':>14}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*14}")
    for p in [80, 85, 90, 92, 95, 97, 99]:
        t = np.percentile(mse, p)
        fpr_est = 100.0 - p
        marker = " ← selected" if p == percentile else ""
        print(f"  {p:>12}  {t:>12.6f}  {fpr_est:>13.1f}%{marker}")

    threshold = float(np.percentile(mse, percentile))

    print(f"\n  ✅ Threshold = {threshold:.6f}  "
          f"(flows with MSE > this are flagged as anomalies)")
    print(f"     Expected FPR on clean traffic : ~{100-percentile:.0f}%")

    joblib.dump({
        "threshold":       threshold,
        "percentile":      percentile,
        "benign_mse_mean": float(mse.mean()),
        "benign_mse_std":  float(mse.std()),
        "benign_mse_p50":  float(np.percentile(mse, 50)),
        "benign_mse_p95":  float(np.percentile(mse, 95)),
        "benign_mse_p99":  float(np.percentile(mse, 99)),
        "timestamp":       TIMESTAMP,
    }, THRESHOLD_PATH)
    print(f"  💾 Threshold saved → {THRESHOLD_PATH}")

    return threshold


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    model:       tf.keras.Model,
    tflite_path: Path,
    X_val_benign: np.ndarray,
    X_attack:    np.ndarray,
    y_attack:    np.ndarray,
    threshold:   float,
) -> dict:
    """
    Evaluate on a combined BENIGN (val) + ATTACK set.
    Reports FPR, FNR, F1, AUC-ROC, and confusion matrix.
    """
    print("\n" + "=" * 70)
    print("EVALUATION — Keras vs TFLite")
    print("=" * 70)

    if len(X_attack) == 0:
        print("  ⚠ No attack rows in dataset — skipping binary eval")
        return {}

    # Combine benign val + attacks for evaluation
    # Cap benign samples to at most 3× the attack count for balanced display
    n_attack = len(X_attack)
    n_benign_eval = min(len(X_val_benign), n_attack * 3)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_val_benign), size=n_benign_eval, replace=False)

    X_eval = np.vstack([X_val_benign[idx], X_attack])
    y_eval = np.concatenate([
        np.zeros(n_benign_eval, dtype=int),
        np.ones(n_attack,       dtype=int),
    ])

    results = {}

    for name, get_recon in [
        ("Keras",       lambda: model.predict(X_eval, verbose=0)),
        ("TFLite FP32", lambda: _tflite_reconstruct(tflite_path, X_eval)),
    ]:
        t0 = time.time()
        recon  = get_recon()
        elapsed = time.time() - t0

        mse_scores = np.mean((X_eval - recon) ** 2, axis=1)
        y_pred     = (mse_scores > threshold).astype(int)

        acc  = accuracy_score(y_eval, y_pred)
        f1   = f1_score(y_eval, y_pred, zero_division=0)
        auc  = roc_auc_score(y_eval, mse_scores)
        cm   = confusion_matrix(y_eval, y_pred)
        TN, FP, FN, TP = cm.ravel()
        fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        fnr  = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        det  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        ms_per = elapsed / len(X_eval) * 1000

        print(f"\n── {name} ({ms_per:.3f} ms/sample) ──")
        print(f"   Accuracy:            {acc*100:.4f}%")
        print(f"   F1-Score:            {f1*100:.4f}%")
        print(f"   AUC-ROC:             {auc*100:.4f}%")
        print(f"   False Positive Rate: {fpr*100:.4f}%")
        print(f"   False Negative Rate: {fnr*100:.4f}%")
        print(f"   Detection Rate:      {det*100:.4f}%")
        print(f"   Threshold:           {threshold:.6f}")
        print(classification_report(y_eval, y_pred,
                                    target_names=["Benign", "Attack"], digits=4))

        results[name] = dict(
            accuracy=acc, f1=f1, auc=auc,
            fpr=fpr, fnr=fnr, detection_rate=det,
            confusion_matrix=cm,
            mse_scores=mse_scores,
            y_eval=y_eval,
            y_pred=y_pred,
            threshold=threshold,
        )

    metrics_path = LOGS_DIR / f"eval_metrics_{TIMESTAMP}.pkl"
    joblib.dump(results, metrics_path)
    print(f"\n💾 Eval metrics saved → {metrics_path}")
    return results


def _tflite_reconstruct(tflite_path: Path, X: np.ndarray) -> np.ndarray:
    """Run TFLite autoencoder and return reconstructed output."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    recons = []
    for i in range(len(X)):
        interpreter.set_tensor(inp_det[0]["index"],
                               X[i:i+1].astype(np.float32))
        interpreter.invoke()
        recons.append(interpreter.get_tensor(out_det[0]["index"])[0])
    return np.array(recons, dtype=np.float32)


# =============================================================================
# TFLITE CONVERSION
# =============================================================================

def convert_to_tflite(
    model:   tf.keras.Model,
    X_calib: np.ndarray,
):
    print("\n" + "=" * 70)
    print("TFLITE CONVERSION")
    print("=" * 70)

    # FP32
    print("[TFLite] FP32 ...")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = []
    tflite_fp32 = conv.convert()
    TFLITE_FP32_PATH.write_bytes(tflite_fp32)
    print(f"[TFLite] FP32 → {TFLITE_FP32_PATH}  ({len(tflite_fp32)/1024:.1f} KB)")

    # INT8
    print("[TFLite] INT8 (full-integer quantisation) ...")

    def representative_dataset():
        idx = np.random.choice(len(X_calib),
                               size=min(500, len(X_calib)), replace=False)
        for i in idx:
            yield [X_calib[i:i+1].astype(np.float32)]

    conv8 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv8.optimizations                 = [tf.lite.Optimize.DEFAULT]
    conv8.representative_dataset        = representative_dataset
    conv8.target_spec.supported_ops     = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv8.inference_input_type          = tf.float32
    conv8.inference_output_type         = tf.float32

    tflite_int8 = conv8.convert()
    TFLITE_INT8_PATH.write_bytes(tflite_int8)
    fp32_kb = len(tflite_fp32) / 1024
    int8_kb = len(tflite_int8) / 1024
    print(f"[TFLite] INT8 → {TFLITE_INT8_PATH}  ({int8_kb:.1f} KB)")
    print(f"[TFLite] Size reduction: {fp32_kb:.1f} KB → {int8_kb:.1f} KB "
          f"({(1-int8_kb/fp32_kb)*100:.0f}% smaller)")

    return TFLITE_FP32_PATH, TFLITE_INT8_PATH


# =============================================================================
# PLOTS
# =============================================================================

def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Autoencoder — Training History", fontsize=14)

    for ax, key, title in [
        (axes[0], "loss", "MSE Loss"),
        (axes[1], "mae",  "MAE"),
    ]:
        ax.plot(history.history[key],         label="Train", linewidth=2)
        ax.plot(history.history[f"val_{key}"], label="Val",   linewidth=2)
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"training_history_{TIMESTAMP}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close()
    print(f"📊 Training plot → {out}")


def plot_threshold_distribution(
    model:       tf.keras.Model,
    X_val_benign: np.ndarray,
    X_attack:    np.ndarray,
    threshold:   float,
):
    """Plot MSE distributions for benign vs attack — shows separability."""
    recon_b = model.predict(X_val_benign, verbose=0)
    mse_b   = np.mean((X_val_benign - recon_b) ** 2, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Reconstruction Error Distribution — Benign vs Attack", fontsize=13)

    # Clip for readability (top 1% can be extreme outliers)
    clip = np.percentile(mse_b, 99.5) * 5
    bins = np.linspace(0, clip, 120)

    ax.hist(np.clip(mse_b, 0, clip), bins=bins,
            alpha=0.6, color="steelblue", label=f"Benign (n={len(mse_b):,})",
            density=True)

    if len(X_attack) > 0:
        recon_a = model.predict(X_attack, verbose=0)
        mse_a   = np.mean((X_attack - recon_a) ** 2, axis=1)
        ax.hist(np.clip(mse_a, 0, clip), bins=bins,
                alpha=0.6, color="tomato", label=f"Attack (n={len(mse_a):,})",
                density=True)

    ax.axvline(threshold, color="black", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Reconstruction MSE"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"threshold_distribution_{TIMESTAMP}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close()
    print(f"📊 Threshold distribution plot → {out}")


def plot_roc(results: dict):
    if not results:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("ROC Curve — Autoencoder Anomaly Detector")

    for name, r in results.items():
        fpr_arr, tpr_arr, _ = roc_curve(r["y_eval"], r["mse_scores"])
        ax.plot(fpr_arr, tpr_arr, linewidth=2, label=f"{name}  AUC={r['auc']:.4f}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"roc_curve_{TIMESTAMP}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close()
    print(f"📊 ROC plot → {out}")


def plot_confusion(results: dict):
    if not results:
        return
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrix — Anomaly Detector", fontsize=13)

    for ax, name in zip(axes, names):
        cm = results[name]["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Benign", "Attack"],
                    yticklabels=["Benign", "Attack"])
        ax.set_title(name)
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")

    plt.tight_layout()
    out = PLOTS_DIR / f"confusion_matrix_{TIMESTAMP}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close()
    print(f"📊 Confusion matrix plot → {out}")


# =============================================================================
# MAIN
# =============================================================================

def main(
    dataset_dir: Path = DATASET_DIR,
    epochs:      int  = 100,
    batch_size:  int  = 256,
    threshold_percentile: float = 95.0,   # p95 → ~5% FPR, much better detection
):
    print("\n" + "=" * 70)
    print("AUTOENCODER — NETWORK ANOMALY DETECTOR")
    print("Trains on BENIGN flows only — detects attacks via reconstruction error")
    print("=" * 70)
    print(f"Dataset : {dataset_dir}")
    print(f"Epochs  : {epochs}   Batch: {batch_size}   Threshold: p{threshold_percentile}")

    # 1. Load
    X_benign, X_attack, y_attack, features = load_network_flows(dataset_dir)

    # 2. Preprocess
    X_tr, X_val, X_attack_sc, scaler = preprocess(X_benign, X_attack)

    # Save feature info
    feat_info = {"features": features, "n_features": len(features), "timestamp": TIMESTAMP}
    joblib.dump(feat_info, FEAT_INFO_PATH)
    print(f"  Feature info saved → {FEAT_INFO_PATH}")

    input_dim = X_tr.shape[1]

    # 3. Train
    keras_model, history = train(X_tr, X_val, input_dim, epochs, batch_size)

    # 4. Plot training
    plot_training(history)

    # 5. Calibrate threshold
    threshold = calibrate_threshold(keras_model, X_val, threshold_percentile)

    # 6. Distribution plot (shows benign vs attack MSE separation)
    plot_threshold_distribution(keras_model, X_val, X_attack_sc, threshold)

    # 7. Convert to TFLite
    cal_idx = np.random.choice(len(X_tr), size=min(1000, len(X_tr)), replace=False)
    fp32_path, int8_path = convert_to_tflite(keras_model, X_tr[cal_idx])

    # 8. Evaluate
    eval_results = evaluate(
        keras_model, fp32_path,
        X_val, X_attack_sc, y_attack, threshold,
    )

    # 9. Plots
    plot_roc(eval_results)
    plot_confusion(eval_results)

    # 10. Summary
    r_k = eval_results.get("Keras", {})
    summary_path = LOGS_DIR / f"training_summary_{TIMESTAMP}.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("AUTOENCODER — TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp        : {TIMESTAMP}\n")
        f.write(f"Input dim        : {input_dim}\n")
        f.write(f"Train samples    : {len(X_tr):,}  (BENIGN only)\n")
        f.write(f"Val   samples    : {len(X_val):,}  (BENIGN only)\n")
        f.write(f"Attack samples   : {len(X_attack_sc):,}\n")
        f.write(f"Threshold        : {threshold:.6f}  (p{threshold_percentile} of benign MSE)\n")
        f.write(f"FP32 TFLite      : {fp32_path}\n")
        f.write(f"INT8 TFLite      : {int8_path}\n")
        f.write("\nKeras Evaluation:\n")
        for k, label in [
            ("accuracy",       "Accuracy"),
            ("f1",             "F1-Score"),
            ("auc",            "AUC-ROC"),
            ("fpr",            "False Positive Rate"),
            ("fnr",            "False Negative Rate"),
            ("detection_rate", "Detection Rate"),
        ]:
            f.write(f"  {label:25s}: {r_k.get(k, 0)*100:.4f}%\n")
    print(f"💾 Summary → {summary_path}")

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print(f"  FP32 TFLite  : {fp32_path}")
    print(f"  INT8 TFLite  : {int8_path}")
    print(f"  Scaler       : {SCALER_PATH}")
    print(f"  Threshold    : {THRESHOLD_PATH}")
    print(f"  Features     : {FEAT_INFO_PATH}")
    print(f"  Plots        : {PLOTS_DIR}")
    print(f"  Logs         : {LOGS_DIR}")
    print("=" * 70 + "\n")

    return keras_model, eval_results, fp32_path, int8_path, threshold


if __name__ == "__main__":
    if not DATASET_DIR.exists():
        print(f"⚠️  Dataset not found: {DATASET_DIR.resolve()}")
        print("Update DATASET_DIR at the top of this script.")
        sys.exit(1)

    main()