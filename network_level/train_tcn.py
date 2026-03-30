"""
train_tcn.py
============
Trains a Temporal Convolutional Network (TCN) as an anomaly detector
for the Network Layer IPS — detecting zero-day / novel network attacks.

Architecture:
  - Autoencoder-style TCN: learns to reconstruct normal (benign) traffic
  - High reconstruction error = anomaly = potential zero-day attack
  - PCA + Mahalanobis distance fusion gate (as per architecture diagram)
  - Exported to TFLite for optimized inference

Outputs:
  - tcn_model/                   → SavedModel format
  - tcn_model.tflite             → TFLite optimized model
  - tcn_scaler.pkl               → StandardScaler for feature normalization
  - tcn_pca.pkl                  → PCA transformer (32 → 16 dims)
  - tcn_threshold.json           → Anomaly threshold + Mahalanobis params
  - tcn_training_report.json     → Training metrics and config
  - tcn_reconstruction_error.png → Error distribution plot

Usage:
  python train_tcn.py --data ./mapped_data/nb15_mapped.csv
  python train_tcn.py --data ./mapped_data/nb15_mapped.csv --epochs 100
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------

UNIFIED_FEATURES = [
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes", "total_bwd_bytes", "flow_bytes_per_sec",
    "flow_pkts_per_sec", "fwd_pkts_per_sec", "bwd_pkts_per_sec",
    "pkt_len_mean", "pkt_len_std", "pkt_len_max", "pkt_len_min",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max",
    "fwd_iat_mean", "bwd_iat_mean",
    "fin_flag_count", "syn_flag_count", "rst_flag_count",
    "psh_flag_count", "ack_flag_count",
    "init_win_fwd", "init_win_bwd",
    "fwd_ttl_mean", "bwd_ttl_mean",
    "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "fwd_pkt_len_std", "bwd_pkt_len_std",
    "down_up_ratio",
]

N_FEATURES      = len(UNIFIED_FEATURES)   # 32
SEQUENCE_LEN    = 10    # 10 consecutive flows per window
PCA_COMPONENTS  = 16    # compress 32 → 16 before TCN

# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------

def load_data(benign_path: str, novel_path: str):
    """
    Load CICIDS2017 benign and novel CSVs produced by dataset_mapper.py.

    benign_path  — cicids_benign.csv  (no label col, features only)
    novel_path   — cicids_novel.csv   (unified_label = "novel")

    TCN trains ONLY on benign traffic (autoencoder learns normal patterns).
    Novel flows are used for threshold calibration and evaluation only.
    """
    print(f"\n[DATA] Loading benign: {benign_path}")
    benign_df = pd.read_csv(benign_path, low_memory=False)

    missing = [f for f in UNIFIED_FEATURES if f not in benign_df.columns]
    if missing:
        raise ValueError(f"Missing features in benign file: {missing}\n"
                         f"Run dataset_mapper.py first.")

    print(f"  Benign rows: {len(benign_df):,}")
    X_benign = benign_df[UNIFIED_FEATURES].values.astype(np.float32)

    print(f"\n[DATA] Loading novel:  {novel_path}")
    novel_df = pd.read_csv(novel_path, low_memory=False)

    missing = [f for f in UNIFIED_FEATURES if f not in novel_df.columns]
    if missing:
        raise ValueError(f"Missing features in novel file: {missing}\n"
                         f"Run dataset_mapper.py first.")

    print(f"  Novel rows:  {len(novel_df):,}")
    if "unified_label" in novel_df.columns:
        print(f"  {novel_df['unified_label'].value_counts().to_dict()}")

    X_attack = novel_df[UNIFIED_FEATURES].values.astype(np.float32)
    y_attack_labels = (
        novel_df["unified_label"].values
        if "unified_label" in novel_df.columns
        else np.array(["novel"] * len(novel_df))
    )

    return X_benign, X_attack, y_attack_labels


# ---------------------------------------------------------------------------
# 3. PREPROCESSING: SCALER + PCA
# ---------------------------------------------------------------------------

def fit_preprocessors(X_benign_train: np.ndarray):
    """
    Fit StandardScaler and PCA on benign training data only.
    This is critical — the TCN learns what 'normal' looks like.
    """
    print("\n[PREPROCESS] Fitting scaler and PCA on benign traffic...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_benign_train)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {N_FEATURES} → {PCA_COMPONENTS} dims")
    print(f"  Explained variance: {explained:.3f} ({explained*100:.1f}%)")

    return scaler, pca, X_pca


def preprocess(X: np.ndarray, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    """Apply fitted scaler and PCA to any data."""
    return pca.transform(scaler.transform(X)).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. SEQUENCE BUILDER
# ---------------------------------------------------------------------------

def build_sequences(X: np.ndarray, seq_len: int = SEQUENCE_LEN) -> np.ndarray:
    """
    Build sliding window sequences for TCN input.
    
    TCN expects: (batch, timesteps, features)
    
    Each sequence = seq_len consecutive flow records.
    This captures temporal patterns across multiple flows from same context.
    
    Args:
        X: (n_samples, n_features) array
        seq_len: number of timesteps per sequence
        
    Returns:
        sequences: (n_samples - seq_len + 1, seq_len, n_features)
    """
    n = len(X)
    sequences = np.stack(
        [X[i:i + seq_len] for i in range(n - seq_len + 1)],
        axis=0
    )
    return sequences.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. TCN ARCHITECTURE
# ---------------------------------------------------------------------------

def residual_block(x, filters: int, kernel_size: int,
                   dilation_rate: int, dropout_rate: float = 0.1):
    """
    TCN Residual Block:
      - Dilated causal convolution
      - Layer normalization
      - Dropout
      - Residual connection
    """
    # Main path
    conv1 = layers.Conv1D(
        filters, kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation=None,
    )(x)
    conv1 = layers.LayerNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)
    conv1 = layers.SpatialDropout1D(dropout_rate)(conv1)

    conv2 = layers.Conv1D(
        filters, kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation=None,
    )(conv1)
    conv2 = layers.LayerNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)
    conv2 = layers.SpatialDropout1D(dropout_rate)(conv2)

    # Residual connection — match dimensions if needed
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding="same")(x)

    return layers.Add()([x, conv2])


def build_tcn_autoencoder(
    seq_len: int = SEQUENCE_LEN,
    n_features: int = PCA_COMPONENTS,
    filters: int = 64,
    kernel_size: int = 3,
    n_blocks: int = 4,
    dropout_rate: float = 0.1,
    bottleneck_dim: int = 8,
) -> Model:
    """
    TCN Autoencoder for anomaly detection.
    
    Encoder: TCN residual blocks → bottleneck
    Decoder: Upsampling → TCN residual blocks → reconstruct input
    
    Anomaly score = MSE reconstruction error
    High error = pattern not seen during benign training = potential attack
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="flow_sequence")

    # ---- ENCODER ----
    x = inputs
    dilation_rates = [2 ** i for i in range(n_blocks)]  # 1, 2, 4, 8

    for i, d in enumerate(dilation_rates):
        x = residual_block(
            x, filters=filters,
            kernel_size=kernel_size,
            dilation_rate=d,
            dropout_rate=dropout_rate,
        )

    # Bottleneck — compressed representation
    bottleneck = layers.GlobalAveragePooling1D(name="bottleneck_pool")(x)
    bottleneck = layers.Dense(bottleneck_dim, activation="relu",
                               name="bottleneck")(bottleneck)

    # ---- DECODER ----
    # Expand bottleneck back to sequence
    decoded = layers.Dense(seq_len * filters // 2,
                           activation="relu")(bottleneck)
    decoded = layers.Reshape((seq_len, filters // 2))(decoded)

    for d in reversed(dilation_rates):
        decoded = residual_block(
            decoded, filters=filters // 2,
            kernel_size=kernel_size,
            dilation_rate=d,
            dropout_rate=dropout_rate,
        )

    # Reconstruct original feature space
    outputs = layers.Conv1D(
        n_features, 1,
        activation="linear",
        name="reconstruction"
    )(decoded)

    model = Model(inputs, outputs, name="TCN_Autoencoder")
    return model


# ---------------------------------------------------------------------------
# 6. MAHALANOBIS FUSION GATE
# ---------------------------------------------------------------------------

class MahalanobisFusionGate:
    """
    PCA + Mahalanobis Distance fusion gate (as shown in architecture diagram).
    
    Computes Mahalanobis distance of a sample from the benign distribution
    in PCA-reduced space. Combined with TCN reconstruction error for final score.
    
    High Mahalanobis distance = far from normal traffic = suspicious
    """

    def __init__(self):
        self.mean = None
        self.inv_cov = None
        self.threshold = None

    def fit(self, X_benign_pca: np.ndarray):
        """Fit on benign PCA-transformed data."""
        self.mean = np.mean(X_benign_pca, axis=0)
        cov = np.cov(X_benign_pca, rowvar=False)
        # Regularize covariance matrix to avoid singularity
        cov += np.eye(cov.shape[0]) * 1e-6
        self.inv_cov = np.linalg.inv(cov)
        print(f"  Mahalanobis gate fitted on {len(X_benign_pca):,} benign samples")

    def score(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance for each sample.
        Returns normalized score 0.0–1.0.
        """
        diff = X_pca - self.mean
        # Vectorized Mahalanobis: sqrt(diff @ inv_cov @ diff.T)
        left = diff @ self.inv_cov
        dist = np.sqrt(np.einsum("ij,ij->i", left, diff))
        # Normalize using fitted threshold
        if self.threshold:
            return np.clip(dist / self.threshold, 0, 1)
        return dist

    def calibrate_threshold(self, X_benign_pca: np.ndarray,
                            percentile: float = 99.0):
        """Set threshold at 99th percentile of benign distances."""
        scores = self.score(X_benign_pca)
        self.threshold = float(np.percentile(scores, percentile))
        # Re-normalize with threshold
        scores_norm = np.clip(scores / self.threshold, 0, 1)
        print(f"  Mahalanobis threshold (p{percentile}): {self.threshold:.4f}")
        return scores_norm

    def to_dict(self) -> dict:
        return {
            "mean":      self.mean.tolist(),
            "inv_cov":   self.inv_cov.tolist(),
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, d: dict):
        gate = cls()
        gate.mean = np.array(d["mean"])
        gate.inv_cov = np.array(d["inv_cov"])
        gate.threshold = d["threshold"]
        return gate


# ---------------------------------------------------------------------------
# 7. ANOMALY SCORING
# ---------------------------------------------------------------------------

def compute_reconstruction_error(model: Model,
                                  sequences: np.ndarray,
                                  batch_size: int = 512) -> np.ndarray:
    """
    Compute per-sequence MSE reconstruction error.
    This is the core anomaly score from the TCN autoencoder.
    """
    reconstructed = model.predict(sequences, batch_size=batch_size, verbose=0)
    errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
    return errors


def fuse_scores(tcn_errors: np.ndarray,
                mahal_scores: np.ndarray,
                tcn_norm_params: dict,
                tcn_weight: float = 0.6,
                mahal_weight: float = 0.4) -> np.ndarray:
    """
    Weighted fusion of TCN reconstruction error and Mahalanobis distance.
    
    CRITICAL: TCN errors must be normalized using benign-ONLY stats.
    If we normalize across benign+attack together, attack errors get
    compressed into the same range as benign → ROC-AUC collapses.
    
    tcn_norm_params: {"mean": float, "std": float, "p99": float}
                     fitted on benign validation errors only.
    """
    # Normalize using benign-fitted p99 as ceiling
    p99 = tcn_norm_params["p99"]
    tcn_norm = np.clip(tcn_errors / (p99 + 1e-8), 0, 2.0)  # allow >1 for attacks

    fused = (tcn_weight * tcn_norm) + (mahal_weight * mahal_scores)
    return np.clip(fused, 0, 2.0)  # keep >1 range so attacks separate clearly


# ---------------------------------------------------------------------------
# 8. THRESHOLD CALIBRATION
# ---------------------------------------------------------------------------

def calibrate_anomaly_threshold(
    benign_scores: np.ndarray,
    attack_scores: np.ndarray,
    target_fpr: float = 0.05,
) -> dict:
    """
    Find anomaly threshold that achieves target false positive rate.
    
    For zero-day detection:
    - FPR <= 5% (don't flag too much benign traffic)
    - Maximize TPR (catch as many novel attacks as possible)
    
    Returns threshold config for tcn_threshold.json
    """
    print(f"\n[THRESHOLD] Calibrating anomaly threshold (target FPR={target_fpr})...")

    all_scores = np.concatenate([benign_scores, attack_scores])
    all_labels = np.concatenate([
        np.zeros(len(benign_scores)),
        np.ones(len(attack_scores))
    ])

    # ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auc = roc_auc_score(all_labels, all_scores)
    print(f"  ROC-AUC: {auc:.4f}")

    # Find threshold at target FPR
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(thresholds):
        idx = len(thresholds) - 1

    chosen_threshold = float(thresholds[idx])
    chosen_tpr = float(tpr[idx])
    chosen_fpr = float(fpr[idx])

    # Also compute F1 at this threshold
    preds = (all_scores >= chosen_threshold).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)

    print(f"  Threshold: {chosen_threshold:.4f}")
    print(f"  TPR (recall): {chosen_tpr:.4f}")
    print(f"  FPR:          {chosen_fpr:.4f}")
    print(f"  F1:           {f1:.4f}")
    print(f"  Precision:    {precision:.4f}")

    # Plot ROC + score distributions
    return {
        "threshold":    chosen_threshold,
        "roc_auc":      auc,
        "tpr":          chosen_tpr,
        "fpr":          chosen_fpr,
        "f1":           f1,
        "precision":    float(precision),
        "recall":       float(recall),
        "target_fpr":   target_fpr,
        # These feed into Layer 3 Score Aggregator
        "score_map": {
            "below_threshold":  "benign (score < threshold)",
            "above_threshold":  "novel_attack (score >= threshold)",
        }
    }


# ---------------------------------------------------------------------------
# 9. PLOTS
# ---------------------------------------------------------------------------

def plot_reconstruction_errors(benign_errors, attack_errors,
                                threshold: float, out_dir: Path):
    """Plot reconstruction error distributions for benign vs attack."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Distribution plot
    axes[0].hist(benign_errors, bins=80, alpha=0.6,
                 color="steelblue", label="Benign", density=True)
    axes[0].hist(attack_errors, bins=80, alpha=0.6,
                 color="crimson", label="Novel Attack", density=True)
    axes[0].axvline(threshold, color="black", linestyle="--",
                    linewidth=2, label=f"Threshold ({threshold:.3f})")
    axes[0].set_xlabel("Fused Anomaly Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("TCN Anomaly Score Distribution")
    axes[0].legend()

    # Box plot
    axes[1].boxplot(
        [benign_errors, attack_errors],
        labels=["Benign", "Novel Attack"],
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
    )
    axes[1].axhline(threshold, color="black", linestyle="--",
                    linewidth=2, label=f"Threshold ({threshold:.3f})")
    axes[1].set_ylabel("Fused Anomaly Score")
    axes[1].set_title("Score Distribution (Box Plot)")
    axes[1].legend()

    plt.tight_layout()
    path = out_dir / "tcn_reconstruction_error.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved error distribution → {path}")


# ---------------------------------------------------------------------------
# 10. TFLITE EXPORT
# ---------------------------------------------------------------------------

def export_tflite(keras_model: Model, out_dir: Path) -> Path:
    """
    Convert Keras model to TFLite for optimized inference.
    Keras 3.x requires model.export() → SavedModel → TFLite converter.
    Target inference time: < 5ms on CPU.
    """
    print("\n[TFLITE] Exporting to TFLite...")

    # Step 1: Export to SavedModel format (required for Keras 3.x)
    export_path = out_dir / "tcn_savedmodel"
    keras_model.export(str(export_path))

    # Step 2: Convert SavedModel → TFLite with float16 quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    tflite_path = out_dir / "tcn_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"  TFLite model size: {size_mb:.2f} MB")
    print(f"  Saved → {tflite_path}")
    return tflite_path


# ---------------------------------------------------------------------------
# 11. MAIN
# ---------------------------------------------------------------------------

def main(benign_path: str, novel_path: str, out_dir: str, epochs: int, batch_size: int):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Load Data ---
    X_benign, X_attack, attack_labels = load_data(benign_path, novel_path)

    # --- Train/Val/Test split on benign data ---
    X_ben_train, X_ben_temp = train_test_split(
        X_benign, test_size=0.30, random_state=42
    )
    X_ben_val, X_ben_test = train_test_split(
        X_ben_temp, test_size=0.50, random_state=42
    )

    print(f"\n  Benign Train: {len(X_ben_train):,}")
    print(f"  Benign Val:   {len(X_ben_val):,}")
    print(f"  Benign Test:  {len(X_ben_test):,}")
    print(f"  Attack Test:  {len(X_attack):,}")

    # --- Preprocessors ---
    scaler, pca, X_ben_train_pca = fit_preprocessors(X_ben_train)
    X_ben_val_pca   = preprocess(X_ben_val, scaler, pca)
    X_ben_test_pca  = preprocess(X_ben_test, scaler, pca)
    X_attack_pca    = preprocess(X_attack, scaler, pca)

    # --- Mahalanobis Fusion Gate ---
    print("\n[MAHALANOBIS] Fitting fusion gate...")
    mahal_gate = MahalanobisFusionGate()
    mahal_gate.fit(X_ben_train_pca)
    mahal_gate.calibrate_threshold(X_ben_val_pca)

    # --- Build Sequences ---
    print(f"\n[SEQUENCE] Building sequences (window={SEQUENCE_LEN})...")
    seq_train = build_sequences(X_ben_train_pca, SEQUENCE_LEN)
    seq_val   = build_sequences(X_ben_val_pca, SEQUENCE_LEN)

    print(f"  Train sequences: {seq_train.shape}")
    print(f"  Val sequences:   {seq_val.shape}")

    # --- Build Model ---
    print("\n[MODEL] Building TCN Autoencoder...")
    model = build_tcn_autoencoder(
        seq_len=SEQUENCE_LEN,
        n_features=PCA_COMPONENTS,
        filters=64,
        kernel_size=3,
        n_blocks=4,
        dropout_rate=0.1,
        bottleneck_dim=8,
    )
    model.summary()

    # --- Compile ---
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    # --- Callbacks ---
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            str(out_dir / "tcn_best.keras"),
            monitor="val_loss",
            save_best_only=True, verbose=0
        ),
    ]

    # --- Train ---
    print(f"\n[TRAIN] Training TCN Autoencoder ({epochs} epochs)...")
    history = model.fit(
        seq_train, seq_train,          # autoencoder: input = target
        validation_data=(seq_val, seq_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
    )

    # Plot training loss
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("TCN Autoencoder Training Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tcn_training_loss.png", dpi=150)
    plt.close()

    # --- Evaluate: Reconstruction Errors ---
    print("\n[EVAL] Computing reconstruction errors...")

    # Build test sequences
    seq_ben_test    = build_sequences(X_ben_test_pca, SEQUENCE_LEN)
    seq_attack_test = build_sequences(X_attack_pca, SEQUENCE_LEN)

    # TCN reconstruction errors
    ben_tcn_errors    = compute_reconstruction_error(model, seq_ben_test)
    attack_tcn_errors = compute_reconstruction_error(model, seq_attack_test)

    # CRITICAL: fit normalization params on benign VAL errors only
    # so attack errors can legitimately exceed the benign ceiling
    ben_val_errors = compute_reconstruction_error(model, seq_val)
    tcn_norm_params = {
        "mean": float(ben_val_errors.mean()),
        "std":  float(ben_val_errors.std()),
        "p99":  float(np.percentile(ben_val_errors, 99)),
    }
    print(f"\n  TCN norm params (benign val):")
    print(f"    mean={tcn_norm_params['mean']:.6f}  "
          f"std={tcn_norm_params['std']:.6f}  "
          f"p99={tcn_norm_params['p99']:.6f}")
    print(f"  Benign test TCN error mean:  {ben_tcn_errors.mean():.6f}")
    print(f"  Attack test TCN error mean:  {attack_tcn_errors.mean():.6f}")

    # Mahalanobis scores (on sequence-level: use last timestep)
    ben_mahal    = mahal_gate.score(X_ben_test_pca[SEQUENCE_LEN - 1:])
    attack_mahal = mahal_gate.score(X_attack_pca[SEQUENCE_LEN - 1:])

    # Align lengths (sequences drop first seq_len-1 samples)
    min_ben = min(len(ben_tcn_errors), len(ben_mahal))
    min_att = min(len(attack_tcn_errors), len(attack_mahal))

    # Fuse scores using benign-fitted normalization
    ben_fused    = fuse_scores(ben_tcn_errors[:min_ben],
                               ben_mahal[:min_ben], tcn_norm_params)
    attack_fused = fuse_scores(attack_tcn_errors[:min_att],
                               attack_mahal[:min_att], tcn_norm_params)

    print(f"  Benign  — mean fused score: {ben_fused.mean():.4f} "
          f"± {ben_fused.std():.4f}")
    print(f"  Attack  — mean fused score: {attack_fused.mean():.4f} "
          f"± {attack_fused.std():.4f}")

    # --- Threshold Calibration ---
    threshold_config = calibrate_anomaly_threshold(ben_fused, attack_fused)

    # --- Plots ---
    plot_reconstruction_errors(
        ben_fused, attack_fused,
        threshold_config["threshold"], out_dir
    )

    # --- Save Artifacts ---
    print("\n[SAVE] Saving model and artifacts...")

    # Keras model
    saved_model_path = out_dir / "tcn_model.keras"
    model.save(str(saved_model_path))
    print(f"✅ Saved Keras model → {saved_model_path}")

    # TFLite
    export_tflite(model, out_dir)

    # Scaler + PCA
    joblib.dump(scaler, out_dir / "tcn_scaler.pkl")
    joblib.dump(pca,    out_dir / "tcn_pca.pkl")
    print(f"✅ Saved scaler → {out_dir / 'tcn_scaler.pkl'}")
    print(f"✅ Saved PCA    → {out_dir / 'tcn_pca.pkl'}")

    # Threshold config
    threshold_config["mahalanobis"] = mahal_gate.to_dict()
    threshold_config["fusion_weights"] = {"tcn": 0.6, "mahalanobis": 0.4}
    threshold_config["sequence_len"] = SEQUENCE_LEN
    threshold_config["pca_components"] = PCA_COMPONENTS
    threshold_config["tcn_norm_params"] = tcn_norm_params

    thresh_path = out_dir / "tcn_threshold.json"
    with open(thresh_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    print(f"✅ Saved threshold config → {thresh_path}")

    # Training report
    report = {
        "trained_at":       datetime.now().isoformat(),
        "benign_path":      str(benign_path),
        "novel_path":       str(novel_path),
        "architecture":     "TCN_Autoencoder",
        "sequence_len":     SEQUENCE_LEN,
        "pca_components":   PCA_COMPONENTS,
        "n_features":       N_FEATURES,
        "benign_train":     int(len(X_ben_train)),
        "benign_val":       int(len(X_ben_val)),
        "benign_test":      int(len(X_ben_test)),
        "attack_test":      int(len(X_attack)),
        "epochs_trained":   int(len(history.history["loss"])),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss":   float(history.history["val_loss"][-1]),
        "anomaly_detection": {
            "roc_auc":      threshold_config["roc_auc"],
            "f1":           threshold_config["f1"],
            "precision":    threshold_config["precision"],
            "recall":       threshold_config["recall"],
            "threshold":    threshold_config["threshold"],
            "fpr":          threshold_config["fpr"],
        },
        "model_config": {
            "filters":          64,
            "kernel_size":      3,
            "n_blocks":         4,
            "bottleneck_dim":   8,
            "dropout_rate":     0.1,
        },
    }

    report_path = out_dir / "tcn_training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved training report → {report_path}")

    # --- Final Summary ---
    print("\n" + "="*60)
    print("TCN TRAINING COMPLETE")
    print("="*60)
    print(f"  ROC-AUC:   {threshold_config['roc_auc']:.4f}")
    print(f"  F1:        {threshold_config['f1']:.4f}")
    print(f"  Recall:    {threshold_config['recall']:.4f}")
    print(f"  Precision: {threshold_config['precision']:.4f}")
    print(f"  Threshold: {threshold_config['threshold']:.4f}")
    print(f"\n  Output dir: {out_dir}")
    print("\nNext steps:")
    print("  python feature_extractor.py  ← build live Scapy pipeline")
    print("  python network_ips.py        ← run full network layer")


# ---------------------------------------------------------------------------
# 12. INFERENCE HELPER
#     Import this in network_ips.py for live anomaly scoring
# ---------------------------------------------------------------------------

class TCNAnomalyDetector:
    """
    Inference wrapper for the TCN Autoencoder.
    Uses TFLite for optimized CPU inference.
    
    Usage in network_ips.py:
        detector = TCNAnomalyDetector.load("./models/")
        score = detector.score(flow_sequence)   # 0.0 - 1.0
        redis.set(f"tcn_threat:{src_ip}", score, ex=60)
    """

    def __init__(self, tflite_path: str, scaler, pca,
                 threshold_config: dict):
        # Load TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.scaler = scaler
        self.pca = pca
        self.threshold = threshold_config["threshold"]
        self.fusion_weights = threshold_config.get(
            "fusion_weights", {"tcn": 0.6, "mahalanobis": 0.4}
        )
        self.tcn_norm_params = threshold_config.get(
            "tcn_norm_params", {"p99": 1.0}
        )
        self.mahal_gate = MahalanobisFusionGate.from_dict(
            threshold_config["mahalanobis"]
        )
        self.seq_len = threshold_config.get("sequence_len", SEQUENCE_LEN)

    @classmethod
    def load(cls, model_dir: str):
        model_dir = Path(model_dir)
        scaler = joblib.load(model_dir / "tcn_scaler.pkl")
        pca    = joblib.load(model_dir / "tcn_pca.pkl")
        with open(model_dir / "tcn_threshold.json") as f:
            threshold_config = json.load(f)
        return cls(
            model_dir / "tcn_model.tflite",
            scaler, pca, threshold_config
        )

    def preprocess_sequence(self, raw_flows: np.ndarray) -> np.ndarray:
        """
        raw_flows: (seq_len, 32) raw unified features
        Returns: (1, seq_len, pca_components) ready for TFLite
        """
        scaled = self.scaler.transform(raw_flows)
        pca_out = self.pca.transform(scaled)
        return pca_out[np.newaxis, :, :].astype(np.float32)

    def score(self, raw_flows: np.ndarray) -> float:
        """
        Score a sequence of flows.
        
        Args:
            raw_flows: np.ndarray (seq_len, 32) — last N flows from this IP
            
        Returns:
            anomaly_score: float 0.0–1.0
            is_anomaly: bool (True if score >= threshold)
        """
        seq = self.preprocess_sequence(raw_flows)

        # TFLite inference
        self.interpreter.set_tensor(self.input_details[0]["index"], seq)
        self.interpreter.invoke()
        reconstruction = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        # Reconstruction error
        tcn_error = float(np.mean((seq - reconstruction) ** 2))

        # Mahalanobis score on last flow
        last_flow_pca = seq[0, -1:, :]
        mahal_score = float(self.mahal_gate.score(last_flow_pca)[0])

        # Fuse using benign-fitted normalization
        fused = fuse_scores(
            np.array([tcn_error]),
            np.array([mahal_score]),
            self.tcn_norm_params,
            self.fusion_weights["tcn"],
            self.fusion_weights["mahalanobis"],
        )[0]

        return float(fused), float(fused) >= self.threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TCN Autoencoder for zero-day network attack detection"
    )
    parser.add_argument(
        "--benign", required=True,
        help="Path to cicids_benign.csv from dataset_mapper.py"
    )
    parser.add_argument(
        "--novel", required=True,
        help="Path to cicids_novel.csv from dataset_mapper.py"
    )
    parser.add_argument(
        "--out", default="./models",
        help="Output directory for model files (default: ./models)"
    )
    parser.add_argument(
        "--epochs", type=int, default=80,
        help="Max training epochs (default: 80, early stopping active)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Training batch size (default: 256)"
    )
    args = parser.parse_args()
    main(args.benign, args.novel, args.out, args.epochs, args.batch_size)