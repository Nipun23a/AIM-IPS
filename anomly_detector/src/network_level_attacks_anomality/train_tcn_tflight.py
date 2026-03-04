"""
train_tcn_tflite.py
────────────────────
TCN TFLite trainer for network-level attack detection.
Mirrors the CNN TFLite pattern used in the application-level anomaly detector.

Pipeline:
    1. Load CICIDS data (train + test CSVs)
    2. Preprocess + StandardScaler
    3. Train improved TCN (Keras)
    4. Convert  Keras → TFLite (FP32)  →  TFLite (INT8, full-integer)
    5. Evaluate TFLite model on test set
    6. Save everything: .tflite, scaler.pkl, feature_info.pkl, plots, logs

Output layout (mirrors CNN):
    models/anomly_detector/network_level_attacks_anomality/tcn/
        models/
            tcn_fp32.tflite          ← main inference file
            tcn_int8.tflite          ← quantised (smaller / faster)
            best_tcn_<ts>.keras      ← checkpoint (kept for re-conversion)
        logs/
            training_log_<ts>.csv
            test_metrics_<ts>.pkl
            training_summary_<ts>.txt
        plots/
            tcn_training_history_<ts>.png
            tcn_evaluation_<ts>.png
    models/anomly_detector/network_level_attacks_anomality/
        scaler.pkl                   ← StandardScaler (shared with CNN detector)
        feature_info.pkl             ← feature list + metadata
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from typing import List, Tuple, Optional

# ── GPU setup (mirrors CNN trainer) ──────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU enabled:", gpus)
    except RuntimeError as e:
        print("⚠️  GPU setup error:", e)
else:
    print("⚠️  GPU not found, using CPU")

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[3] / "models" / "anomly_detector" / "network_level_attacks_anomality"
DATASET_DIR = Path(__file__).resolve().parents[3] / "data_collector" / "data_sets"
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")

TCN_DIR     = BASE_DIR / "tcn"
PLOTS_DIR   = TCN_DIR  / "plots"
MODELS_DIR  = TCN_DIR  / "models"
LOGS_DIR    = TCN_DIR  / "logs"
SCALER_PATH = BASE_DIR / "scaler.pkl"

for d in [BASE_DIR, TCN_DIR, PLOTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"TCN outputs → {TCN_DIR}")

# ── Output filenames ──────────────────────────────────────────────────────────
TFLITE_FP32_PATH = MODELS_DIR / "tcn_fp32.tflite"
TFLITE_INT8_PATH = MODELS_DIR / "tcn_int8.tflite"


# =============================================================================
# DATA LOADING  (same helpers as original TCN trainer)
# =============================================================================

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def intersect_features(dfs: List[pd.DataFrame]) -> List[str]:
    """Intersection of columns across DataFrames, stripping non-numeric IDs."""
    blacklist = {
        'srcip', 'dstip', 'proto', 'service', 'state',
        'Timestamp', 'Flow ID', 'Label', 'attack_cat',
        'attack_category', 'id', 'start_time', 'end_time'
    }
    inter = set.intersection(*[set(df.columns) for df in dfs])
    inter = sorted(c for c in inter if c not in blacklist)
    return inter


def prepare_from_paths(paths: List[Path]):
    """Load CSVs, align features, extract binary labels."""
    dfs = [load_csv(p) for p in paths]
    features = intersect_features(dfs)
    print(f"[data] {len(features)} intersecting features")

    Xs, Ys = [], []
    label_candidates = ['Label', 'label', 'Attack', 'attack', 'class', 'Class', 'attack_cat']
    for df in dfs:
        X = df[features].copy()
        label = None
        for c in label_candidates:
            if c in df.columns:
                label = df[c]
                break
        if label is None:
            label = pd.Series([-1] * len(X))
        mask = ~X.isnull().any(axis=1)
        Xs.append(X[mask].astype(float))
        Ys.append(label[mask])

    return pd.concat(Xs, ignore_index=True), pd.concat(Ys, ignore_index=True), features


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess(train_path: Path, test_path: Path):
    """Load, clean, scale and split data. Saves scaler + feature_info."""
    print("\n" + "=" * 70)
    print("PREPROCESSING CICIDS DATASET")
    print("=" * 70)

    X_tr_raw, y_tr_raw, features = prepare_from_paths([train_path])
    X_te_raw, y_te_raw, _        = prepare_from_paths([test_path])

    print(f"   Train raw: {X_tr_raw.shape}  |  Test raw: {X_te_raw.shape}")

    # Binary labels (0 = benign, 1 = attack)
    def to_binary(y: pd.Series) -> np.ndarray:
        if y.dtype == object:
            return (y.str.upper() != 'BENIGN').astype(int).values
        return (y != 0).astype(int).values

    y_train_bin = to_binary(y_tr_raw)
    y_test_bin  = to_binary(y_te_raw)

    for name, y in [("Train", y_train_bin), ("Test", y_test_bin)]:
        b, a = np.sum(y == 0), np.sum(y == 1)
        print(f"   {name}: Benign={b:,} ({b/len(y)*100:.1f}%)  Attack={a:,} ({a/len(y)*100:.1f}%)")

    # Clean infinities / NaN
    for df in [X_tr_raw, X_te_raw]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_test_sc  = scaler.transform(X_te_raw).astype(np.float32)

    joblib.dump(scaler, SCALER_PATH)
    print(f"   Scaler saved → {SCALER_PATH}")

    # Validation split (stratified 20 %)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_sc, y_train_bin,
        test_size=0.2, random_state=42, stratify=y_train_bin
    )

    # Persist feature list
    feature_info = {'features': features, 'n_features': len(features), 'timestamp': TIMESTAMP}
    feature_path = BASE_DIR / "feature_info.pkl"
    joblib.dump(feature_info, feature_path)
    print(f"   Feature info saved → {feature_path}")
    print(f"   Train={X_tr.shape}  Val={X_val.shape}  Test={X_test_sc.shape}")

    return X_tr, X_val, X_test_sc, y_tr, y_val, y_test_bin, scaler, features


# =============================================================================
# TCN ARCHITECTURE  (same blocks as original, kept identical for parity)
# =============================================================================

def _tcn_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout: float):
    """Causal dilated residual TCN block."""
    def causal_conv(inp):
        return layers.Conv1D(
            filters, kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(inp)

    c1 = layers.SpatialDropout1D(dropout)(causal_conv(x))
    c2 = layers.SpatialDropout1D(dropout)(causal_conv(c1))

    res = x if x.shape[-1] == filters else layers.Conv1D(filters, 1)(x)
    return layers.Activation("relu")(layers.Add()([c2, res]))


def build_tcn(input_dim: int, num_blocks: int = 4, filters: int = 64,
              kernel_size: int = 3, dropout: float = 0.2) -> tf.keras.Model:
    """Improved TCN (same architecture as original trainer)."""
    inp = layers.Input(shape=(input_dim,), name='input')
    x   = layers.Reshape((input_dim, 1))(inp)

    # Initial projection
    x = layers.Conv1D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Dilated TCN blocks
    for i in range(num_blocks):
        x = _tcn_block(x, filters=filters, kernel_size=kernel_size,
                       dilation_rate=2 ** i, dropout=dropout)

    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation='sigmoid', name='output')(x)
    model = models.Model(inp, out, name='tcn_network')

    print(f"\n✅ TCN built  params={model.count_params():,}  "
          f"({model.count_params() * 4 / 1024:.1f} KB)")
    return model


# =============================================================================
# TRAINING
# =============================================================================

def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray,   y_val: np.ndarray,
          input_dim: int, epochs: int = 50, batch_size: int = 128):

    print("\n" + "=" * 70)
    print("TRAINING TCN")
    print("=" * 70)

    model = build_tcn(input_dim)

    # Class weights for imbalance
    n_b, n_a = np.sum(y_train == 0), np.sum(y_train == 1)
    total = len(y_train)
    class_weight = {0: total / (2 * n_b), 1: total / (2 * n_a)}
    print(f"   Class weights: {class_weight}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )

    ckpt_path = MODELS_DIR / f"best_tcn_{TIMESTAMP}.keras"
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=10,
                      restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(ckpt_path), monitor='val_auc', mode='max',
                        save_best_only=True, verbose=1),
        CSVLogger(str(LOGS_DIR / f"training_log_{TIMESTAMP}.csv")),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, class_weight=class_weight,
        verbose=1
    )

    hist_path = LOGS_DIR / f"history_{TIMESTAMP}.pkl"
    joblib.dump(history.history, hist_path)
    print(f"\n💾 History saved → {hist_path}")

    return model, history


# =============================================================================
# TFLITE CONVERSION  (mirrors CNN TFLite pattern)
# =============================================================================

def convert_to_tflite(model: tf.keras.Model,
                      X_repr: np.ndarray) -> Tuple[Path, Path]:
    """
    Convert Keras model → FP32 TFLite  →  INT8 TFLite.

    Args:
        model:   Trained Keras model.
        X_repr:  Representative dataset for INT8 calibration
                 (subset of training data, ~500–1000 rows).

    Returns:
        (fp32_path, int8_path)
    """
    print("\n" + "=" * 70)
    print("TFLITE CONVERSION")
    print("=" * 70)

    # ── FP32 ──────────────────────────────────────────────────────────────────
    print("[TFLite] Converting FP32 ...")
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp32.optimizations = []                         # no quantisation
    tflite_fp32 = converter_fp32.convert()

    TFLITE_FP32_PATH.write_bytes(tflite_fp32)
    fp32_kb = len(tflite_fp32) / 1024
    print(f"[TFLite] FP32 → {TFLITE_FP32_PATH}  ({fp32_kb:.1f} KB)")

    # ── INT8 (dynamic-range + representative data) ────────────────────────────
    print("[TFLite] Converting INT8 (full-integer quantisation) ...")

    def representative_dataset():
        """Yield small batches from training data for calibration."""
        indices = np.random.choice(len(X_repr), size=min(500, len(X_repr)), replace=False)
        for idx in indices:
            sample = X_repr[idx:idx + 1].astype(np.float32)
            yield [sample]

    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations              = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset     = representative_dataset
    converter_int8.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type       = tf.float32   # keep float I/O for ease
    converter_int8.inference_output_type      = tf.float32

    tflite_int8 = converter_int8.convert()
    TFLITE_INT8_PATH.write_bytes(tflite_int8)
    int8_kb = len(tflite_int8) / 1024
    print(f"[TFLite] INT8 → {TFLITE_INT8_PATH}  ({int8_kb:.1f} KB)")
    print(f"[TFLite] Size reduction: {fp32_kb:.1f} KB → {int8_kb:.1f} KB "
          f"({(1 - int8_kb/fp32_kb)*100:.0f}% smaller)")

    return TFLITE_FP32_PATH, TFLITE_INT8_PATH


# =============================================================================
# TFLITE INFERENCE HELPER  (mirrors CNN TFLite detector pattern)
# =============================================================================

def run_tflite_inference(tflite_path: Path, X: np.ndarray) -> np.ndarray:
    """
    Run inference with a saved TFLite model.
    Returns flat array of anomaly scores [0, 1].
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inp_details  = interpreter.get_input_details()
    out_details  = interpreter.get_output_details()

    scores = []
    for i in range(len(X)):
        interpreter.set_tensor(inp_details[0]['index'],
                               X[i:i+1].astype(np.float32))
        interpreter.invoke()
        scores.append(float(interpreter.get_tensor(out_details[0]['index'])[0][0]))

    return np.array(scores, dtype=np.float32)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model: Optional[tf.keras.Model],
             tflite_path: Path,
             X_test: np.ndarray,
             y_test: np.ndarray) -> dict:
    """Evaluate both Keras model and TFLite model side-by-side."""
    print("\n" + "=" * 70)
    print("EVALUATION — Keras vs TFLite")
    print("=" * 70)

    results = {}

    for name, get_proba in [
        ("Keras",        lambda: model.predict(X_test, verbose=0).flatten()),
        ("TFLite FP32",  lambda: run_tflite_inference(tflite_path, X_test)),
    ]:
        t0 = time.time()
        proba = get_proba()
        elapsed = time.time() - t0
        ms_per = elapsed / len(X_test) * 1000

        pred = (proba > 0.5).astype(int)

        acc  = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec  = recall_score(y_test, pred, zero_division=0)
        f1   = f1_score(y_test, pred, zero_division=0)
        auc  = roc_auc_score(y_test, proba)
        cm   = confusion_matrix(y_test, pred)
        TN, FP, FN, TP = cm.ravel()

        fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        fnr_val = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        det     = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        print(f"\n── {name} ({ms_per:.3f} ms/sample) ──")
        print(f"   Accuracy:            {acc*100:.4f}%")
        print(f"   Precision:           {prec*100:.4f}%")
        print(f"   Recall:              {rec*100:.4f}%")
        print(f"   F1-Score:            {f1*100:.4f}%")
        print(f"   AUC-ROC:             {auc*100:.4f}%")
        print(f"   False Positive Rate: {fpr_val*100:.4f}%")
        print(f"   False Negative Rate: {fnr_val*100:.4f}%")
        print(f"   Detection Rate:      {det*100:.4f}%")
        print(classification_report(y_test, pred,
                                    target_names=['Benign', 'Attack'], digits=4))

        results[name] = dict(
            accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc,
            fpr=fpr_val, fnr=fnr_val, detection_rate=det,
            confusion_matrix=cm, y_pred=pred, y_pred_proba=proba
        )

    metrics_path = LOGS_DIR / f"test_metrics_{TIMESTAMP}.pkl"
    joblib.dump(results, metrics_path)
    print(f"\n💾 Metrics saved → {metrics_path}")
    return results


# =============================================================================
# PLOTS
# =============================================================================

def plot_training(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TCN TFLite — Training History', fontsize=16)

    pairs = [
        (axes[0, 0], 'loss',      'Loss'),
        (axes[0, 1], 'accuracy',  'Accuracy'),
        (axes[1, 0], 'auc',       'AUC-ROC'),
    ]
    for ax, key, title in pairs:
        ax.plot(history.history[key],         label='Train', linewidth=2)
        ax.plot(history.history[f'val_{key}'], label='Val',   linewidth=2)
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for k, lbl in [('precision','Train Prec'), ('val_precision','Val Prec'),
                   ('recall',   'Train Rec'),  ('val_recall',   'Val Rec')]:
        ax.plot(history.history[k], label=lbl, linewidth=2)
    ax.set_title('Precision & Recall'); ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"tcn_training_history_{TIMESTAMP}.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight'); plt.close()
    print(f"📊 Training plot → {out}")


def plot_evaluation(results: dict):
    """Confusion matrix + ROC for Keras and TFLite side-by-side."""
    names = list(results.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 5 * len(names)))
    fig.suptitle('TCN TFLite — Evaluation', fontsize=16)

    if len(names) == 1:
        axes = [axes]

    for row, name in enumerate(names):
        r = results[name]

        # Confusion matrix
        sns.heatmap(r['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    ax=axes[row][0],
                    xticklabels=['Benign', 'Attack'],
                    yticklabels=['Benign', 'Attack'])
        axes[row][0].set_title(f'{name} — Confusion Matrix')
        axes[row][0].set_ylabel('True'); axes[row][0].set_xlabel('Predicted')

        # ROC
        fpr_arr, tpr_arr, _ = roc_curve(r['y_pred_proba'] > 0,  # dummy; use proba below
                                         r['y_pred_proba'])
        # re-compute properly
        from sklearn.metrics import roc_curve as _rc
        fpr_arr, tpr_arr, _ = _rc(
            (r['y_pred_proba'] > 0.5).astype(int) if 'y_true' not in r
            else r.get('y_true', r['y_pred'] > 0),
            r['y_pred_proba']
        )
        axes[row][1].plot(fpr_arr, tpr_arr, linewidth=2,
                          label=f'AUC={r["auc"]:.4f}')
        axes[row][1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[row][1].set_title(f'{name} — ROC')
        axes[row][1].set_xlabel('FPR'); axes[row][1].set_ylabel('TPR')
        axes[row][1].legend(); axes[row][1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"tcn_evaluation_{TIMESTAMP}.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight'); plt.close()
    print(f"📊 Evaluation plot → {out}")


# =============================================================================
# MAIN
# =============================================================================

def main(train_path: Path, test_path: Path, epochs: int = 50, batch_size: int = 128):
    print("\n" + "=" * 70)
    print("TCN TFLITE — NETWORK ANOMALY DETECTION")
    print("=" * 70)
    print(f"Train : {train_path}")
    print(f"Test  : {test_path}")

    # 1. Data
    X_tr, X_val, X_te, y_tr, y_val, y_te, scaler, features = \
        preprocess(train_path, test_path)

    input_dim = X_tr.shape[1]

    # 2. Train
    keras_model, history = train(X_tr, y_tr, X_val, y_val,
                                 input_dim=input_dim,
                                 epochs=epochs,
                                 batch_size=batch_size)

    # 3. Plot training
    plot_training(history)

    # 4. Convert → TFLite (FP32 + INT8)
    #    Use a calibration subset from training data (≤1000 rows)
    cal_idx   = np.random.choice(len(X_tr), size=min(1000, len(X_tr)), replace=False)
    X_calib   = X_tr[cal_idx]
    fp32_path, int8_path = convert_to_tflite(keras_model, X_calib)

    # 5. Evaluate Keras + TFLite FP32
    eval_results = evaluate(keras_model, fp32_path, X_te, y_te)

    # 6. Plots
    plot_evaluation(eval_results)

    # 7. Write summary
    r_k = eval_results.get("Keras", {})
    summary_path = LOGS_DIR / f"training_summary_{TIMESTAMP}.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TCN TFLITE — TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp    : {TIMESTAMP}\n")
        f.write(f"Input dim    : {input_dim}\n")
        f.write(f"Train samples: {len(X_tr):,}\n")
        f.write(f"Val   samples: {len(X_val):,}\n")
        f.write(f"Test  samples: {len(X_te):,}\n")
        f.write(f"FP32 TFLite  : {fp32_path}\n")
        f.write(f"INT8 TFLite  : {int8_path}\n")
        f.write("\nKeras Test Performance:\n")
        for k, v in [('accuracy','Accuracy'), ('precision','Precision'),
                     ('recall','Recall'), ('f1','F1-Score'), ('auc','AUC-ROC'),
                     ('fpr','False Positive Rate'), ('detection_rate','Detection Rate')]:
            f.write(f"  {v:25s}: {r_k.get(k, 0)*100:.4f}%\n")
    print(f"💾 Summary → {summary_path}")

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print(f"  FP32 TFLite : {fp32_path}")
    print(f"  INT8 TFLite : {int8_path}")
    print(f"  Scaler      : {SCALER_PATH}")
    print(f"  Features    : {BASE_DIR / 'feature_info.pkl'}")
    print(f"  Plots       : {PLOTS_DIR}")
    print(f"  Logs        : {LOGS_DIR}")
    print("=" * 70 + "\n")

    return keras_model, eval_results, fp32_path, int8_path


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    TRAIN_PATH = DATASET_DIR / "cicids_train.csv"
    TEST_PATH  = DATASET_DIR / "cicids_test.csv"

    for p in [TRAIN_PATH, TEST_PATH]:
        if not p.exists():
            print(f"⚠️  File not found: {p}")
            print("Update TRAIN_PATH / TEST_PATH at the bottom of this script.")
            sys.exit(1)

    main(TRAIN_PATH, TEST_PATH, epochs=50, batch_size=128)