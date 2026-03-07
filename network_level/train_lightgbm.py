"""
train_lightgbm.py
=================
Trains a LightGBM multiclass classifier on CICIDS2017 mapped data
for the Network Layer IPS — detecting DDoS, DoS, PortScan, Botnet.

Outputs:
  - lightgbm_network.pkl         → trained model (joblib)
  - lightgbm_network.txt         → native LightGBM model (for inspection)
  - label_encoder.pkl            → label encoder for inference
  - training_report.json         → metrics, feature importance, thresholds
  - confusion_matrix.png         → visual evaluation

Usage:
  python train_lightgbm.py --data ./mapped_data/cicids_mapped.csv
  python train_lightgbm.py --data ./mapped_data/cicids_mapped.csv --tune
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
import seaborn as sns
import joblib

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
import lightgbm as lgb

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

# Default LightGBM hyperparameters
# Tuned for network IDS: fast inference, handles imbalance, low false positives
DEFAULT_PARAMS = {
    "objective":        "multiclass",
    "metric":           "multi_logloss",
    "boosting_type":    "gbdt",
    "num_leaves":       63,
    "max_depth":        8,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------

def load_data(data_path: str):
    """Load mapped CICIDS CSV and return X, y, label_encoder."""
    print(f"\n[DATA] Loading {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    # Validate features
    missing = [f for f in UNIFIED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}\n"
                         f"Run dataset_mapper.py first.")

    if "unified_label" not in df.columns:
        raise ValueError("Missing 'unified_label' column. Run dataset_mapper.py first.")

    X = df[UNIFIED_FEATURES].values.astype(np.float32)
    
    le = LabelEncoder()
    y = le.fit_transform(df["unified_label"])

    print(f"  Samples:  {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes:  {list(le.classes_)}")
    print(f"\n  Class distribution:")
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"    {le.classes_[cls]:<15} {count:>8,} "
              f"({count/len(y)*100:.1f}%)")

    return X, y, le


# ---------------------------------------------------------------------------
# 3. CLASS WEIGHTS
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, le: LabelEncoder) -> dict:
    """
    Compute class weights for LightGBM.
    Attacks should have higher weight than benign to minimize false negatives.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    weights = {}
    for cls, count in zip(classes, counts):
        label = le.classes_[cls]
        base_weight = total / (len(classes) * count)
        
        # Extra penalty for missing attacks vs false alarms
        # For a security system: missing an attack is worse than a false alarm
        if label == "benign":
            weights[cls] = base_weight * 0.5   # reduce benign weight
        else:
            weights[cls] = base_weight * 2.0   # boost attack weight

    print(f"\n  Class weights:")
    for cls, w in weights.items():
        print(f"    {le.classes_[cls]:<15} {w:.4f}")

    return weights


# ---------------------------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------------------------

def train(X_train, y_train, X_val, y_val, params: dict,
          class_weights: dict, le: LabelEncoder):
    """Train LightGBM with early stopping."""
    print("\n[TRAIN] Training LightGBM...")

    # Build sample weights array
    sample_weights = np.array([class_weights[c] for c in y_train])

    # LightGBM datasets
    train_set = lgb.Dataset(
        X_train, label=y_train,
        weight=sample_weights,
        feature_name=UNIFIED_FEATURES,
    )
    val_set = lgb.Dataset(
        X_val, label=y_val,
        reference=train_set,
        feature_name=UNIFIED_FEATURES,
    )

    # Add num_class to params
    train_params = params.copy()
    train_params["num_class"] = len(le.classes_)
    n_estimators = train_params.pop("n_estimators", 500)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        train_params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[val_set],
        callbacks=callbacks,
    )

    print(f"  Best iteration: {model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, le: LabelEncoder, out_dir: Path):
    """Full evaluation suite — metrics, confusion matrix, feature importance."""
    print("\n[EVAL] Evaluating model...")

    # Predictions
    y_prob = model.predict(X_test)             # shape: (n, n_classes)
    y_pred = np.argmax(y_prob, axis=1)

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

    # Per-class metrics
    metrics = {
        "accuracy":         float((y_pred == y_test).mean()),
        "macro_f1":         float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1":      float(f1_score(y_test, y_pred, average="weighted")),
        "macro_precision":  float(precision_score(y_test, y_pred, average="macro")),
        "macro_recall":     float(recall_score(y_test, y_pred, average="macro")),
        "per_class":        report,
    }

    # ROC-AUC (one-vs-rest)
    try:
        auc = roc_auc_score(
            y_test, y_prob,
            multi_class="ovr",
            average="macro"
        )
        metrics["roc_auc_macro"] = float(auc)
        print(f"  ROC-AUC (macro OvR): {auc:.4f}")
    except Exception as e:
        print(f"  ROC-AUC skipped: {e}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix → {cm_path}")

    # Feature importance plot
    importance_df = pd.DataFrame({
        "feature": UNIFIED_FEATURES,
        "gain":    model.feature_importance(importance_type="gain"),
        "split":   model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
    bars = ax.barh(importance_df["feature"], importance_df["gain"], color=colors)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("LightGBM Network Classifier — Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    fi_path = out_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature importance → {fi_path}")

    metrics["top_10_features"] = importance_df.head(10)["feature"].tolist()
    metrics["feature_importance"] = importance_df.set_index("feature")["gain"].to_dict()

    return metrics, importance_df


# ---------------------------------------------------------------------------
# 6. THRESHOLD CALIBRATION
# ---------------------------------------------------------------------------

def calibrate_thresholds(model, X_val, y_val, le: LabelEncoder) -> dict:
    """
    Find optimal per-class confidence thresholds for the Score Aggregator.
    
    For the Network Layer IPS, we want:
    - High recall on attacks (don't miss real attacks)
    - Acceptable precision (avoid flooding Redis with false positives)
    
    Returns thresholds that achieve >= 95% recall per attack class.
    """
    print("\n[THRESHOLD] Calibrating per-class thresholds...")

    y_prob = model.predict(X_val)
    thresholds = {}

    for cls_idx, cls_name in enumerate(le.classes_):
        if cls_name == "benign":
            thresholds[cls_name] = 0.5
            continue

        probs = y_prob[:, cls_idx]
        true_binary = (y_val == cls_idx).astype(int)

        if true_binary.sum() == 0:
            thresholds[cls_name] = 0.5
            continue

        # Find threshold where recall >= 0.95
        best_thresh = 0.5
        best_f1 = 0.0

        for t in np.arange(0.1, 0.9, 0.01):
            preds = (probs >= t).astype(int)
            if preds.sum() == 0:
                continue
            recall = recall_score(true_binary, preds, zero_division=0)
            f1 = f1_score(true_binary, preds, zero_division=0)

            # Prioritize recall >= 0.95, then maximize F1
            if recall >= 0.95 and f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        thresholds[cls_name] = round(float(best_thresh), 3)
        print(f"  {cls_name:<15} threshold: {best_thresh:.3f}  (F1: {best_f1:.3f})")

    return thresholds


# ---------------------------------------------------------------------------
# 7. HYPERPARAMETER TUNING (optional)
# ---------------------------------------------------------------------------

def tune_hyperparameters(X_train, y_train, le: LabelEncoder) -> dict:
    """
    Simple grid search over key LightGBM hyperparameters.
    Uses 3-fold CV. Run with --tune flag.
    """
    print("\n[TUNE] Running hyperparameter search...")

    try:
        from sklearn.model_selection import GridSearchCV
        from lightgbm import LGBMClassifier

        param_grid = {
            "num_leaves":    [31, 63, 127],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth":     [6, 8, 10],
        }

        base = LGBMClassifier(
            objective="multiclass",
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(
            base, param_grid,
            cv=cv, scoring="f1_macro",
            n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)

        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV F1:  {grid.best_score_:.4f}")

        best = DEFAULT_PARAMS.copy()
        best.update(grid.best_params_)
        return best

    except Exception as e:
        print(f"  Tuning failed: {e}, using default params")
        return DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

def main(data_path: str, out_dir: str, tune: bool):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, le = load_data(data_path)

    # Train / val / test split — stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"\n  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Class weights
    class_weights = compute_class_weights(y_train, le)

    # Optionally tune
    params = tune_hyperparameters(X_train, y_train, le) if tune else DEFAULT_PARAMS

    # Train
    model = train(X_train, y_train, X_val, y_val, params, class_weights, le)

    # Evaluate
    metrics, importance_df = evaluate(model, X_test, y_test, le, out_dir)

    # Calibrate thresholds for Score Aggregator
    thresholds = calibrate_thresholds(model, X_val, y_val, le)

    # Save model
    model_path = out_dir / "lightgbm_network.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Saved model → {model_path}")

    # Save native LightGBM model text
    txt_path = out_dir / "lightgbm_network.txt"
    model.save_model(str(txt_path))
    print(f"✅ Saved model text → {txt_path}")

    # Save label encoder
    le_path = out_dir / "label_encoder.pkl"
    joblib.dump(le, le_path)
    print(f"✅ Saved label encoder → {le_path}")

    # Save training report
    report = {
        "trained_at":       datetime.now().isoformat(),
        "data_path":        str(data_path),
        "train_samples":    int(len(X_train)),
        "val_samples":      int(len(X_val)),
        "test_samples":     int(len(X_test)),
        "classes":          list(le.classes_),
        "features":         UNIFIED_FEATURES,
        "feature_count":    len(UNIFIED_FEATURES),
        "best_iteration":   int(model.best_iteration),
        "metrics":          {
            "accuracy":     metrics["accuracy"],
            "macro_f1":     metrics["macro_f1"],
            "weighted_f1":  metrics["weighted_f1"],
            "roc_auc":      metrics.get("roc_auc_macro", None),
        },
        "thresholds":       thresholds,
        "top_10_features":  metrics["top_10_features"],
        "hyperparameters":  params,
    }

    report_path = out_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved training report → {report_path}")

    # Final summary
    print("\n" + "="*60)
    print("LIGHTGBM TRAINING COMPLETE")
    print("="*60)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    if "roc_auc_macro" in metrics:
        print(f"  ROC-AUC:     {metrics['roc_auc_macro']:.4f}")
    print(f"\n  Score Aggregator thresholds:")
    for cls, t in thresholds.items():
        print(f"    {cls:<15} {t}")
    print(f"\n  Output dir: {out_dir}")
    print("\nNext steps:")
    print("  python train_tcn.py --data ./mapped_data/nb15_mapped.csv")
    print("  python network_ips.py --model ./models/lightgbm_network.pkl")


# ---------------------------------------------------------------------------
# 9. INFERENCE HELPER
#    Import this in network_ips.py for live scoring
# ---------------------------------------------------------------------------

class LightGBMNetworkClassifier:
    """
    Lightweight wrapper for inference in the Network Layer IPS pipeline.
    Designed for minimal overhead — target < 0.5ms per inference.

    Usage in network_ips.py:
        classifier = LightGBMNetworkClassifier.load("./models/")
        score, label = classifier.predict(feature_vector)
        redis.set(f"threat:{src_ip}", score, ex=60)
    """

    def __init__(self, model, le: LabelEncoder, thresholds: dict):
        self.model = model
        self.le = le
        self.thresholds = thresholds
        self.feature_names = UNIFIED_FEATURES

    @classmethod
    def load(cls, model_dir: str):
        model_dir = Path(model_dir)
        model = joblib.load(model_dir / "lightgbm_network.pkl")
        le = joblib.load(model_dir / "label_encoder.pkl")

        with open(model_dir / "training_report.json") as f:
            report = json.load(f)

        return cls(model, le, report["thresholds"])

    def predict(self, features: np.ndarray):
        """
        Predict threat class and confidence score.
        
        Args:
            features: np.ndarray of shape (32,) or (1, 32)
            
        Returns:
            threat_score: float 0.0–1.0 (0 = benign, 1 = high threat)
            predicted_class: str label
            class_probs: dict of {class: probability}
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        probs = self.model.predict(features)[0]  # shape: (n_classes,)
        pred_idx = np.argmax(probs)
        pred_class = self.le.classes_[pred_idx]

        # Threat score = 1 - P(benign)
        benign_idx = list(self.le.classes_).index("benign")
        threat_score = float(1.0 - probs[benign_idx])

        class_probs = {
            cls: float(prob)
            for cls, prob in zip(self.le.classes_, probs)
        }

        return threat_score, pred_class, class_probs

    def batch_predict(self, features: np.ndarray):
        """
        Predict for multiple flows at once (used in flow aggregator flush).
        
        Args:
            features: np.ndarray of shape (n_flows, 32)
            
        Returns:
            threat_scores: np.ndarray of shape (n_flows,)
            predicted_classes: list of str labels
        """
        probs = self.model.predict(features)          # (n_flows, n_classes)
        pred_idxs = np.argmax(probs, axis=1)
        pred_classes = [self.le.classes_[i] for i in pred_idxs]

        benign_idx = list(self.le.classes_).index("benign")
        threat_scores = 1.0 - probs[:, benign_idx]

        return threat_scores, pred_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LightGBM Network Layer IPS classifier"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to cicids_mapped.csv from dataset_mapper.py"
    )
    parser.add_argument(
        "--out", default="./models",
        help="Output directory for model files (default: ./models)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run hyperparameter search before training (slower)"
    )
    args = parser.parse_args()
    main(args.data, args.out, args.tune)