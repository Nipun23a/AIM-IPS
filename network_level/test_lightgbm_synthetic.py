"""
test_lightgbm_real.py
=====================
Samples real flows from the CICIDS2017 mapped dataset and verifies
the LightGBM Network Classifier predicts them correctly.

No synthetic data — ground truth flows straight from your dataset.

Usage:
  python test_lightgbm_real.py \
      --data  ./mapped_data/cicids_mapped.csv \
      --models ./models/

  python test_lightgbm_real.py \
      --data  ./mapped_data/cicids_mapped.csv \
      --models ./models/ \
      --n 500 --verbose --export results.csv
"""

import json
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# CONSTANTS
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

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_dataset(data_path: str) -> pd.DataFrame:
    """Load cicids_mapped.csv produced by dataset_mapper.py."""
    print(f"\n[DATA] Loading {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    missing_feats = [f for f in UNIFIED_FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing features: {missing_feats}\n"
                         f"Run dataset_mapper.py first.")

    if "unified_label" not in df.columns:
        raise ValueError("Missing 'unified_label' column. "
                         "Run dataset_mapper.py first.")

    print(f"  Total rows: {len(df):,}")
    print(f"  Class distribution:")
    for label, count in df["unified_label"].value_counts().items():
        print(f"    {label:<15} {count:>10,}  ({count/len(df)*100:.1f}%)")

    return df


def sample_balanced(df: pd.DataFrame, n_per_class: int,
                    seed: int = 42) -> pd.DataFrame:
    """
    Sample n_per_class rows from each class.
    If a class has fewer rows than n_per_class, take all of them.
    Returns a shuffled combined dataframe.
    """
    samples = []
    for label in df["unified_label"].unique():
        class_df = df[df["unified_label"] == label]
        n = min(n_per_class, len(class_df))
        samples.append(class_df.sample(n=n, random_state=seed))

    combined = pd.concat(samples, ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n[SAMPLE] Balanced sample: {len(combined):,} flows")
    for label, count in combined["unified_label"].value_counts().items():
        print(f"  {label:<15} {count:>6,}")

    return combined


# ---------------------------------------------------------------------------
# 2. MODEL LOADING
# ---------------------------------------------------------------------------

def load_model(models_dir: str):
    models_dir = Path(models_dir)

    model = joblib.load(models_dir / "lightgbm_network.pkl")
    le    = joblib.load(models_dir / "label_encoder.pkl")

    with open(models_dir / "training_report.json") as f:
        report = json.load(f)

    print(f"\n[MODEL] Loaded LightGBM")
    print(f"  Classes:      {list(le.classes_)}")
    print(f"  Macro F1:     {report['metrics'].get('macro_f1', 'N/A'):.4f}")
    print(f"  Accuracy:     {report['metrics'].get('accuracy', 'N/A'):.4f}")
    print(f"  Best iter:    {report.get('best_iteration', 'N/A')}")

    return model, le, report


# ---------------------------------------------------------------------------
# 3. INFERENCE
# ---------------------------------------------------------------------------

def run_inference(model, le, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on all flows in df.
    Returns df with added columns:
      predicted_label, threat_score, confidence, correct
    """
    X = df[UNIFIED_FEATURES].values.astype(np.float32)

    probs        = model.predict(X)                      # (n, n_classes)
    pred_idxs    = np.argmax(probs, axis=1)
    pred_labels  = [le.classes_[i] for i in pred_idxs]

    benign_idx   = list(le.classes_).index("benign")
    threat_scores = 1.0 - probs[:, benign_idx]
    confidence    = np.max(probs, axis=1)

    result = df.copy()
    result["predicted_label"] = pred_labels
    result["threat_score"]    = threat_scores
    result["confidence"]      = confidence
    result["correct"]         = (result["predicted_label"] ==
                                  result["unified_label"]).astype(int)

    # Add per-class probability columns for deeper inspection
    for i, cls in enumerate(le.classes_):
        result[f"prob_{cls}"] = probs[:, i]

    return result


# ---------------------------------------------------------------------------
# 4. METRICS
# ---------------------------------------------------------------------------

def compute_metrics(result: pd.DataFrame, le) -> dict:
    """Compute per-class and overall metrics from inference results."""
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, f1_score
    )

    y_true = result["unified_label"].values
    y_pred = result["predicted_label"].values

    # Overall
    accuracy   = float(result["correct"].mean())
    macro_f1   = float(f1_score(y_true, y_pred,
                                 average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred,
                                  average="weighted", zero_division=0))

    # Per-class report
    report = classification_report(
        y_true, y_pred,
        target_names=sorted(result["unified_label"].unique()),
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("PER-CLASS METRICS ON REAL FLOWS")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred,
        target_names=sorted(result["unified_label"].unique()),
        zero_division=0,
    ))

    # Threat score statistics per class
    print("THREAT SCORE STATS PER CLASS:")
    print(f"  {'Class':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 50)
    for label in sorted(result["unified_label"].unique()):
        mask   = result["unified_label"] == label
        scores = result.loc[mask, "threat_score"]
        print(f"  {label:<15} {scores.mean():>8.4f} {scores.std():>8.4f} "
              f"{scores.min():>8.4f} {scores.max():>8.4f}")

    return {
        "accuracy":    accuracy,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "per_class":   report,
    }


# ---------------------------------------------------------------------------
# 5. FAILURE ANALYSIS
# ---------------------------------------------------------------------------

def analyze_failures(result: pd.DataFrame, verbose: bool = False):
    """
    Inspect misclassified flows — what did the model get wrong and why?
    Shows the actual feature values of failed cases.
    """
    failures = result[result["correct"] == 0].copy()

    if len(failures) == 0:
        print("\n✅ No misclassifications!")
        return

    print(f"\n[FAILURES] {len(failures):,} misclassified flows "
          f"({len(failures)/len(result)*100:.1f}%)")
    print("\n  Confusion pairs (true → predicted):")

    pairs = Counter(zip(failures["unified_label"],
                        failures["predicted_label"]))
    for (true, pred), count in pairs.most_common(10):
        pct = count / len(failures) * 100
        print(f"    {true:<15} → {pred:<15}  {count:>6,} ({pct:.1f}%)")

    if verbose:
        print("\n  Sample misclassified flows (up to 5 per confusion pair):")
        for (true, pred), count in pairs.most_common(5):
            sample = failures[
                (failures["unified_label"] == true) &
                (failures["predicted_label"] == pred)
            ].head(3)

            print(f"\n  [{true} → {pred}]  ({count} total)")
            key_features = [
                "flow_duration", "total_fwd_packets", "total_bwd_packets",
                "flow_bytes_per_sec", "flow_pkts_per_sec",
                "syn_flag_count", "rst_flag_count", "psh_flag_count",
                "threat_score", "confidence"
            ]
            display_cols = [c for c in key_features if c in sample.columns]
            print(sample[display_cols].to_string(index=False))


# ---------------------------------------------------------------------------
# 6. PLOTS
# ---------------------------------------------------------------------------

def plot_results(result: pd.DataFrame, le, out_dir: Path):
    """Generate evaluation plots."""

    classes = sorted(result["unified_label"].unique())

    # --- Confusion matrix ---
    from sklearn.metrics import confusion_matrix

    cm      = confusion_matrix(result["unified_label"],
                                result["predicted_label"],
                                labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title("Confusion Matrix — Real Flows (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title("Confusion Matrix — Real Flows (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.suptitle("LightGBM Network Classifier — Real Flow Evaluation",
                 fontsize=13)
    plt.tight_layout()
    p = out_dir / "real_flow_confusion_matrix.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved confusion matrix → {p}")

    # --- Threat score distribution per class ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, label in enumerate(classes):
        if i >= len(axes):
            break
        ax = axes[i]
        mask = result["unified_label"] == label

        correct_scores   = result.loc[mask & (result["correct"] == 1),
                                       "threat_score"]
        incorrect_scores = result.loc[mask & (result["correct"] == 0),
                                       "threat_score"]

        ax.hist(correct_scores,   bins=40, alpha=0.7,
                color="steelblue", label=f"Correct ({len(correct_scores):,})",
                density=True)
        ax.hist(incorrect_scores, bins=40, alpha=0.7,
                color="crimson",   label=f"Wrong ({len(incorrect_scores):,})",
                density=True)
        ax.axvline(0.5, color="black", linestyle="--",
                   linewidth=1.5, label="threshold=0.5")

        accuracy = correct_scores.sum() / max(mask.sum(), 1)
        ax.set_title(f"{label}\n"
                     f"acc={mask.sum() and len(correct_scores)/mask.sum():.1%}  "
                     f"n={mask.sum():,}")
        ax.set_xlabel("Threat Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    for j in range(len(classes), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Threat Score Distribution by True Class — Real Flows",
                 fontsize=13)
    plt.tight_layout()
    p2 = out_dir / "real_flow_threat_scores.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved threat scores → {p2}")

    # --- Per-class accuracy bar chart ---
    class_acc = (result.groupby("unified_label")["correct"]
                 .mean()
                 .sort_values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if v >= 0.90 else
              "#f39c12" if v >= 0.75 else
              "#e74c3c" for v in class_acc.values]
    bars = ax.barh(class_acc.index, class_acc.values, color=colors)
    ax.axvline(0.90, color="green",  linestyle="--",
               linewidth=1.5, label="90% target", alpha=0.7)
    ax.axvline(0.75, color="orange", linestyle="--",
               linewidth=1.5, label="75% minimum", alpha=0.7)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Class Accuracy — Real Flows")
    ax.set_xlim(0, 1.1)
    ax.legend()
    for bar, val in zip(bars, class_acc.values):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10)
    plt.tight_layout()
    p3 = out_dir / "real_flow_accuracy.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved accuracy chart → {p3}")


# ---------------------------------------------------------------------------
# 7. SPOT CHECK — print individual flows and predictions
# ---------------------------------------------------------------------------

def spot_check(result: pd.DataFrame, n: int = 5):
    """
    Print n random flows from each class with their predictions.
    Good for quick sanity checking in the terminal.
    """
    print("\n" + "=" * 60)
    print("SPOT CHECK — Sample flows per class")
    print("=" * 60)

    key_features = [
        "flow_duration", "total_fwd_packets", "total_bwd_packets",
        "flow_bytes_per_sec", "syn_flag_count", "rst_flag_count",
    ]

    for label in sorted(result["unified_label"].unique()):
        mask   = result["unified_label"] == label
        sample = result[mask].sample(n=min(n, mask.sum()),
                                      random_state=42)

        correct_count = sample["correct"].sum()
        print(f"\n  [{label.upper()}]  {correct_count}/{len(sample)} correct")
        print(f"  {'true':<10} {'predicted':<10} {'threat':>7} "
              f"{'conf':>6}  key features →")

        for _, row in sample.iterrows():
            status = "✅" if row["correct"] else "❌"
            feat_str = "  ".join(
                f"{f.replace('total_','').replace('_count','').replace('flow_','')}"
                f"={row[f]:.0f}"
                for f in key_features if f in row
            )
            print(f"  {status} {row['unified_label']:<10} "
                  f"{row['predicted_label']:<10} "
                  f"{row['threat_score']:>7.4f} "
                  f"{row['confidence']:>6.4f}  {feat_str}")


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

def main(data_path: str, models_dir: str, n_per_class: int,
         verbose: bool, export_path: str, out_dir: str):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LIGHTGBM REAL FLOW TEST")
    print("=" * 60)

    # Load data and model
    df      = load_dataset(data_path)
    model, le, train_report = load_model(models_dir)

    # Sample balanced flows from each class
    sample  = sample_balanced(df, n_per_class=n_per_class)

    # Run inference
    print(f"\n[INFERENCE] Predicting {len(sample):,} flows...")
    result  = run_inference(model, le, sample)

    # Metrics
    metrics = compute_metrics(result, le)

    # Failure analysis
    analyze_failures(result, verbose=verbose)

    # Spot check
    spot_check(result, n=5)

    # Plots
    plot_results(result, le, out_dir)

    # Export raw predictions
    if export_path:
        result.to_csv(export_path, index=False)
        print(f"\n✅ Exported full predictions → {export_path}")

    # Save test report
    report = {
        "data_path":       str(data_path),
        "models_dir":      str(models_dir),
        "n_per_class":     n_per_class,
        "total_flows":     len(sample),
        "overall": {
            "accuracy":    metrics["accuracy"],
            "macro_f1":    metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        },
        "per_class_accuracy": {
            label: float(result[result["unified_label"] == label]
                         ["correct"].mean())
            for label in sorted(result["unified_label"].unique())
        },
        "training_macro_f1": train_report["metrics"].get("macro_f1"),
        "class_distribution": sample["unified_label"].value_counts().to_dict(),
    }
    rp = out_dir / "real_flow_test_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved test report → {rp}")

    # Final verdict
    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    print(f"  Overall Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1      : {metrics['weighted_f1']:.4f}")
    print(f"\n  Per-class accuracy:")
    for label, acc in report["per_class_accuracy"].items():
        bar    = "█" * int(acc * 20)
        status = "✅" if acc >= 0.90 else "⚠️ " if acc >= 0.75 else "❌"
        print(f"    {status} {label:<15} {bar:<20} {acc:.1%}")

    print(f"\n  Training macro F1 : {report['training_macro_f1']:.4f}")
    print(f"  Test macro F1     : {metrics['macro_f1']:.4f}")
    gap = abs(metrics["macro_f1"] - report["training_macro_f1"])
    if gap > 0.05:
        print(f"  ⚠️  Gap: {gap:.4f} — possible overfitting, "
              f"check if test data leaked into training")
    else:
        print(f"  ✅ Gap: {gap:.4f} — model generalises well")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test LightGBM classifier on real CICIDS2017 flows"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to cicids_mapped.csv from dataset_mapper.py"
    )
    parser.add_argument(
        "--models", default="./models",
        help="Directory with lightgbm_network.pkl etc. (default: ./models)"
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Flows to sample per class (default: 200)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print feature values of misclassified flows"
    )
    parser.add_argument(
        "--export", default=None,
        help="Optional CSV path to export all predictions"
    )
    parser.add_argument(
        "--out", default="./models",
        help="Output directory for plots and report (default: ./models)"
    )
    args = parser.parse_args()
    main(args.data, args.models, args.n, args.verbose, args.export, args.out)