import pandas as pd
import numpy as np
import json
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

from threat_classifier.src.network_level_threat_classifier.features import THREAT_FEATURES
from threat_classifier.src.network_level_threat_classifier.labels import LABEL_MAP


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets"
MODEL_DIR = PROJECT_ROOT / "models" / "threat_classifier"


class NetworkThreatEvaluator:
    """
    Comprehensive evaluation for Network-Level LightGBM Threat Classifier.
    Focuses on:
        - Overall performance
        - Minority class detection
        - Inference latency
        - Error breakdown
    """

    def __init__(self):
        self.model = None
        self.inverse_label_map = {v: k for k, v in LABEL_MAP.items()}

    # ======================================================
    # Load Model
    # ======================================================
    def load_model(self):
        print("\n" + "="*80)
        print("LOADING NETWORK THREAT MODEL")
        print("="*80)

        model_path = MODEL_DIR / "lgb_model.pkl"
        self.model = joblib.load(model_path)

        print(f"[INFO] Model loaded from {model_path}")

    # ======================================================
    # Load Test Data (Recreate Same Split)
    # ======================================================
    def load_test_data(self):
        print("\n" + "="*80)
        print("LOADING TEST DATA")
        print("="*80)

        data_files = list(DATA_DIR.glob("cicids_test.csv"))

        Xs, Ys = [], []

        for path in data_files:
            df = pd.read_csv(path)

            df.columns = df.columns.str.strip().str.lower()

            if "label" not in df.columns:
                continue

            df["label"] = (
                df["label"]
                .astype(str)
                .str.strip()
                .str.replace("�", "-", regex=False)
                .str.lower()
            )

            df = df[df["label"].isin(LABEL_MAP.keys())]

            X = df[THREAT_FEATURES].astype(float)
            Y = df["label"].map(LABEL_MAP)

            mask = ~X.isnull().any(axis=1)
            X = X[mask]
            Y = Y[mask]

            Xs.append(X)
            Ys.append(Y)

        X_all = pd.concat(Xs, ignore_index=True)
        Y_all = pd.concat(Ys, ignore_index=True)

        # Same split as training
        _, X_test, _, y_test = train_test_split(
            X_all,
            Y_all,
            test_size=0.2,
            stratify=Y_all,
            random_state=42
        )

        print(f"[INFO] Test set size: {len(X_test)}")
        print("\n[INFO] Label distribution in test set:")
        print(y_test.value_counts())

        return X_test, y_test

    # ======================================================
    # Evaluation
    # ======================================================
    def evaluate(self, X_test, y_test):
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)

        start_time = time.time()

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        inference_time = (time.time() - start_time) / len(X_test) * 1000

        print(f"[INFO] Average inference time: {inference_time:.3f} ms/sample")

        # Convert labels to string form
        y_test_labels = [self.inverse_label_map[y] for y in y_test]
        y_pred_labels = [self.inverse_label_map[y] for y in y_pred]

        # ================= OVERALL METRICS =================
        print("\nCLASSIFICATION REPORT")
        print("-"*80)

        report = classification_report(
            y_test_labels,
            y_pred_labels,
            output_dict=True,
            zero_division=0
        )

        print(classification_report(
            y_test_labels,
            y_pred_labels,
            zero_division=0
        ))

        print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
        print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")

        # ================= CONFUSION MATRIX =================
        print("\nCONFUSION MATRIX")
        print("-"*80)

        labels_sorted = sorted(LABEL_MAP.keys())

        cm = confusion_matrix(
            y_test_labels,
            y_pred_labels,
            labels=labels_sorted
        )

        cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
        print(cm_df)

        # ================= ERROR ANALYSIS =================
        print("\nERROR ANALYSIS")
        print("-"*80)

        errors = np.sum(np.array(y_test_labels) != np.array(y_pred_labels))
        print(f"Total errors: {errors} / {len(y_test_labels)} "
              f"({errors/len(y_test_labels)*100:.2f}%)")

        # ================= SAVE RESULTS =================
        self.save_results(report, cm_df, inference_time)

        return report, cm_df

    # ======================================================
    # Save Evaluation Results
    # ======================================================
    def save_results(self, report, cm_df, inference_time):

        eval_dir = MODEL_DIR / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        with open(eval_dir / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        cm_df.to_csv(eval_dir / "confusion_matrix.csv")

        summary = {
            "accuracy": report["accuracy"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
            "inference_time_ms": inference_time
        }

        with open(eval_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[INFO] Evaluation results saved to {eval_dir}")

    # ======================================================
    # Plot Confusion Matrix
    # ======================================================
    def plot_confusion_matrix(self, cm_df):

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title("Network-Level Threat Classification Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        plot_path = MODEL_DIR / "evaluation" / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"[INFO] Confusion matrix plot saved to {plot_path}")


# ==========================================================
# MAIN
# ==========================================================
def main():

    print("\n" + "="*80)
    print("NETWORK LEVEL THREAT CLASSIFIER - EVALUATION")
    print("="*80)

    evaluator = NetworkThreatEvaluator()

    evaluator.load_model()
    X_test, y_test = evaluator.load_test_data()
    report, cm_df = evaluator.evaluate(X_test, y_test)
    evaluator.plot_confusion_matrix(cm_df)

    print("\nEVALUATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
