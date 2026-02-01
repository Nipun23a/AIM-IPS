# threat_classifier/src/application_level_threat_classifier/model_evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    precision_recall_curve
)
import joblib
import json
from pathlib import Path
import time

from threat_classifier.src.application_level_threat_classifier.feature_engineering import ThreatFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets" / "application_level"
MODEL_DIR = PROJECT_ROOT / "models" / "application_layer"


class ThreatClassifierEvaluator:
    """
    Comprehensive evaluation of the threat classification model.
    Focuses on minority class performance due to high imbalance.
    """
    
    def __init__(self):
        self.model = None
        self.feature_extractor = ThreatFeatureExtractor()
        self.class_mapping = None
        self.inverse_class_mapping = None
        
    def load_model(self):
        """Load trained model and artifacts"""
        print("\n" + "="*80)
        print("LOADING MODEL")
        print("="*80)
        
        # Load model
        model_path = MODEL_DIR / "threat_classifier_lgb.pkl"
        self.model = joblib.load(model_path)
        print(f"[INFO] Model loaded from {model_path}")
        
        # Load feature configuration
        feature_config_path = MODEL_DIR / "feature_config.pkl"
        self.feature_extractor.load_feature_config(feature_config_path)
        
        # Load class mapping
        class_mapping_path = MODEL_DIR / "class_mapping.json"
        with open(class_mapping_path, 'r') as f:
            mappings = json.load(f)
            self.class_mapping = {k: int(v) for k, v in mappings['class_to_idx'].items()}
            self.inverse_class_mapping = {int(k): v for k, v in mappings['idx_to_class'].items()}
        
        print(f"[INFO] Class mapping loaded: {self.class_mapping}")
        
    def load_test_data(self):
        """Load and prepare test data"""
        print("\n" + "="*80)
        print("LOADING TEST DATA")
        print("="*80)
        
        # Load full dataset and split (using same random state as training)
        data_path = DATA_DIR / "combined_threat_dataset_clean.csv"
        df = pd.read_csv(data_path)
        
        from sklearn.model_selection import train_test_split
        
        # Extract features
        X = self.feature_extractor.extract_batch_features(df['payload'])
        y = df['attack_type']
        y_encoded = y.map(self.class_mapping)
        
        # Split to get test set (same as training)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Get original labels for test set
        _, y_test_original = train_test_split(
            y, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"[INFO] Test set size: {len(X_test)}")
        print(f"\n[INFO] Test set distribution:")
        print(y_test_original.value_counts())
        
        return X_test, y_test, y_test_original
    
    def evaluate_predictions(self, X_test, y_test, y_test_original):
        """Comprehensive evaluation of model predictions"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Make predictions
        print(f"[INFO] Making predictions on {len(X_test)} samples...")
        start_time = time.time()
        
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
        
        print(f"[INFO] Predictions completed")
        print(f"[INFO] Average inference time: {inference_time:.3f} ms per sample")
        
        # Convert predictions to class names
        y_pred_labels = [self.inverse_class_mapping[idx] for idx in y_pred]
        
        # ============ OVERALL METRICS ============
        print("\n" + "="*80)
        print("OVERALL PERFORMANCE")
        print("="*80)
        
        # Classification report
        report = classification_report(
            y_test_original, 
            y_pred_labels, 
            output_dict=True,
            zero_division=0
        )
        
        print(classification_report(
            y_test_original, 
            y_pred_labels,
            zero_division=0
        ))
        
        # Overall metrics
        overall_accuracy = report['accuracy']
        weighted_f1 = report['weighted avg']['f1-score']
        macro_f1 = report['macro avg']['f1-score']
        
        print(f"\n[SUMMARY]")
        print(f"  Overall Accuracy: {overall_accuracy:.4f}")
        print(f"  Weighted F1-Score: {weighted_f1:.4f}")
        print(f"  Macro F1-Score: {macro_f1:.4f}")
        
        # ============ PER-CLASS METRICS ============
        print("\n" + "="*80)
        print("PER-CLASS PERFORMANCE")
        print("="*80)
        
        per_class_metrics = []
        for class_name in sorted(self.class_mapping.keys()):
            if class_name in report:
                metrics = report[class_name]
                per_class_metrics.append({
                    'Class': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': int(metrics['support'])
                })
        
        metrics_df = pd.DataFrame(per_class_metrics)
        print(metrics_df.to_string(index=False))
        
        # ============ CONFUSION MATRIX ============
        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)
        
        cm = confusion_matrix(y_test_original, y_pred_labels, 
                              labels=sorted(self.class_mapping.keys()))
        
        cm_df = pd.DataFrame(
            cm,
            index=sorted(self.class_mapping.keys()),
            columns=sorted(self.class_mapping.keys())
        )
        
        print(cm_df)
        
        # ============ MINORITY CLASS FOCUS ============
        print("\n" + "="*80)
        print("MINORITY CLASS PERFORMANCE (Critical for Imbalanced Data)")
        print("="*80)
        
        # Identify minority classes (< 1% of data)
        class_counts = y_test_original.value_counts()
        total_samples = len(y_test_original)
        
        for class_name in sorted(self.class_mapping.keys()):
            count = class_counts.get(class_name, 0)
            percentage = (count / total_samples) * 100
            
            if class_name in report:
                metrics = report[class_name]
                status = "✅" if metrics['f1-score'] >= 0.85 else "⚠️" if metrics['f1-score'] >= 0.70 else "❌"
                
                print(f"\n{status} {class_name} ({percentage:.2f}% of test set):")
                print(f"    Samples: {count}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1-Score: {metrics['f1-score']:.4f}")
        
        # ============ ERROR ANALYSIS ============
        print("\n" + "="*80)
        print("ERROR ANALYSIS")
        print("="*80)
        
        errors = []
        for i in range(len(y_test_original)):
            if y_test_original.iloc[i] != y_pred_labels[i]:
                errors.append({
                    'true_label': y_test_original.iloc[i],
                    'predicted_label': y_pred_labels[i],
                    'confidence': np.max(y_pred_proba[i])
                })
        
        print(f"[INFO] Total errors: {len(errors)} / {len(y_test_original)} ({len(errors)/len(y_test_original)*100:.2f}%)")
        
        if errors:
            error_df = pd.DataFrame(errors)
            print(f"\n[INFO] Error breakdown:")
            error_summary = error_df.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
            error_summary = error_summary.sort_values('count', ascending=False)
            print(error_summary.head(10).to_string(index=False))
        
        # Save evaluation results
        self.save_evaluation_results(report, cm_df, metrics_df, inference_time)
        
        return report, cm_df, metrics_df
    
    def save_evaluation_results(self, report, cm_df, metrics_df, inference_time):
        """Save evaluation results to files"""
        print("\n" + "="*80)
        print("SAVING EVALUATION RESULTS")
        print("="*80)
        
        eval_dir = MODEL_DIR / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        # Save classification report
        report_path = eval_dir / "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Classification report saved to {report_path}")
        
        # Save confusion matrix
        cm_path = eval_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        print(f"[INFO] Confusion matrix saved to {cm_path}")
        
        # Save per-class metrics
        metrics_path = eval_dir / "per_class_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"[INFO] Per-class metrics saved to {metrics_path}")
        
        # Save summary
        summary = {
            'overall_accuracy': report['accuracy'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'macro_f1': report['macro avg']['f1-score'],
            'inference_time_ms': inference_time,
            'total_test_samples': int(report['weighted avg']['support'])
        }
        
        summary_path = eval_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Evaluation summary saved to {summary_path}")
    
    def plot_confusion_matrix(self, cm_df):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix - Threat Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = MODEL_DIR / "evaluation" / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Confusion matrix plot saved to {plot_path}")
        plt.close()


def main():
    """Main evaluation pipeline"""
    
    print("\n" + "="*80)
    print("APPLICATION LAYER THREAT CLASSIFIER - EVALUATION")
    print("="*80)
    
    evaluator = ThreatClassifierEvaluator()
    
    # Step 1: Load model
    evaluator.load_model()
    
    # Step 2: Load test data
    X_test, y_test, y_test_original = evaluator.load_test_data()
    
    # Step 3: Evaluate
    report, cm_df, metrics_df = evaluator.evaluate_predictions(X_test, y_test, y_test_original)
    
    # Step 4: Plot confusion matrix
    evaluator.plot_confusion_matrix(cm_df)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to: {MODEL_DIR / 'evaluation'}")


if __name__ == "__main__":
    main()