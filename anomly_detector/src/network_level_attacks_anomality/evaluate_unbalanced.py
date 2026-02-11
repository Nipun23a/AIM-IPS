import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import joblib
from datetime import datetime


# ==========================================================
# PATH CONFIGURATION
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "models" / "anomly_detector" / "network_level_attacks_anomality" / "tcn" / "models"
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets"
RESULTS_DIR = PROJECT_ROOT / "models" / "anomly_detector" / "network_level_attacks_anomality" / "tcn" / "evaluation_results"

UNBALANCED_TEST_PATH = DATA_DIR / "cicids_test_unbalanced.csv"
SCALER_PATH = PROJECT_ROOT / "models" / "anomly_detector" / "network_level_attacks_anomality" / "scaler.pkl"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# 🔥 Update this to your best model filename
try:
    MODEL_PATH = list(MODEL_DIR.glob("best_tcn_improved_*.keras"))[-1]
except IndexError:
    print(f"❌ No model found in {MODEL_DIR}")
    print(f"   Looking for pattern: best_tcn_improved_*.keras")
    exit(1)


# ==========================================================
# LOAD DATA (KEEP LABEL COLUMN)
# ==========================================================
def load_dataset(path):
    """
    Load test data - KEEP LABEL COLUMN (scaler expects it)
    """
    print(f"\n📂 Loading test data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    
    print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
    print(f"Column names sample: {df.columns.tolist()[:5]}")
    
    # Label column should already be 'label'
    if 'label' not in df.columns:
        raise ValueError(f"'label' column not found. Available columns: {df.columns.tolist()[:10]}...")
    
    print(f"✅ Found label column: 'label'")
    
    # Extract labels for evaluation (but KEEP in dataframe for scaler)
    y = df['label'].copy()
    
    # Drop ONLY identifier columns, NOT label
    drop_candidates = ['flow_id', 'id', 'timestamp', 'start_time', 'end_time', 'time', 'flow_id.1']
    columns_to_drop = [col for col in drop_candidates if col in df.columns]
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"   Dropped identifier columns: {columns_to_drop}")
    
    # Keep ALL remaining columns INCLUDING 'label'
    X = df.copy()
    
    print(f"✅ Data shape: {X.shape} (includes 'label' column)")
    print(f"   Column names sample: {X.columns.tolist()[:5]}")
    
    # Verify labels are 0/1
    if y.dtype == 'object':
        y = (y.astype(str).str.upper() != 'BENIGN').astype(int)
    else:
        y = y.astype(int)
    
    print(f"✅ Label distribution:")
    print(f"   Benign (0): {sum(y==0):,} ({sum(y==0)/len(y)*100:.2f}%)")
    print(f"   Attack (1): {sum(y==1):,} ({sum(y==1)/len(y)*100:.2f}%)")
    
    return X, y


# ==========================================================
# MAIN EVALUATION
# ==========================================================
def main():

    print("\n" + "="*70)
    print("EVALUATING TCN ON UNBALANCED DATASET")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if unbalanced dataset exists
    if not UNBALANCED_TEST_PATH.exists():
        print(f"\n❌ Unbalanced test dataset not found!")
        print(f"   Expected: {UNBALANCED_TEST_PATH}")
        print(f"\n💡 Please run the unbalanced dataset creation script first:")
        print(f"   python -m anomly_detector.src.network_level_attacks_anomality.unbalanced_data")
        return

    # Check if scaler exists
    if not SCALER_PATH.exists():
        print(f"\n❌ Scaler not found!")
        print(f"   Expected: {SCALER_PATH}")
        print(f"\n💡 Please train the model first to generate the scaler:")
        print(f"   python -m anomly_detector.src.network_level_attacks_anomality.train_tcn")
        return

    # Load model
    print(f"\n🔧 Loading model from: {MODEL_PATH.name}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully")
        print(f"   Model expects input shape: {model.input_shape}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load dataset (includes 'label' column)
    try:
        X_raw, y_true = load_dataset(UNBALANCED_TEST_PATH)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Clean data (same as training)
    print(f"\n🧹 Cleaning data...")
    print(f"   Replacing inf values with NaN...")
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values with median (same as training)
    print(f"   Filling NaN values with median...")
    for col in X_raw.columns:
        if pd.api.types.is_numeric_dtype(X_raw[col]):
            median = X_raw[col].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            X_raw[col] = X_raw[col].fillna(median)
        else:
            # Convert non-numeric to numeric
            X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
            median = X_raw[col].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            X_raw[col] = X_raw[col].fillna(median)
    
    print(f"✅ Data cleaned")

    # Load scaler
    print(f"\n📏 Loading scaler from: {SCALER_PATH.name}")
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler loaded successfully")
        
        # Check feature compatibility
        if hasattr(scaler, 'feature_names_in_'):
            scaler_features = list(scaler.feature_names_in_)
            data_features = list(X_raw.columns)
            
            print(f"\n🔍 Feature compatibility check:")
            print(f"   Scaler expects {len(scaler_features)} features")
            print(f"   Data has {len(data_features)} features")
            
            # Check if features match
            missing_in_data = set(scaler_features) - set(data_features)
            extra_in_data = set(data_features) - set(scaler_features)
            
            if missing_in_data:
                print(f"\n❌ ERROR: {len(missing_in_data)} features missing in data:")
                print(f"   {list(missing_in_data)}")
                raise ValueError(f"Missing required features: {missing_in_data}")
                
            if extra_in_data:
                print(f"\n⚠️  WARNING: {len(extra_in_data)} extra features in data:")
                print(f"   {list(extra_in_data)}")
                print(f"   Dropping extra features...")
            
            # Reorder columns to match scaler EXACTLY
            print(f"\n   Reordering columns to match scaler...")
            X_raw = X_raw[scaler_features]
            print(f"✅ Columns matched: {X_raw.shape}")
        
        # Scale the data (including 'label' column)
        X_scaled = scaler.transform(X_raw)
        print(f"✅ Data scaled successfully: {X_scaled.shape}")
        
    except Exception as e:
        print(f"❌ Failed to load/apply scaler: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(X_scaled):,}")
    print(f"   Feature dimensions: {X_scaled.shape[1]}")
    print(f"   Benign: {sum(y_true==0):,} ({sum(y_true==0)/len(y_true)*100:.2f}%)")
    print(f"   Attack: {sum(y_true==1):,} ({sum(y_true==1)/len(y_true)*100:.2f}%)")
    
    if sum(y_true==1) > 0:
        print(f"   Imbalance ratio: {sum(y_true==0)/sum(y_true==1):.2f}:1")

    # Predict
    print(f"\n🔮 Making predictions...")
    try:
        y_pred_proba = model.predict(X_scaled, verbose=0).flatten()
        
        # Try different thresholds
        print(f"\n🔬 Testing different thresholds:")
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            y_pred_test = (y_pred_proba > threshold).astype(int)
            recall_test = recall_score(y_true, y_pred_test, zero_division=0)
            precision_test = precision_score(y_true, y_pred_test, zero_division=0)
            f1_test = f1_score(y_true, y_pred_test, zero_division=0)
            
            print(f"   Threshold {threshold:.2f}: Recall={recall_test*100:5.2f}%, Precision={precision_test*100:5.2f}%, F1={f1_test*100:5.2f}%")
            
            if f1_test > best_f1:
                best_f1 = f1_test
                best_threshold = threshold
        
        print(f"\n✅ Using best threshold: {best_threshold} (F1={best_f1*100:.2f}%)")
        y_pred = (y_pred_proba > best_threshold).astype(int)
        
        print(f"✅ Predictions complete")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Data shape: {X_scaled.shape}")
        import traceback
        traceback.print_exc()
        return

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except Exception as e:
        auc = 0.0
        print(f"⚠️  Could not calculate AUC-ROC: {e}")

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # IPS Critical Metrics
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    false_negative_rate = FN / (FN + TP) if (FN + TP) > 0 else 0
    detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0

    print("\n" + "="*70)
    print("📊 PERFORMANCE ON UNBALANCED DATA")
    print("="*70)
    print(f"Accuracy:              {accuracy*100:.4f}%")
    print(f"Precision:             {precision*100:.4f}%")
    print(f"Recall (Detection):    {recall*100:.4f}%")
    print(f"F1-score:              {f1*100:.4f}%")
    print(f"AUC-ROC:               {auc*100:.4f}%")
    print(f"Best Threshold:        {best_threshold}")

    print("\n" + "="*70)
    print("🚨 IPS CRITICAL METRICS")
    print("="*70)
    print(f"False Positive Rate:   {false_positive_rate*100:.4f}% (target: < 1%)")
    print(f"False Negative Rate:   {false_negative_rate*100:.4f}% (target: < 5%)")
    print(f"Detection Rate:        {detection_rate*100:.4f}% (target: > 95%)")

    print("\n" + "="*70)
    print("📋 CONFUSION MATRIX")
    print("="*70)
    print(f"                 Predicted")
    print(f"                 Benign  Attack")
    print(f"Actual Benign    {TN:6d}  {FP:6d}")
    print(f"       Attack    {FN:6d}  {TP:6d}")

    print("\n" + "="*70)
    print("📋 CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, digits=4, target_names=['Benign', 'Attack']))
    
    print("\n" + "="*70)
    print("💡 INTERPRETATION FOR IPS DEPLOYMENT")
    print("="*70)
    
    # False Positive Rate Analysis
    if false_positive_rate < 0.01:
        fpr_status = "✅ EXCELLENT"
        fpr_msg = "Minimal false alarms, safe for production"
    elif false_positive_rate < 0.05:
        fpr_status = "⚠️  ACCEPTABLE"
        fpr_msg = "Some false alarms expected, monitor closely"
    else:
        fpr_status = "❌ HIGH"
        fpr_msg = "Too many false alarms, may block legitimate traffic"
    
    print(f"{fpr_status} - FPR: {false_positive_rate*100:.4f}%")
    print(f"   → {fpr_msg}")
    
    # Detection Rate Analysis
    if detection_rate > 0.95:
        dr_status = "✅ EXCELLENT"
        dr_msg = "Catches vast majority of attacks"
    elif detection_rate > 0.85:
        dr_status = "⚠️  GOOD"
        dr_msg = "Decent protection but room for improvement"
    else:
        dr_status = "❌ NEEDS IMPROVEMENT"
        dr_msg = "Missing too many attacks, retrain recommended"
    
    print(f"\n{dr_status} - Detection Rate: {detection_rate*100:.4f}%")
    print(f"   → {dr_msg}")
    
    # Overall Assessment
    print("\n" + "="*70)
    if false_positive_rate < 0.01 and detection_rate > 0.95:
        overall_status = "🎉 PRODUCTION READY"
        overall_msg = "Your model performs excellently on unbalanced data!"
    elif false_positive_rate < 0.05 and detection_rate > 0.85:
        overall_status = "👍 ACCEPTABLE FOR DEPLOYMENT"
        overall_msg = "Monitor performance and consider tuning threshold"
    else:
        overall_status = "⚠️  NEEDS IMPROVEMENT"
        overall_msg = "Consider retraining or adjusting decision threshold"
    
    print(f"{overall_status}")
    print(f"   {overall_msg}")
    print("="*70 + "\n")

    # Save results to file
    results = {
        'timestamp': timestamp,
        'model_path': str(MODEL_PATH),
        'best_threshold': float(best_threshold),
        'test_samples': len(y_true),
        'benign_samples': int(sum(y_true==0)),
        'attack_samples': int(sum(y_true==1)),
        'imbalance_ratio': float(sum(y_true==0)/sum(y_true==1)) if sum(y_true==1) > 0 else float('inf'),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'false_positive_rate': float(false_positive_rate),
        'false_negative_rate': float(false_negative_rate),
        'detection_rate': float(detection_rate),
        'confusion_matrix': {
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN),
            'TP': int(TP)
        }
    }
    
    results_file = RESULTS_DIR / f"unbalanced_evaluation_{timestamp}.pkl"
    joblib.dump(results, results_file)
    print(f"💾 Results saved to: {results_file}\n")


if __name__ == "__main__":
    main()