# cross_validate_nslkdd.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "anomly_detector" / "network_level_attacks_anomality" / "tcn" / "models"
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets" / "nsl-kdd"
RESULTS_DIR = PROJECT_ROOT / "models" / "anomly_detector" / "network_level_attacks_anomality" / "tcn" / "cross_validation_results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# NSL-KDD column names
NSLKDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]


def download_nslkdd():
    """Download NSL-KDD dataset if not present"""
    train_path = DATA_DIR / "KDDTrain+.txt"
    test_path = DATA_DIR / "KDDTest+.txt"
    
    if train_path.exists() and test_path.exists():
        print("✅ NSL-KDD dataset already downloaded")
        return train_path, test_path
    
    print("📥 Downloading NSL-KDD dataset...")
    import urllib.request
    
    base_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    
    try:
        urllib.request.urlretrieve(
            base_url + "KDDTrain+.txt",
            train_path
        )
        urllib.request.urlretrieve(
            base_url + "KDDTest+.txt",
            test_path
        )
        print("✅ Download complete")
        return train_path, test_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Please manually download from: https://github.com/defcom17/NSL_KDD")
        return None, None


def load_nslkdd_data(path):
    """Load and preprocess NSL-KDD data"""
    print(f"\n📂 Loading {path.name}...")
    
    # Load data
    df = pd.read_csv(path, names=NSLKDD_COLUMNS, header=None)
    
    print(f"✅ Loaded {len(df):,} samples")
    
    # Convert label to binary (normal=0, attack=1)
    df['label_binary'] = (df['label'] != 'normal').astype(int)
    
    print(f"   Benign: {sum(df['label_binary']==0):,}")
    print(f"   Attack: {sum(df['label_binary']==1):,}")
    
    # Separate features and labels
    y = df['label_binary']
    X = df.drop(columns=['label', 'label_binary', 'difficulty'])
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    print(f"\n🔧 Encoding categorical features...")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Ensure all numeric
    X = X.astype(float)
    
    return X, y


def align_features_to_cicids(X_nslkdd, cicids_scaler):
    """
    Align NSL-KDD features to CICIDS feature space
    NSL-KDD has 41 features, CICIDS has 79
    Strategy: Use available features, pad rest with zeros
    """
    print("\n🔄 Aligning NSL-KDD features to CICIDS feature space...")
    
    cicids_features = list(cicids_scaler.feature_names_in_)
    nslkdd_features = list(X_nslkdd.columns)
    
    print(f"   CICIDS expects: {len(cicids_features)} features")
    print(f"   NSL-KDD has: {len(nslkdd_features)} features")
    
    # Create feature mapping (some features may have similar names)
    # For simplicity, we'll create a zero-padded feature set
    X_aligned = pd.DataFrame(0.0, index=X_nslkdd.index, columns=cicids_features)
    
    # Map common features (case-insensitive matching)
    cicids_lower = {f.lower(): f for f in cicids_features}
    
    for nsl_col in nslkdd_features:
        nsl_lower = nsl_col.lower().replace('_', ' ')
        
        # Try to find matching CICIDS feature
        if nsl_lower in cicids_lower:
            cicids_col = cicids_lower[nsl_lower]
            X_aligned[cicids_col] = X_nslkdd[nsl_col].values
            print(f"   ✓ Mapped: {nsl_col} → {cicids_col}")
    
    # For unmapped features, we'll just leave them as zeros
    # This represents "unknown" features from CICIDS perspective
    
    print(f"✅ Feature alignment complete")
    print(f"   Aligned shape: {X_aligned.shape}")
    
    return X_aligned


def evaluate_on_nslkdd():
    """Cross-validate CICIDS-trained model on NSL-KDD"""
    print("\n" + "="*70)
    print("CROSS-DATASET VALIDATION: CICIDS → NSL-KDD")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Download NSL-KDD
    train_path, test_path = download_nslkdd()
    if not train_path:
        return
    
    # Load NSL-KDD test data
    X_nslkdd, y_nslkdd = load_nslkdd_data(test_path)
    
    # Load CICIDS model and scaler
    print("\n🔧 Loading CICIDS-trained model...")
    MODEL_PATH = list(MODEL_DIR.glob("best_tcn_improved_*.keras"))[-1]
    SCALER_PATH = PROJECT_ROOT / "models" / "anomly_detector" / "network_level_attacks_anomality" / "scaler.pkl"
    
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    print(f"✅ Model loaded: {MODEL_PATH.name}")
    
    # Align NSL-KDD features to CICIDS
    X_aligned = align_features_to_cicids(X_nslkdd, scaler)
    
    # Clean and scale
    print("\n🧹 Preprocessing...")
    X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan)
    X_aligned = X_aligned.fillna(0)
    
    # Scale using CICIDS scaler
    X_scaled = scaler.transform(X_aligned)
    
    print(f"✅ Data ready: {X_scaled.shape}")
    
    # Load threshold (if available)
    THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"
    if THRESHOLD_PATH.exists():
        threshold_info = joblib.load(THRESHOLD_PATH)
        threshold = threshold_info['best_threshold']
        print(f"\n📊 Using threshold from CICIDS validation: {threshold:.2f}")
    else:
        threshold = 0.5
        print(f"\n📊 Using default threshold: {threshold:.2f}")
    
    # Predict
    print("\n🔮 Making predictions on NSL-KDD test set...")
    y_pred_proba = model.predict(X_scaled, verbose=0).flatten()
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_nslkdd, y_pred)
    precision = precision_score(y_nslkdd, y_pred, zero_division=0)
    recall = recall_score(y_nslkdd, y_pred, zero_division=0)
    f1 = f1_score(y_nslkdd, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_nslkdd, y_pred_proba)
    except:
        auc = 0.0
    
    cm = confusion_matrix(y_nslkdd, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("📊 CROSS-VALIDATION RESULTS: NSL-KDD")
    print("="*70)
    print(f"Accuracy:              {accuracy*100:.4f}%")
    print(f"Precision:             {precision*100:.4f}%")
    print(f"Recall (Detection):    {recall*100:.4f}%")
    print(f"F1-Score:              {f1*100:.4f}%")
    print(f"AUC-ROC:               {auc*100:.4f}%")
    
    print("\n" + "="*70)
    print("🚨 IPS CRITICAL METRICS")
    print("="*70)
    print(f"False Positive Rate:   {fpr*100:.4f}%")
    print(f"False Negative Rate:   {fnr*100:.4f}%")
    print(f"Detection Rate:        {detection_rate*100:.4f}%")
    
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
    print(classification_report(y_nslkdd, y_pred, digits=4, target_names=['Benign', 'Attack']))
    
    # Interpretation
    print("\n" + "="*70)
    print("💡 CROSS-DATASET GENERALIZATION ANALYSIS")
    print("="*70)
    
    if f1 > 0.90:
        print("🎉 EXCELLENT: Model generalizes very well to NSL-KDD")
        print("   → Strong evidence of learning generalizable patterns")
    elif f1 > 0.80:
        print("✅ GOOD: Model shows good generalization despite dataset differences")
        print("   → Acceptable performance drop from CICIDS to NSL-KDD")
    elif f1 > 0.70:
        print("⚠️  MODERATE: Significant performance drop on NSL-KDD")
        print("   → Model may be somewhat CICIDS-specific")
    else:
        print("❌ POOR: Substantial performance degradation")
        print("   → Model appears highly CICIDS-specific, limited generalization")
    
    print("\nNote: Some performance drop is expected due to:")
    print("  • Different feature sets (41 vs 79)")
    print("  • Different attack types and distributions")
    print("  • Different network conditions and traffic patterns")
    print("="*70 + "\n")
    
    # Save results
    results = {
        'timestamp': timestamp,
        'dataset': 'NSL-KDD',
        'model_trained_on': 'CICIDS2017',
        'threshold': float(threshold),
        'test_samples': len(y_nslkdd),
        'benign_samples': int(sum(y_nslkdd==0)),
        'attack_samples': int(sum(y_nslkdd==1)),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'detection_rate': float(detection_rate),
        'confusion_matrix': {
            'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP)
        }
    }
    
    results_file = RESULTS_DIR / f"nslkdd_cross_validation_{timestamp}.pkl"
    joblib.dump(results, results_file)
    print(f"💾 Results saved to: {results_file}\n")
    
    return results


if __name__ == "__main__":
    evaluate_on_nslkdd()
