# threshold_optimizer.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "anomly_detector" / "network_level_attacks_anomality" / "tcn" / "models"
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets"
SCALER_PATH = PROJECT_ROOT / "models" / "anomly_detector" / "network_level_attacks_anomality" / "scaler.pkl"

# Paths
TRAIN_PATH = DATA_DIR / "cicids_train.csv"  # Your training data
THRESHOLD_SAVE_PATH = MODEL_DIR / "best_threshold.pkl"

def load_and_prepare_data(path, scaler):
    """Load data with same preprocessing"""
    df = pd.read_csv(path, low_memory=False)
    
    # Separate features and labels
    y = df['label'].copy().astype(int)
    X = df.copy()
    
    # Clean
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median = X[col].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            X[col] = X[col].fillna(median)
    
    # Scale
    scaler_features = list(scaler.feature_names_in_)
    X = X[scaler_features]
    X_scaled = scaler.transform(X)
    
    return X_scaled, y


def find_best_threshold_on_validation():
    """
    Find best threshold using VALIDATION data from training
    This should be run ONCE after training, BEFORE testing
    """
    print("="*70)
    print("FINDING BEST THRESHOLD ON VALIDATION DATA")
    print("="*70)
    
    # Load model
    MODEL_PATH = list(MODEL_DIR.glob("best_tcn_improved_*.keras"))[-1]
    print(f"\n🔧 Loading model: {MODEL_PATH.name}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load scaler
    print(f"📏 Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    
    # Load FULL training data
    print(f"\n📂 Loading training data...")
    X_full, y_full = load_and_prepare_data(TRAIN_PATH, scaler)
    
    # Create validation split (same as training: 20% validation)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, 
        test_size=0.2, 
        random_state=42,  # SAME random state as training!
        stratify=y_full
    )
    
    print(f"✅ Validation set: {len(X_val):,} samples")
    print(f"   Benign: {sum(y_val==0):,}, Attack: {sum(y_val==1):,}")
    
    # Get predictions on VALIDATION set
    print(f"\n🔮 Getting predictions on validation set...")
    y_val_proba = model.predict(X_val, verbose=0).flatten()
    
    # Try different thresholds ON VALIDATION DATA
    print(f"\n🔬 Testing thresholds on VALIDATION data:")
    best_threshold = 0.5
    best_f1 = 0
    
    results = []
    for threshold in np.arange(0.05, 0.55, 0.05):
        y_pred = (y_val_proba > threshold).astype(int)
        
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"   Threshold {threshold:.2f}: Precision={precision*100:5.2f}%, "
              f"Recall={recall*100:5.2f}%, F1={f1*100:5.2f}%")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1*100:.2f}%)")
    
    # Save best threshold
    threshold_info = {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'validation_size': len(y_val),
        'all_results': results
    }
    
    joblib.dump(threshold_info, THRESHOLD_SAVE_PATH)
    print(f"💾 Threshold saved to: {THRESHOLD_SAVE_PATH}")
    
    return best_threshold, best_f1


if __name__ == "__main__":
    find_best_threshold_on_validation()