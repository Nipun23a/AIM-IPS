# threshold_optimizer_fixed.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "anomly_detector" / "network_level_attacks_anomality" / "tcn" / "models"
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets"
SCALER_PATH = PROJECT_ROOT / "models" / "anomly_detector" / "network_level_attacks_anomality" / "scaler.pkl"

# Use the ALREADY SPLIT training file
TRAIN_DATA_PATH = DATA_DIR / "cicids_train.csv"
THRESHOLD_SAVE_PATH = MODEL_DIR / "best_threshold.pkl"


def create_imbalanced_validation_from_training(benign_ratio=0.95, n_samples=20000, random_state=999):
    """
    Create imbalanced validation set from cicids_train.csv
    This ensures NO overlap with test set (cicids_test.csv)
    """
    print("="*70)
    print("CREATING IMBALANCED VALIDATION SET FROM TRAINING DATA")
    print("="*70)
    
    # Load TRAINING data only
    print(f"\n📂 Loading training data: {TRAIN_DATA_PATH}")
    train_df = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    
    print(f"✅ Loaded {len(train_df):,} training samples")
    
    # Separate by class
    benign_samples = train_df[train_df['label'] == 0]
    attack_samples = train_df[train_df['label'] == 1]
    
    print(f"\n📊 Training data distribution:")
    print(f"  Benign: {len(benign_samples):,}")
    print(f"  Attack: {len(attack_samples):,}")
    
    # Calculate samples needed
    n_benign = int(n_samples * benign_ratio)
    n_attack = n_samples - n_benign
    
    print(f"\n🎯 Creating imbalanced validation set:")
    print(f"  Target: {n_samples:,} samples")
    print(f"  Benign: {n_benign:,} ({benign_ratio*100:.1f}%)")
    print(f"  Attack: {n_attack:,} ({(1-benign_ratio)*100:.1f}%)")
    
    # Sample from TRAINING data
    # Using different random state than test split (999 vs 42)
    sampled_benign = benign_samples.sample(
        n=min(n_benign, len(benign_samples)), 
        replace=False, 
        random_state=random_state
    )
    sampled_attack = attack_samples.sample(
        n=min(n_attack, len(attack_samples)), 
        replace=False, 
        random_state=random_state
    )
    
    # Combine and shuffle
    validation_df = pd.concat([sampled_benign, sampled_attack], ignore_index=True)
    validation_df = validation_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"✅ Validation set created: {len(validation_df):,} samples")
    print(f"   (Guaranteed NO overlap with test set)")
    
    return validation_df


def find_best_threshold():
    """
    Find best threshold using imbalanced validation from TRAINING data
    """
    print("="*70)
    print("THRESHOLD OPTIMIZATION ON VALIDATION DATA")
    print("="*70)
    
    # Load model
    MODEL_PATH = list(MODEL_DIR.glob("best_tcn_improved_*.keras"))[-1]
    print(f"\n🔧 Loading model: {MODEL_PATH.name}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load scaler
    print(f"📏 Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    
    # Create validation set from TRAINING data (no test contamination)
    validation_df = create_imbalanced_validation_from_training(
        benign_ratio=0.95,
        n_samples=20000,
        random_state=999  # Different from test split (42)
    )
    
    # Prepare data
    print(f"\n🧹 Preprocessing validation data...")
    y_val = validation_df['label'].copy().astype(int)
    X_val = validation_df.copy()
    
    # Clean (same as training)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    for col in X_val.columns:
        if col == 'label':
            continue
        if pd.api.types.is_numeric_dtype(X_val[col]):
            median = X_val[col].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            X_val[col] = X_val[col].fillna(median)
    
    # Match scaler features
    scaler_features = list(scaler.feature_names_in_)
    X_val = X_val[scaler_features]
    X_val_scaled = scaler.transform(X_val)
    
    print(f"✅ Validation data ready:")
    print(f"   Shape: {X_val_scaled.shape}")
    print(f"   Benign: {sum(y_val==0):,} ({sum(y_val==0)/len(y_val)*100:.2f}%)")
    print(f"   Attack: {sum(y_val==1):,} ({sum(y_val==1)/len(y_val)*100:.2f}%)")
    
    # Predict
    print(f"\n🔮 Getting predictions...")
    y_val_proba = model.predict(X_val_scaled, verbose=0).flatten()
    
    # Try thresholds
    print(f"\n🔬 Testing thresholds on imbalanced validation:")
    best_threshold = 0.5
    best_f1 = 0
    results = []
    
    for threshold in np.arange(0.05, 0.55, 0.05):
        y_pred = (y_val_proba > threshold).astype(int)
        
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_val, y_pred)
        TN, FP, FN, TP = cm.ravel()
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr
        })
        
        print(f"   Threshold {threshold:.2f}: "
              f"Precision={precision*100:5.2f}%, "
              f"Recall={recall*100:5.2f}%, "
              f"F1={f1*100:5.2f}%, "
              f"FPR={fpr*100:6.4f}%")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1={best_f1*100:.2f}%)")
    
    # Save
    threshold_info = {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'validation_size': len(y_val),
        'validation_source': 'cicids_train.csv subset',
        'validation_distribution': {
            'benign': int(sum(y_val==0)),
            'attack': int(sum(y_val==1))
        },
        'all_results': results
    }
    
    joblib.dump(threshold_info, THRESHOLD_SAVE_PATH)
    print(f"💾 Saved to: {THRESHOLD_SAVE_PATH}")
    
    return best_threshold


if __name__ == "__main__":
    find_best_threshold()