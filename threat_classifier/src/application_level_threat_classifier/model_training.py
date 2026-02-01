# threat_classifier/src/application_level_threat_classifier/model_training.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path
import json
import time

from threat_classifier.src.application_level_threat_classifier.feature_engineering import ThreatFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets" / "application_level"
MODEL_DIR = PROJECT_ROOT / "models" / "application_layer"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ThreatClassifierTrainer:
    """
    Train LightGBM model for application-layer threat detection.
    Handles class imbalance and provides comprehensive evaluation.
    """
    
    def __init__(self):
        self.model = None
        self.feature_extractor = ThreatFeatureExtractor()
        self.class_mapping = None
        self.feature_importance = None
        
    def load_data(self):
        """Load the combined dataset"""
        print("\n" + "="*80)
        print("LOADING DATASET")
        print("="*80)
        
        data_path = DATA_DIR / "combined_threat_dataset.csv"
        df = pd.read_csv(data_path)
        
        # Remove auth-bruteforce if exists (only 4 samples)
        df = df[df['attack_type'] != 'auth-bruteforce']
        
        print(f"[INFO] Dataset loaded: {len(df)} samples")
        print(f"\n[INFO] Class distribution:")
        print(df['attack_type'].value_counts())
        
        print(f"\n[INFO] Class distribution (%):")
        print(df['attack_type'].value_counts(normalize=True) * 100)
        
        return df
    
    def prepare_features(self, df):
        """Extract features from payloads"""
        print("\n" + "="*80)
        print("FEATURE EXTRACTION")
        print("="*80)
        
        X = self.feature_extractor.extract_batch_features(df['payload'])
        y = df['attack_type']
        
        # Create class mapping
        self.class_mapping = {label: idx for idx, label in enumerate(sorted(y.unique()))}
        self.inverse_class_mapping = {idx: label for label, idx in self.class_mapping.items()}
        
        print(f"\n[INFO] Class mapping:")
        for label, idx in self.class_mapping.items():
            print(f"  {label}: {idx}")
        
        # Convert labels to integers
        y_encoded = y.map(self.class_mapping)
        
        return X, y_encoded, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        print("\n" + "="*80)
        print("DATA SPLITTING")
        print("="*80)
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"[INFO] Train set: {len(X_train)} samples")
        print(f"[INFO] Validation set: {len(X_val)} samples")
        print(f"[INFO] Test set: {len(X_test)} samples")
        
        print(f"\n[INFO] Train set distribution:")
        print(pd.Series(y_train).value_counts())
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def compute_class_weights(self, y_train):
        """Compute class weights for handling imbalance"""
        print("\n" + "="*80)
        print("CLASS WEIGHT COMPUTATION")
        print("="*80)
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        
        print(f"[INFO] Class weights:")
        for class_idx, weight in weight_dict.items():
            class_name = self.inverse_class_mapping[class_idx]
            print(f"  {class_name} ({class_idx}): {weight:.2f}")
        
        return weight_dict
    
    def train_model(self, X_train, y_train, X_val, y_val, class_weights):
        """Train LightGBM model"""
        print("\n" + "="*80)
        print("MODEL TRAINING")
        print("="*80)
        
        # Prepare sample weights
        sample_weights = np.array([class_weights[y] for y in y_train])
        
        # LightGBM parameters
        lgb_params = {
            'objective': 'multiclass',
            'num_class': len(self.class_mapping),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 8,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        print(f"[INFO] Model parameters:")
        for key, value in lgb_params.items():
            print(f"  {key}: {value}")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        print(f"\n[INFO] Starting training...")
        start_time = time.time()
        
        self.model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50)
            ]
        )
        
        training_time = time.time() - start_time
        print(f"\n[INFO] Training completed in {training_time:.2f} seconds")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_extractor.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"\n[INFO] Top 20 most important features:")
        print(self.feature_importance.head(20))
        
        return self.model
    
    def save_model(self):
        """Save model and related artifacts"""
        print("\n" + "="*80)
        print("SAVING MODEL")
        print("="*80)
        
        # Save LightGBM model
        model_path = MODEL_DIR / "threat_classifier_lgb.pkl"
        joblib.dump(self.model, model_path)
        print(f"[INFO] Model saved to {model_path}")
        
        # Save feature configuration
        feature_config_path = MODEL_DIR / "feature_config.pkl"
        self.feature_extractor.save_feature_config(feature_config_path)
        
        # Save class mapping
        class_mapping_path = MODEL_DIR / "class_mapping.json"
        with open(class_mapping_path, 'w') as f:
            json.dump({
                'class_to_idx': self.class_mapping,
                'idx_to_class': self.inverse_class_mapping
            }, f, indent=2)
        print(f"[INFO] Class mapping saved to {class_mapping_path}")
        
        # Save feature importance
        importance_path = MODEL_DIR / "feature_importance.csv"
        self.feature_importance.to_csv(importance_path, index=False)
        print(f"[INFO] Feature importance saved to {importance_path}")
        
        print(f"\n[INFO] All artifacts saved to {MODEL_DIR}")


def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("APPLICATION LAYER THREAT CLASSIFIER - TRAINING PIPELINE")
    print("="*80)
    
    trainer = ThreatClassifierTrainer()
    
    # Step 1: Load data
    df = trainer.load_data()
    
    # Step 2: Extract features
    X, y_encoded, y_original = trainer.prepare_features(df)
    
    # Step 3: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y_encoded)
    
    # Step 4: Compute class weights
    class_weights = trainer.compute_class_weights(y_train)
    
    # Step 5: Train model
    model = trainer.train_model(X_train, y_train, X_val, y_val, class_weights)
    
    # Step 6: Save model
    trainer.save_model()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Run model evaluation: python model_evaluation.py")
    print(f"2. Test inference: python inference.py")
    

if __name__ == "__main__":
    main()