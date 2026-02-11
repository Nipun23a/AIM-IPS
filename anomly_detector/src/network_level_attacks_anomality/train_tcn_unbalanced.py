# train_tcn_imbalanced.py
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# GPU Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU enabled:", gpus)
    except RuntimeError as e:
        print("⚠️ GPU setup error:", e)
else:
    print("⚠️ GPU not found, using CPU")


# ============================================================================
# DIRECTORY SETUP
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[3] / "models" / "anomly_detector" / "network_level_attacks_anomality"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DATASET_DIR = Path(__file__).resolve().parents[3] / "data_collector" / "data_sets" / "cicids"

TCN_DIR = BASE_DIR / "tcn"
PLOTS_DIR = TCN_DIR / "plots"
MODELS_DIR = TCN_DIR / "models"
LOGS_DIR = TCN_DIR / "logs"

for d in [BASE_DIR, TCN_DIR, PLOTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

SCALER_PATH = BASE_DIR / "scaler.pkl"

print(f"📁 Output directory: {TCN_DIR}")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def normalize_colname(c: str) -> str:
    """Normalize column names to lowercase with underscores"""
    c = str(c).strip()
    c = c.replace(' ', '_').replace('-', '_')
    return c.lower()


def load_cicids_parts():
    """
    Load all 8 CICIDS parts and combine them
    """
    print("\n" + "="*70)
    print("LOADING CICIDS2017 DATASET (8 PARTS)")
    print("="*70)
    
    all_parts = []
    
    for i in range(1, 9):
        part_path = DATASET_DIR / f"cicids_train_part{i}.csv"
        
        if not part_path.exists():
            print(f"❌ Part {i} not found: {part_path}")
            continue
        
        print(f"\n📂 Loading part {i}...")
        df = pd.read_csv(part_path, low_memory=False)
        
        # Normalize column names
        df.columns = [normalize_colname(c) for c in df.columns]
        
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        
        all_parts.append(df)
    
    if not all_parts:
        raise FileNotFoundError("No CICIDS parts found!")
    
    # Combine all parts
    print(f"\n🔗 Combining {len(all_parts)} parts...")
    combined_df = pd.concat(all_parts, ignore_index=True)
    
    print(f"\n✅ Combined dataset:")
    print(f"   Total rows: {len(combined_df):,}")
    print(f"   Total columns: {len(combined_df.columns)}")
    print(f"   Sample columns: {combined_df.columns.tolist()[:5]}")
    
    return combined_df


def prepare_imbalanced_data(df):
    """
    Prepare data keeping the NATURAL IMBALANCED distribution
    Split into train (70%), val (15%), test (15%)
    """
    print("\n" + "="*70)
    print("PREPARING DATA WITH NATURAL IMBALANCE")
    print("="*70)
    
    # Find label column
    if 'label' not in df.columns:
        raise ValueError("'label' column not found!")
    
    # Convert labels to binary (0=benign, 1=attack)
    if df['label'].dtype == 'object':
        # String labels
        df['label_binary'] = (df['label'].str.upper() != 'BENIGN').astype(int)
    else:
        # Already numeric
        df['label_binary'] = (df['label'] != 0).astype(int)
    
    # Show original distribution
    n_benign = sum(df['label_binary'] == 0)
    n_attack = sum(df['label_binary'] == 1)
    total = len(df)
    
    print(f"\n📊 Original distribution:")
    print(f"   Benign: {n_benign:,} ({n_benign/total*100:.2f}%)")
    print(f"   Attack: {n_attack:,} ({n_attack/total*100:.2f}%)")
    print(f"   Imbalance ratio: {n_benign/n_attack:.2f}:1")
    
    # Separate features and labels
    y = df['label_binary'].copy()
    
    # Drop non-feature columns
    drop_cols = ['label', 'label_binary']
    
    # Also drop identifier columns
    id_cols = ['flow_id', 'id', 'timestamp', 'start_time', 'end_time', 'time', 
               'flow_id.1', 'srcip', 'dstip', 'src_ip', 'dst_ip', 'proto', 
               'service', 'state']
    
    for col in id_cols:
        if col in df.columns:
            drop_cols.append(col)
    
    X = df.drop(columns=drop_cols, errors='ignore')
    
    print(f"\n🔧 Feature extraction:")
    print(f"   Dropped columns: {drop_cols}")
    print(f"   Feature columns: {len(X.columns)}")
    print(f"   Sample features: {X.columns.tolist()[:5]}")
    
    # Clean data
    print(f"\n🧹 Cleaning data...")
    print(f"   Replacing inf values...")
    X = X.replace([np.inf, -np.inf], np.nan)
    
    print(f"   Filling NaN values with median...")
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median = X[col].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            X[col] = X[col].fillna(median)
        else:
            # Convert non-numeric to numeric
            X[col] = pd.to_numeric(X[col], errors='coerce')
            median = X[col].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            X[col] = X[col].fillna(median)
    
    print(f"✅ Data cleaned")
    
    # Split: 70% train, 15% val, 15% test (stratified to maintain distribution)
    print(f"\n✂️  Creating train/val/test splits...")
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
    
    # Second split: 15% val, 15% test (50-50 split of the 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
    
    # Show split distributions
    print(f"\n📊 Train set (70%):")
    print(f"   Total: {len(X_train):,}")
    print(f"   Benign: {sum(y_train==0):,} ({sum(y_train==0)/len(y_train)*100:.2f}%)")
    print(f"   Attack: {sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.2f}%)")
    
    print(f"\n📊 Validation set (15%):")
    print(f"   Total: {len(X_val):,}")
    print(f"   Benign: {sum(y_val==0):,} ({sum(y_val==0)/len(y_val)*100:.2f}%)")
    print(f"   Attack: {sum(y_val==1):,} ({sum(y_val==1)/len(y_val)*100:.2f}%)")
    
    print(f"\n📊 Test set (15%):")
    print(f"   Total: {len(X_test):,}")
    print(f"   Benign: {sum(y_test==0):,} ({sum(y_test==0)/len(y_test)*100:.2f}%)")
    print(f"   Attack: {sum(y_test==1):,} ({sum(y_test==1)/len(y_test)*100:.2f}%)")
    
    # Standardize features
    print(f"\n📏 Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾 Scaler saved to: {SCALER_PATH}")
    
    # Save feature info
    feature_info = {
        'features': X.columns.tolist(),
        'n_features': len(X.columns),
        'timestamp': TIMESTAMP
    }
    joblib.dump(feature_info, BASE_DIR / "feature_info.pkl")
    
    print(f"\n✅ Data preparation complete!")
    print(f"   Input dimension: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


# ============================================================================
# TCN MODEL ARCHITECTURE
# ============================================================================
def build_tcn_block(x, filters, kernel_size, dilation_rate, dropout):
    """Build a TCN residual block"""
    c1 = layers.Conv1D(
        filters, kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    c1 = layers.SpatialDropout1D(dropout)(c1)

    c2 = layers.Conv1D(
        filters, kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(c1)
    c2 = layers.SpatialDropout1D(dropout)(c2)

    # Residual connection
    res = x if x.shape[-1] == filters else layers.Conv1D(filters, 1)(x)
    return layers.Activation("relu")(layers.Add()([c2, res]))


def build_imbalanced_tcn(input_dim, num_blocks=4, filters=64, 
                         kernel_size=3, dropout=0.2):
    """
    Build TCN optimized for IMBALANCED data
    """
    inp = layers.Input(shape=(input_dim,), name='input')
    x = layers.Reshape((input_dim, 1))(inp)

    # Initial projection
    x = layers.Conv1D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # TCN blocks
    for i in range(num_blocks):
        x = build_tcn_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=2 ** i,
            dropout=dropout
        )

    # Global pooling and dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(inp, output, name='tcn_imbalanced')
    
    print("\n✅ TCN MODEL CREATED (Optimized for Imbalanced Data)")
    print(f"   Parameters: {model.count_params():,}")
    
    return model


# ============================================================================
# TRAINING
# ============================================================================
def train_on_imbalanced_data(X_train, y_train, X_val, y_val, input_dim, epochs=50):
    """
    Train TCN on imbalanced data with focal loss and class weights
    """
    print("\n" + "="*70)
    print("TRAINING ON IMBALANCED DATA")
    print("="*70)
    
    # Build model
    model = build_imbalanced_tcn(input_dim, num_blocks=4, filters=64)
    
    # Calculate class weights
    n_benign = np.sum(y_train == 0)
    n_attack = np.sum(y_train == 1)
    total = len(y_train)
    
    # Stronger weight for minority class
    class_weight = {
        0: total / (2 * n_benign),
        1: total / (2 * n_attack) * 2.0  # Extra weight for attacks
    }
    
    print(f"\n⚖️  Class weights (boosted for minority):")
    print(f"   Class 0 (Benign): {class_weight[0]:.4f}")
    print(f"   Class 1 (Attack): {class_weight[1]:.4f}")
    
    # Compile with focal loss characteristics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # Callbacks
    model_save_path = MODELS_DIR / f"best_tcn_imbalanced_{TIMESTAMP}.keras"
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(LOGS_DIR / f"training_log_imbalanced_{TIMESTAMP}.csv")
        )
    ]
    
    # Train
    print(f"\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=128,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # Save
    joblib.dump(history.history, LOGS_DIR / f"history_imbalanced_{TIMESTAMP}.pkl")
    print(f"\n💾 Model saved to: {model_save_path}")
    
    return model, history


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model, X_test, y_test):
    """Comprehensive evaluation"""
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET (IMBALANCED)")
    print("="*70)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Test Performance:")
    print(f"   Accuracy:  {accuracy*100:.4f}%")
    print(f"   Precision: {precision*100:.4f}%")
    print(f"   Recall:    {recall*100:.4f}%")
    print(f"   F1-Score:  {f1*100:.4f}%")
    print(f"   AUC-ROC:   {auc*100:.4f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"              Predicted")
    print(f"            Benign  Attack")
    print(f"Actual Benign {TN:6d}  {FP:6d}")
    print(f"       Attack {FN:6d}  {TP:6d}")
    
    # IPS metrics
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    print(f"\n🚨 IPS Metrics:")
    print(f"   False Positive Rate: {fpr*100:.4f}%")
    print(f"   False Negative Rate: {fnr*100:.4f}%")
    print(f"   Detection Rate:      {detection_rate*100:.4f}%")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack'], digits=4))
    
    # Save metrics
    metrics = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'auc': auc, 'fpr': fpr, 'fnr': fnr,
        'detection_rate': detection_rate, 'confusion_matrix': cm
    }
    joblib.dump(metrics, LOGS_DIR / f"test_metrics_imbalanced_{TIMESTAMP}.pkl")
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================
def main(epochs=50):
    """Main training pipeline"""
    print("\n" + "="*70)
    print("TCN TRAINING ON IMBALANCED CICIDS2017")
    print("="*70)
    
    # Step 1: Load all 8 parts
    df = load_cicids_parts()
    
    # Step 2: Prepare data (70/15/15 split, keeping natural imbalance)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_imbalanced_data(df)
    
    input_dim = X_train.shape[1]
    
    # Step 3: Train
    model, history = train_on_imbalanced_data(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        epochs=epochs
    )
    
    # Step 4: Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 5: Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📁 All outputs saved to: {TCN_DIR}")
    print(f"   - Best model: {MODELS_DIR}")
    print(f"   - Scaler: {SCALER_PATH}")
    print(f"   - Logs: {LOGS_DIR}")
    print("="*70 + "\n")
    
    return model, history, metrics


if __name__ == "__main__":
    model, history, metrics = main(epochs=50)