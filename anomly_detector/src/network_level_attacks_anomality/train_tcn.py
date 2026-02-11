import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import joblib


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
DATASET_DIR = Path(__file__).resolve().parents[3] / "data_collector" / "data_sets"  

TCN_DIR = BASE_DIR / "tcn"
PLOTS_DIR = TCN_DIR / "plots"
MODELS_DIR = TCN_DIR / "models"
LOGS_DIR = TCN_DIR / "logs"

# Create all directories
for d in [BASE_DIR, TCN_DIR, PLOTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"TCN outputs will be saved to: {TCN_DIR}")

# Scaler will be saved to the base anomaly detector directory
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_csv(path):
    """Load CSV file"""
    return pd.read_csv(path)


def intersect_features(dfs: List[pd.DataFrame]) -> List[str]:
    """Return intersection of column names across provided DataFrames"""
    sets = [set(df.columns) for df in dfs]
    inter = set.intersection(*sets)
    inter = list(inter)
    # Remove columns that are obviously non-numeric or identifiers we don't want
    blacklist = {'srcip', 'proto', 'service', 'state', 'dstip', 'Timestamp', 
                 'Flow ID', 'Label', 'attack_cat', 'attack_category', 'id', 
                 'start_time', 'end_time'}
    inter = [c for c in inter if c not in blacklist]
    # Keep stable order
    inter.sort()
    return inter


def prepare_from_paths(paths, label_col_candidates=None):
    """Load and prepare datasets from paths"""
    # load all datasets
    dfs = [load_csv(p) for p in paths]
    features = intersect_features(dfs)
    print(f"[prepare] Found {len(features)} intersecting features.")
    
    # For each df, keep only intersecting features; drop rows with NaN in those features
    Xs = []
    Ys = []
    for df in dfs:
        dff = df[features].copy()
        # try to find a label column (common names)
        label = None
        possible = ['Label', 'label', 'Attack', 'attack', 'class', 'Class', 'attack_cat']
        for cand in possible:
            if cand in df.columns:
                label = df[cand]
                break
        # if label not found, mark as unknown (-1)
        if label is None:
            label = pd.Series([-1]*len(dff))
        # drop rows with missing values in features
        mask = ~dff.isnull().any(axis=1)
        Xs.append(dff[mask].astype(float))
        Ys.append(label[mask])
    
    # concat all
    X = pd.concat(Xs, ignore_index=True)
    Y = pd.concat(Ys, ignore_index=True)
    return X, Y, features


# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def preprocess_cicids_data(train_path, test_path):
    """
    Load and preprocess CICIDS train and test datasets
    Returns preprocessed data + saves scaler for later use
    """
    print("\n" + "="*70)
    print("LOADING AND PREPROCESSING CICIDS2017 DATASET")
    print("="*70)
    
    # Load data using your existing function
    print(f"\n📂 Loading data...")
    X_train_raw, y_train_raw, features = prepare_from_paths([train_path])
    X_test_raw, y_test_raw, _ = prepare_from_paths([test_path])
    
    print(f"   Train shape: {X_train_raw.shape}")
    print(f"   Test shape: {X_test_raw.shape}")
    print(f"   Number of features: {len(features)}")
    
    # Convert labels to binary (0 = benign, 1 = attack)
    def convert_to_binary(y):
        if y.dtype == 'object':
            # String labels
            binary = (y.str.upper() != 'BENIGN').astype(int)
        else:
            # Numeric labels (assuming 0 = benign)
            binary = (y != 0).astype(int)
        return binary
    
    y_train = convert_to_binary(y_train_raw)
    y_test = convert_to_binary(y_test_raw)
    
    print(f"\n📊 Label distribution:")
    print(f"   Train - Benign: {sum(y_train==0):,} ({sum(y_train==0)/len(y_train)*100:.1f}%), "
          f"Attack: {sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.1f}%)")
    print(f"   Test  - Benign: {sum(y_test==0):,} ({sum(y_test==0)/len(y_test)*100:.1f}%), "
          f"Attack: {sum(y_test==1):,} ({sum(y_test==1)/len(y_test)*100:.1f}%)")
    
    # Handle infinite values
    print("\n🔧 Cleaning data...")
    X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
    X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with 0
    X_train_raw = X_train_raw.fillna(0)
    X_test_raw = X_test_raw.fillna(0)
    
    # Standardize features
    print("   Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 🔥 SAVE SCALER FOR LATER USE
    print(f"\n💾 Saving scaler to: {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved successfully!")
    
    # Create validation split from training data (20% for validation)
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Final train shape: {X_train_final.shape}")
    print(f"   Validation shape: {X_val.shape}")
    print(f"   Test shape: {X_test_scaled.shape}")
    print(f"   Input dimension: {X_train_final.shape[1]}")
    
    # Save feature names for reference
    feature_info = {
        'features': features,
        'n_features': len(features),
        'timestamp': TIMESTAMP
    }
    feature_path = BASE_DIR / "feature_info.pkl"
    joblib.dump(feature_info, feature_path)
    print(f"   Feature info saved to: {feature_path}")
    
    return X_train_final, X_val, X_test_scaled, y_train_final, y_val, y_test, scaler, features


# ============================================================================
# TCN MODEL ARCHITECTURE
# ============================================================================
def build_tcn_block(x, filters, kernel_size, dilation_rate, dropout):
    """Build a TCN residual block with causal convolutions"""
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


def build_improved_tcn(input_dim, num_blocks=4, filters=64, 
                       kernel_size=3, dropout=0.2):
    """
    Build improved TCN for better accuracy on CICIDS2017
    Increased capacity for better performance
    """
    inp = layers.Input(shape=(input_dim,), name='input')
    x = layers.Reshape((input_dim, 1))(inp)

    # Initial projection
    x = layers.Conv1D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # TCN blocks with increasing dilation
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
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(inp, output, name='improved_tcn')
    
    print("\n✅ IMPROVED TCN CREATED")
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Model size: {model.count_params() * 4 / 1024:.1f} KB")
    
    return model


def build_lightweight_tcn(input_dim, num_blocks=3, filters=32,
                          kernel_size=3, dropout=0.2):
    """Original lightweight TCN (for comparison)"""
    inp = layers.Input(shape=(input_dim,), name='input')
    x = layers.Reshape((input_dim, 1))(inp)

    x = layers.Conv1D(filters, 1, padding='same')(x)

    for i in range(num_blocks):
        x = build_tcn_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=2 ** i,
            dropout=dropout
        )

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(inp, output, name='lightweight_tcn')
    
    print("\n✅ LIGHTWEIGHT TCN CREATED")
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Model size: {model.count_params() * 4 / 1024:.1f} KB")

    return model


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
def train_tcn(X_train, y_train, X_val, y_val, input_dim,
              model_type='improved', epochs=50, batch_size=128):
    """
    Train TCN model with proper callbacks and monitoring
    """
    print("\n" + "="*70)
    print("TRAINING TCN MODEL")
    print("="*70)
    
    # Build model
    if model_type == 'improved':
        model = build_improved_tcn(input_dim, num_blocks=4, filters=64)
    else:
        model = build_lightweight_tcn(input_dim, num_blocks=3, filters=32)
    
    # Calculate class weights for imbalanced data
    n_benign = np.sum(y_train == 0)
    n_attack = np.sum(y_train == 1)
    total = len(y_train)
    
    class_weight = {
        0: total / (2 * n_benign),
        1: total / (2 * n_attack)
    }
    print(f"\n⚖️  Class distribution:")
    print(f"   Benign: {n_benign:,} ({n_benign/total*100:.1f}%)")
    print(f"   Attack: {n_attack:,} ({n_attack/total*100:.1f}%)")
    print(f"   Class weights: {class_weight}")
    
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
    model_save_path = MODELS_DIR / f"best_tcn_{model_type}_{TIMESTAMP}.keras"
    
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
            filename=str(LOGS_DIR / f"training_log_{model_type}_{TIMESTAMP}.csv"),
            separator=',',
            append=False
        )
    ]
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Best model will be saved to: {model_save_path}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # Save final model
    final_path = MODELS_DIR / f"final_tcn_{model_type}_{TIMESTAMP}.keras"
    model.save(str(final_path))
    print(f"\n💾 Final model saved to: {final_path}")
    
    # Save training history
    history_path = LOGS_DIR / f"history_{model_type}_{TIMESTAMP}.pkl"
    joblib.dump(history.history, history_path)
    print(f"💾 Training history saved to: {history_path}")
    
    return model, history


def evaluate_model(model, X_test, y_test, model_name="TCN"):
    """
    Comprehensive model evaluation
    """
    print("\n" + "="*70)
    print(f"EVALUATING {model_name} ON TEST SET")
    print("="*70)
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Test Set Performance:")
    print(f"   Accuracy:  {accuracy*100:.4f}%")
    print(f"   Precision: {precision*100:.4f}%")
    print(f"   Recall:    {recall*100:.4f}%")
    print(f"   F1-Score:  {f1*100:.4f}%")
    print(f"   AUC-ROC:   {auc*100:.4f}%")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Benign', 'Attack'],
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Benign  Attack")
    print(f"Actual Benign  {TN:6d}  {FP:6d}")
    print(f"       Attack  {FN:6d}  {TP:6d}")
    
    # IPS-specific metrics
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    print(f"\n🚨 IPS Critical Metrics:")
    print(f"   False Positive Rate: {fpr*100:.4f}%")
    print(f"   False Negative Rate: {fnr*100:.4f}%")
    print(f"   Detection Rate:      {detection_rate*100:.4f}%")
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'fpr': fpr,
        'fnr': fnr,
        'detection_rate': detection_rate,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_true': y_test
    }
    
    metrics_path = LOGS_DIR / f"test_metrics_{TIMESTAMP}.pkl"
    joblib.dump(metrics, metrics_path)
    print(f"\n💾 Metrics saved to: {metrics_path}")
    
    return metrics


def plot_training_history(history, model_type):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training History - {model_type.upper()} TCN', fontsize=16)
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(history.history['auc'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_auc'], label='Validation', linewidth=2)
    axes[1, 0].set_title('AUC-ROC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 1].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 1].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"{model_type}_training_history_{TIMESTAMP}.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved to: {plot_path}")


def plot_evaluation_results(results, model_name):
    """Plot confusion matrix and ROC curve"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} - Test Set Evaluation', fontsize=16)
    
    # Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_proba'])
    axes[1].plot(fpr, tpr, linewidth=2, label=f'AUC = {results["auc"]:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1].set_title('ROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"{model_name}_evaluation_{TIMESTAMP}.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Evaluation plots saved to: {plot_path}")


def main(train_path, test_path, model_type='improved', epochs=50):
    """
    Main training and evaluation pipeline
    """
    print("\n" + "="*70)
    print("TCN ANOMALY DETECTION - CICIDS2017")
    print("="*70)
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")
    print(f"Model type: {model_type}")
    
    # Step 1: Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, features = \
        preprocess_cicids_data(train_path, test_path)
    
    input_dim = X_train.shape[1]
    
    # Step 2: Train model
    model, history = train_tcn(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        model_type=model_type,
        epochs=epochs,
        batch_size=128
    )
    
    # Step 3: Plot training history
    plot_training_history(history, model_type)
    
    # Step 4: Evaluate on test set
    results = evaluate_model(model, X_test, y_test, model_name=f"{model_type.upper()} TCN")
    
    # Step 5: Plot evaluation results
    plot_evaluation_results(results, f"{model_type.upper()}_TCN")
    
    # Step 6: Model summary
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    model.summary()
    
    # Step 7: Save final summary
    summary_path = LOGS_DIR / f"training_summary_{TIMESTAMP}.txt"
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Timestamp: {TIMESTAMP}\n")
        f.write(f"Input Dimension: {input_dim}\n")
        f.write(f"Training Samples: {len(X_train):,}\n")
        f.write(f"Validation Samples: {len(X_val):,}\n")
        f.write(f"Test Samples: {len(X_test):,}\n")
        f.write("\nTest Performance:\n")
        f.write(f"  Accuracy: {results['accuracy']*100:.4f}%\n")
        f.write(f"  Precision: {results['precision']*100:.4f}%\n")
        f.write(f"  Recall: {results['recall']*100:.4f}%\n")
        f.write(f"  F1-Score: {results['f1']*100:.4f}%\n")
        f.write(f"  AUC-ROC: {results['auc']*100:.4f}%\n")
        f.write(f"  False Positive Rate: {results['fpr']*100:.4f}%\n")
        f.write(f"  Detection Rate: {results['detection_rate']*100:.4f}%\n")
    
    print(f"\n💾 Training summary saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("✅ TRAINING AND EVALUATION COMPLETE!")
    print("="*70)
    print(f"📁 All outputs saved to: {TCN_DIR}")
    print(f"   - Models: {MODELS_DIR}")
    print(f"   - Plots: {PLOTS_DIR}")
    print(f"   - Logs: {LOGS_DIR}")
    print(f"   - Scaler: {SCALER_PATH}")
    print("="*70 + "\n")
    
    return model, history, results


if __name__ == "__main__":

    TRAIN_PATH = DATASET_DIR / "cicids_train.csv"  
    TEST_PATH = DATASET_DIR / "cicids_test.csv"    
    
    # Check if paths exist
    if not os.path.exists(TRAIN_PATH):
        print(f"⚠️  ERROR: Train file not found at {TRAIN_PATH}")
        print("Please update TRAIN_PATH in the script to point to your cicids_train.csv file")
        exit(1)
    
    if not os.path.exists(TEST_PATH):
        print(f"⚠️  ERROR: Test file not found at {TEST_PATH}")
        print("Please update TEST_PATH in the script to point to your cicids_test.csv file")
        exit(1)
    
    print("\n🎯 Training IMPROVED TCN model...")
    model_improved, history_improved, results_improved = main(
        TRAIN_PATH, TEST_PATH,
        model_type='improved',
        epochs=50
    )