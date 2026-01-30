# src/train.py
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, regularizers
from anomly_detector.src.network_level_attacks_anomality.prepare_features import prepare_from_paths
from anomly_detector.src.network_level_attacks_anomality.utils import save_scaler, save_ae, ensure_dir
from pathlib import Path
import sys
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets"
MODELS_DIR = PROJECT_ROOT / "models" / "anomaly_detector" / "network_level_anomality"


def fit_pca_mahalanobis(X_train_norm_scaled, n_components=0.95):
    """
    Fit PCA on normal data and compute mean + inverse covariance
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_train_norm_scaled)

    mean_vec = np.mean(X_pca, axis=0)
    cov = np.cov(X_pca, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6  # numerical stability
    inv_cov = np.linalg.inv(cov)

    return pca, mean_vec, inv_cov


def mahalanobis_scores(X_scaled, pca, mean_vec, inv_cov):
    X_pca = pca.transform(X_scaled)
    diffs = X_pca - mean_vec
    scores = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)
    return scores


def build_sparse_autoencoder(input_dim, encoding_dim1=64, encoding_dim2=32, l1=1e-5):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(encoding_dim1, activation='relu', activity_regularizer=regularizers.l1(l1))(inp)
    x = layers.Dense(encoding_dim2, activation='relu')(x)
    x = layers.Dense(encoding_dim1, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    ae = models.Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    return ae

def fusion_score_array(recon_err_arr, maha_scores, ae_threshold,
                       w_ae=0.7, w_stat=0.3):
    recon_norm = recon_err_arr / (ae_threshold + 1e-12)
    recon_norm = np.clip(recon_norm, 0.0, 1.0)

    # Normalize Mahalanobis scores
    maha_norm = (maha_scores - maha_scores.min()) / (np.ptp(maha_scores) + 1e-12)
    maha_norm = np.clip(maha_norm, 0.0, 1.0)

    score = (w_ae * recon_norm) + (w_stat * maha_norm)
    return np.clip(score, 0.0, 1.0)


def _make_sample_inference_csv(features, scaler, X_full, out_path: Path, n_rows: int = 10):
    if scaler is not None and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        try:
            mins = np.array(scaler.data_min_, dtype=float)
            maxs = np.array(scaler.data_max_, dtype=float)
            ranges = np.maximum(maxs - mins, 1e-6)
            samples = np.random.random(size=(n_rows, len(features))) * ranges + mins
            df_sample = pd.DataFrame(samples, columns=features).astype(float)
            df_sample.to_csv(out_path, index=False)
            print(f"[train] Created sample inference CSV using scaler ranges -> {out_path}")
            return
        except Exception as e:
            print("[train] Unable to use scaler ranges for sample CSV:", e)

    try:
        medians = X_full[features].median().fillna(0.0).values
        samples = np.tile(medians.reshape(1, -1), (n_rows, 1)).astype(float)
        df_sample = pd.DataFrame(samples, columns=features)
        df_sample.to_csv(out_path, index=False)
        print(f"[train] Created sample inference CSV using training medians -> {out_path}")
        return
    except Exception as e:
        print("[train] Fallback median generation failed:", e)

    df_sample = pd.DataFrame(np.zeros((n_rows, len(features))), columns=features)
    df_sample.to_csv(out_path, index=False)
    print(f"[train] Created zero-filled sample inference CSV -> {out_path}")

def evaluate_models_on_labeled(X_scaled, recon_err, maha_scores, y_true, ae_threshold, weights=(0.7,0.3)):
    y_pred_ae = (recon_err > ae_threshold).astype(int)
    if np.any(y_true == 0):
        maha_threshold = np.percentile(maha_scores[y_true == 0], 95)
    else:
        maha_threshold = np.percentile(maha_scores, 95)
    y_pred_stat = (maha_scores > maha_threshold).astype(int)

    fusion_scores = fusion_score_array(recon_err_arr = recon_err,
                                       maha_scores = maha_scores,
                                       ae_threshold= ae_threshold,
                                       w_ae = weights[0],
                                       w_stat = weights[1]
                                       )
    y_pred_fusion = (fusion_scores > 0.5).astype(int)

    results = {}


    def compute_binary_metrics(y_true, y_pred, name):
        acc = accuracy_score(y_true, y_pred)
        prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        print(f"\n=== {name} METRICS ===")
        print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
        print("Confusion matrix:\n", cm)

    compute_binary_metrics(y_true, y_pred_ae, "Autoencoder (AE)")
    compute_binary_metrics(y_true, y_pred_stat, "PCA + Mahalanobis")
    compute_binary_metrics(y_true, y_pred_fusion, "Fusion Gate (AE+Stat)")

    try:
        if len(np.unique(y_true)) > 1:
            auc_ae = roc_auc_score(y_true, recon_err)
            auc_stat = roc_auc_score(y_true, maha_scores)
            auc_fusion = roc_auc_score(y_true, fusion_scores)
            print(f"\nAUCs: AE={auc_ae:.4f}, PCA={auc_stat:.4f}, Fusion={auc_fusion:.4f}")
            results['auc'] = {'ae': float(auc_ae), 'PCA/Maha': float(auc_stat), 'fusion': float(auc_fusion)}
    except Exception as e:
        print("AUC calculation failed:", e)

    return results

def train_with_train_and_test_files(train_paths, test_paths, model_name_prefix="fusion",
                                    epochs=20, batch_size=256, create_sample_csv=True, sample_rows=10, debug_eval=True):
    """
    Train using data from train_paths (list of file paths) and evaluate on test_paths.
    """
    print("[train] Preparing training data...")
    X_train_df, Y_train_ser, features_train = prepare_from_paths(train_paths)
    print(f"[train] Training X shape: {X_train_df.shape}; labels shape: {Y_train_ser.shape}")

    print("[train] Preparing test data...")
    X_test_df, Y_test_ser, features_test = prepare_from_paths(test_paths)
    print(f"[train] Test X shape: {X_test_df.shape}; labels shape: {Y_test_ser.shape}")

    # Align features if needed
    if features_train != features_test:
        print("[train] WARNING: feature lists differ between train and test. Aligning test to training features (missing -> 0).")
        X_test_df = X_test_df.reindex(columns=features_train).fillna(0.0)
        features = features_train
    else:
        features = features_train

    ensure_dir(MODELS_DIR)

    # Convert to DataFrame and ensure numeric
    if not isinstance(X_train_df, pd.DataFrame):
        X_train_df = pd.DataFrame(X_train_df, columns=features)
    if not isinstance(X_test_df, pd.DataFrame):
        X_test_df = pd.DataFrame(X_test_df, columns=features)

    # Basic cleaning: numeric coercion, replace inf, fill na with medians (computed from train)
    X_train_df = X_train_df.apply(pd.to_numeric, errors='coerce')
    X_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_medians = X_train_df.median().fillna(0.0)
    X_train_df = X_train_df.fillna(train_medians)

    X_test_df = X_test_df.apply(pd.to_numeric, errors='coerce')
    X_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # fill NAs in test using training medians to avoid leakage from test
    X_test_df = X_test_df.fillna(train_medians)

    # Clip extreme values (safeguard)
    CLIP_MAG = 1e12
    X_train_df = X_train_df.clip(lower=-CLIP_MAG, upper=CLIP_MAG)
    X_test_df = X_test_df.clip(lower=-CLIP_MAG, upper=CLIP_MAG)

    # Fit scaler on training data only
    scaler = MinMaxScaler()
    X_train_vals = X_train_df.values.astype(np.float64, copy=False)
    X_test_vals = X_test_df.values.astype(np.float64, copy=False)

    X_train_scaled = scaler.fit_transform(X_train_vals)
    X_test_scaled = scaler.transform(X_test_vals)

    ensure_dir(MODELS_DIR)
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    print(f"[train] Saving scaler -> {scaler_path}")
    save_scaler(scaler, scaler_path)

    # Prepare AE training data: normals from training set only
    if 0 in Y_train_ser.unique():
        X_train_norm_df = X_train_df[np.array(Y_train_ser) == 0].copy()
        print(f"[train] AE training on {len(X_train_norm_df)} normal rows from training set.")
    else:
        X_train_norm_df = X_train_df.copy()
        print("[train] No explicit normal labels in training set; AE trained on all training rows.")

    X_train_norm_vals = X_train_norm_df.values.astype(np.float64, copy=False)
    X_train_norm_scaled = scaler.transform(X_train_norm_vals)

    # Build & train AE
    ae = build_sparse_autoencoder(input_dim=len(features))
    print("[train] Training autoencoder (on training normals)...")
    ae.fit(X_train_norm_scaled, X_train_norm_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)

    # Save AE
    ae_filepath = os.path.join(MODELS_DIR, f"{model_name_prefix}_ae.keras")
    print(f"[train] Saving autoencoder -> {ae_filepath}")
    ae.save(ae_filepath, include_optimizer=False)
    save_ae(ae, ae_filepath)  # in case save_ae does extra actions in utils

    # recon errors on train and test
    reconstructed_train = ae.predict(X_train_scaled)
    recon_err_train = np.mean(np.power(X_train_scaled - reconstructed_train, 2), axis=1)

    reconstructed_test = ae.predict(X_test_scaled)
    recon_err_test = np.mean(np.power(X_test_scaled - reconstructed_test, 2), axis=1)

    # Train PCA + Mahalanobis on NORMAL training data only
    print("[train] Training PCA + Mahalanobis (on training normals)...")
    pca, maha_mean, maha_inv_cov = fit_pca_mahalanobis(X_train_norm_scaled)

    # Save statistical model
    joblib.dump(pca, MODELS_DIR / f"{model_name_prefix}_pca.pkl")
    joblib.dump(maha_mean, MODELS_DIR / f"{model_name_prefix}_maha_mean.pkl")
    joblib.dump(maha_inv_cov, MODELS_DIR / f"{model_name_prefix}_maha_inv_cov.pkl")

    print("[train] Saved PCA + Mahalanobis models")


    # Save features
    features_path = os.path.join(MODELS_DIR, f"{model_name_prefix}_features.pkl")
    joblib.dump(features, features_path)
    print(f"[train] Saved features -> {features_path}")

    # AE threshold from training normals only (95th percentile)
    try:
        if 0 in Y_train_ser.unique():
            y_train_arr = np.array(Y_train_ser)
            recon_err_train_normals = recon_err_train[y_train_arr == 0]
            ae_threshold = float(np.percentile(recon_err_train_normals, 95))
        else:
            ae_threshold = float(np.percentile(recon_err_train, 95))
    except Exception as e:
        print("[train] AE threshold computation failed, using overall 95th percentile on training:", e)
        ae_threshold = float(np.percentile(recon_err_train, 95))

    threshold_path = os.path.join(MODELS_DIR, f"{model_name_prefix}_ae_threshold.pkl")
    joblib.dump(ae_threshold, threshold_path)
    print(f"[train] AE threshold saved -> {threshold_path} (value={ae_threshold})")

    # Create sample_inference.csv using training scaler/features
    if create_sample_csv:
        sample_out = DATA_DIR / "sample_inference.csv"
        try:
            _make_sample_inference_csv(features, scaler, X_train_df if isinstance(X_train_df, pd.DataFrame) else pd.DataFrame(X_train_vals, columns=features), sample_out, n_rows=sample_rows)
        except Exception as e:
            print("[train] Failed to create sample_inference.csv:", e)

    print(f"[train] Saved models to {MODELS_DIR}")

    # Evaluation on combined test set
    if debug_eval:
        print("\n[train] Evaluating models on combined test set (labeled rows only)...")
        labeled_mask_test = (Y_test_ser != -1)
        if labeled_mask_test.sum() == 0:
            print("[train] No labeled rows found in test set; skipping labeled evaluation.")
        else:
            X_test_labeled_scaled = X_test_scaled[labeled_mask_test]
            y_test_labeled = np.array(Y_test_ser)[labeled_mask_test].astype(int)

            recon_err_test_labeled = recon_err_test[labeled_mask_test]

            maha_scores_train = mahalanobis_scores(X_train_scaled, pca, maha_mean, maha_inv_cov)
            maha_scores_test = mahalanobis_scores(X_test_scaled, pca, maha_mean, maha_inv_cov)

            maha_scores_test_labeled = maha_scores_test[labeled_mask_test]

            metrics = evaluate_models_on_labeled(
                X_test_labeled_scaled,
                recon_err_test_labeled,
                maha_scores_test_labeled,
                y_test_labeled,
                ae_threshold,
                weights=(0.7, 0.3)
            )

            print("\n[train] Evaluation summary on test set:", metrics)

    return {
        'ae': ae,
        'pca': pca,
        'maha_mean': maha_mean,
        'maha_inv_cov': maha_inv_cov,
        'scaler': scaler,
        'features': features,
        'ae_threshold': ae_threshold
    }


# Keep original single-file train() for backward compatibility if needed
def train(paths, model_name_prefix="fusion", epochs=20, batch_size=256, create_sample_csv=True, sample_rows=10, debug_eval=True):
    raise RuntimeError("Use train_with_train_and_test_files() to train on train files and evaluate on test files.")

if __name__ == "__main__":
    # Example: change the filenames to match the CSVs you have in data/
    train_files = [
        str(DATA_DIR / "cicids_train.csv"),
    ]
    test_files = [
        str(DATA_DIR / "cicids_test.csv"),
    ]

    all_paths = train_files + test_files
    print("Resolved paths:")
    for p in all_paths:
        print("  ", p)

    missing = [p for p in all_paths if not Path(p).exists()]
    if missing:
        print("\nERROR: The following required file(s) were NOT found:")
        for m in missing:
            print("  -", m)
        print("\nFiles actually found in", DATA_DIR, ":")
        try:
            for found in sorted([str(x.name) for x in DATA_DIR.iterdir()]):
                print("  ", found)
        except Exception as e:
            print("  (couldn't list directory contents:", e, ")")
        sys.exit(2)

    train_with_train_and_test_files(train_files, test_files, epochs=20, batch_size=256, create_sample_csv=True, sample_rows=10, debug_eval=True)
