# src/train.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from src.prepare_features import prepare_from_paths
from src.utils import save_scaler, save_ae
from pathlib import Path
import sys
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

# Project directories (resolve relative to this file so running from any CWD works)
BASE_DIR = Path(__file__).resolve().parents[1]   # <project-root>/anomly_detector
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"           # AE, scaler, features, threshold saved here
MODELS_LOF_DIR = BASE_DIR / "models_lof"   # LOF model saved here (separate folder)

# LOF fitting settings (tuneable)
LOF_NEIGHBORS = 50        # n_neighbors param for LOF
LOF_CONTAMINATION = 0.02  # expected fraction of outliers
LOF_SAMPLE_FOR_FIT = True  # If True and dataset large, sample for LOF training
LOF_SAMPLE_SIZE = 100000   # max rows to use for LOF.fit (if sampling enabled)

def build_sparse_autoencoder(input_dim, encoding_dim1=64, encoding_dim2=32, l1=1e-5):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(encoding_dim1, activation='relu', activity_regularizer=regularizers.l1(l1))(inp)
    x = layers.Dense(encoding_dim2, activation='relu')(x)
    x = layers.Dense(encoding_dim1, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    ae = models.Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    return ae

def fusion_score_array(recon_err_arr, lof_pred_arr, ae_threshold, w_ae=0.7, w_if=0.3,
                       use_lof_continuous=False, lof_decision_scores=None):
    """
    Returns fusion score in [0,1] where:
      - recon contribution = recon_err / ae_threshold (clipped 0..1)
      - lof contribution = binary (0/1) or normalized continuous lof_decision_scores (higher -> more anomalous)
    """
    recon_norm = np.array(recon_err_arr, dtype=float) / (ae_threshold + 1e-12)
    recon_norm = np.clip(recon_norm, 0.0, 1.0)

    if use_lof_continuous and (lof_decision_scores is not None):
        s = np.asarray(lof_decision_scores, dtype=float)
        # LocalOutlierFactor.decision_function -> higher = more normal
        # invert so higher => more anomalous
        s_inv = -s
        denom = float(np.ptp(s_inv)) if np.ptp(s_inv) is not None else 0.0
        if denom > 0.0:
            lof_norm = (s_inv - s_inv.min()) / denom
        else:
            lof_norm = np.zeros_like(s_inv, dtype=float)
    else:
        # binary lof_pred_arr: -1 -> anomaly, 1 -> normal
        lof_norm = np.where(np.asarray(lof_pred_arr) == -1, 1.0, 0.0).astype(float)

    score = (w_ae * recon_norm) + (w_if * lof_norm)
    return np.clip(score, 0.0, 1.0)

def _make_sample_inference_csv(features, scaler, X_full, out_path: Path, n_rows: int = 10):
    """
    Create sample_inference.csv with n_rows using scaler ranges or fallback medians.
    features: list of feature names (ordered)
    scaler: fitted MinMaxScaler or None
    X_full: the original full X DataFrame used in training (pandas.DataFrame)
    out_path: Path to write CSV
    """
    # Attempt to sample realistic ranges from scaler if possible
    if scaler is not None and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        try:
            mins = np.array(scaler.data_min_, dtype=float)
            maxs = np.array(scaler.data_max_, dtype=float)
            ranges = np.maximum(maxs - mins, 1e-6)
            # sample uniformly in the original scale
            samples = np.random.random(size=(n_rows, len(features))) * ranges + mins
            df_sample = pd.DataFrame(samples, columns=features)
            df_sample = df_sample.astype(float)
            df_sample.to_csv(out_path, index=False)
            print(f"[train] Created sample inference CSV using scaler ranges -> {out_path}")
            return
        except Exception as e:
            print("[train] Unable to use scaler ranges for sample CSV:", e)

    # Fallback: use medians from X_full if available
    try:
        medians = X_full[features].median().fillna(0.0).values
        samples = np.tile(medians.reshape(1, -1), (n_rows, 1)).astype(float)
        df_sample = pd.DataFrame(samples, columns=features)
        df_sample.to_csv(out_path, index=False)
        print(f"[train] Created sample inference CSV using training medians -> {out_path}")
        return
    except Exception as e:
        print("[train] Fallback median generation failed:", e)

    # Last fallback: zeros
    df_sample = pd.DataFrame(np.zeros((n_rows, len(features))), columns=features)
    df_sample.to_csv(out_path, index=False)
    print(f"[train] Created zero-filled sample inference CSV -> {out_path}")

def evaluate_models_on_labeled(X_scaled, recon_err, lof, lof_scores, y_true, ae_threshold, weights=(0.7,0.3)):
    """
    Evaluate AE (threshold), LOF (binary), and Fusion (weighted).
    Returns a dict of metrics and prints summary.
    """
    # AE binary prediction using threshold (1 -> anomaly)
    y_pred_ae = (recon_err > ae_threshold).astype(int)

    # LOF binary prediction: lof.predict -> -1 anomaly, 1 normal
    lof_pred = lof.predict(X_scaled)
    y_pred_lof = np.where(lof_pred == -1, 1, 0).astype(int)

    # Fusion (use LOF decision_function as continuous signal)
    lof_dec_scores = lof_scores  # decision_function values (higher -> more normal)
    fusion_scores = fusion_score_array(recon_err, lof_pred, ae_threshold, w_ae=weights[0], w_if=weights[1], use_lof_continuous=True, lof_decision_scores=lof_dec_scores)
    y_pred_fusion = (fusion_scores >= 0.5).astype(int)

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
    compute_binary_metrics(y_true, y_pred_lof, "LocalOutlierFactor (LOF)")
    compute_binary_metrics(y_true, y_pred_fusion, "Fusion Gate (AE+LOF)")

    # AUC for continuous signals (if both classes present)
    try:
        if len(np.unique(y_true)) > 1:
            auc_ae = roc_auc_score(y_true, recon_err)
            lof_cont = -np.array(lof_scores)
            auc_lof = roc_auc_score(y_true, lof_cont)
            auc_fusion = roc_auc_score(y_true, fusion_scores)
            print(f"\nAUCs: AE={auc_ae:.4f}, LOF={auc_lof:.4f}, Fusion={auc_fusion:.4f}")
            results['auc'] = {'ae': float(auc_ae), 'lof': float(auc_lof), 'fusion': float(auc_fusion)}
    except Exception as e:
        print("AUC calculation failed:", e)

    return results

def train(paths, model_name_prefix="fusion", epochs=20, batch_size=256, create_sample_csv=True, sample_rows=10, debug_eval=True,
          lof_n_neighbors=LOF_NEIGHBORS, lof_contamination=LOF_CONTAMINATION, lof_sample_for_fit=LOF_SAMPLE_FOR_FIT, lof_sample_size=LOF_SAMPLE_SIZE):
    print("[train] Preparing data...")
    X, Y, features = prepare_from_paths(paths)
    print(f"[train] X shape: {X.shape}; labels shape: {Y.shape}")

    # Keep only rows that are labeled normal (0) for autoencoder base training if available.
    if 0 in Y.unique():
        X_normal = X[Y == 0].copy()
        print(f"[train] Using {len(X_normal)} normal rows for autoencoder.")
    else:
        X_normal = X.copy()
        print("[train] No explicit normal labels found; using all data for AE training.")

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)                # fit on whole dataset for LOF
    X_norm_scaled = scaler.transform(X_normal)        # normal-only scaled for AE

    # Ensure models dir exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MODELS_LOF_DIR, exist_ok=True)

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    print(f"[train] Saving scaler -> {scaler_path}")
    save_scaler(scaler, scaler_path)

    # Autoencoder
    ae = build_sparse_autoencoder(input_dim=X.shape[1])
    print("[train] Training autoencoder...")
    ae.fit(X_norm_scaled, X_norm_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)

    # Save AE as .keras single-file format
    ae_filepath = os.path.join(MODELS_DIR, f"{model_name_prefix}_ae.keras")
    print(f"[train] Saving autoencoder -> {ae_filepath}")
    ae.save(ae_filepath, include_optimizer=False)

    # Create reconstruction error for whole dataset to help LOF setup or thresholding
    reconstructed = ae.predict(X_scaled)
    recon_err = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

    # ---------- LOCAL OUTLIER FACTOR (LOF) ----------
    print("[train] Preparing to train LocalOutlierFactor (LOF)...")
    n_rows = X_scaled.shape[0]
    lof_fit_X = X_scaled

    if lof_sample_for_fit and n_rows > lof_sample_size:
        # sample without replacement
        print(f"[train] Dataset large ({n_rows} rows). Sampling {lof_sample_size} rows for LOF.fit() (set lof_sample_for_fit=False to disable).")
        rng = np.random.default_rng(42)
        idx = rng.choice(n_rows, size=lof_sample_size, replace=False)
        lof_fit_X = X_scaled[idx]

    print(f"[train] Fitting LOF with n_neighbors={lof_n_neighbors}, contamination={lof_contamination} on data shape {lof_fit_X.shape} ...")
    lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, contamination=lof_contamination, novelty=True)
    lof.fit(lof_fit_X)   # fit on sampled/whole data depending on settings

    # Save LOF model (joblib) to separate folder
    lof_path = os.path.join(MODELS_LOF_DIR, f"{model_name_prefix}_lof.pkl")
    print(f"[train] Saving LOF model -> {lof_path}")
    joblib.dump(lof, lof_path)

    # Use LOF to predict / compute decision_function on the full dataset
    lof_pred = lof.predict(X_scaled)               # -1 anomaly, 1 normal
    lof_decision = lof.decision_function(X_scaled)  # higher -> more normal

    # Save features list too
    features_path = os.path.join(MODELS_DIR, f"{model_name_prefix}_features.pkl")
    joblib.dump(features, features_path)
    print(f"[train] Saved features -> {features_path}")

    # Optionally compute threshold for reconstruction error (e.g., 95th percentile over normals)
    try:
        ae_threshold = np.percentile(recon_err[Y==0] if 0 in Y.unique() else recon_err, 95)
    except Exception:
        ae_threshold = float(np.percentile(recon_err, 95))
    threshold_path = os.path.join(MODELS_DIR, f"{model_name_prefix}_ae_threshold.pkl")
    joblib.dump(ae_threshold, threshold_path)
    print(f"[train] AE threshold saved -> {threshold_path} (value={ae_threshold})")

    # Create sample_inference.csv so inference can be tested immediately
    if create_sample_csv:
        sample_out = DATA_DIR / "sample_inference.csv"
        try:
            _make_sample_inference_csv(features, scaler, X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=features), sample_out, n_rows=sample_rows)
        except Exception as e:
            print("[train] Failed to create sample_inference.csv:", e)

    print(f"[train] Saved models to {MODELS_DIR} and LOF to {MODELS_LOF_DIR}")

    # ---------- EVALUATION on labeled data ----------
    if debug_eval:
        print("\n[train] Evaluating models on labeled data (if available)...")
        labeled_mask = (Y != -1)
        if labeled_mask.sum() == 0:
            print("[train] No labeled rows found; skipping evaluation.")
        else:
            X_labeled = np.array(X)[labeled_mask]
            y_labeled = np.array(Y)[labeled_mask].astype(int)
            X_labeled_scaled = scaler.transform(X_labeled)

            reconstructed_l = ae.predict(X_labeled_scaled)
            recon_err_l = np.mean(np.power(X_labeled_scaled - reconstructed_l, 2), axis=1)

            lof_pred_l = lof.predict(X_labeled_scaled)
            lof_decision_l = lof.decision_function(X_labeled_scaled)

            metrics = evaluate_models_on_labeled(X_labeled_scaled, recon_err_l, lof, lof_decision_l, y_labeled, ae_threshold, weights=(0.7,0.3))
            print("\n[train] Evaluation summary:", metrics)

    return {
        'ae': ae,
        'lof': lof,
        'scaler': scaler,
        'features': features,
        'threshold': ae_threshold,
        'lof_path': str(lof_path)
    }

if __name__ == "__main__":
    # filenames you said are present under data
    files = [
        "cicids_combined.csv"
    ]

    # build absolute paths from DATA_DIR
    paths = [str(DATA_DIR / f) for f in files]

    print("Resolved paths:")
    for p in paths:
        print("  ", p)

    # quick existence check
    missing = [p for p in paths if not Path(p).exists()]
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

    # If all files exist, call train
    train(paths, epochs=20, batch_size=256, create_sample_csv=True, sample_rows=10, debug_eval=True)

