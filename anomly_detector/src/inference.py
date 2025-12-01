import os 
import numpy as np
import pandas as pd
import joblib
from src.utils import load_ae,load_isof,load_scaler
from typing import Tuple

MODELS_DIR = "models"
PREFIX = "fusion"

def load_all(prefix=PREFIX):
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    ae = load_ae(os.path.join(MODELS_DIR, f"{prefix}_ae"))
    iso = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_isof.pkl"))
    features = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_features.pkl"))
    threshold = joblib.load(os.path.join(MODELS_DIR, f"{prefix}_ae_threshold.pkl"))
    return scaler, ae, iso, features, threshold

def fusion_score(recon_err_arr, iso_pred_arr, w_ae=0.7, w_iso=0.3):
    # recon_err_arr: array of reconstruction errors (higher -> more anomalous)
    # iso_pred_arr: -1 for anomaly, 1 for normal
    iso_score = np.where(iso_pred_arr == -1, 1.0, 0.0)
    # Normalize recon_err to 0-1 (simple scaling by max)
    if recon_err_arr.max() > 0:
        norm_re = recon_err_arr / recon_err_arr.max()
    else:
        norm_re = recon_err_arr
    return w_ae*norm_re + w_iso*iso_score

def predict_from_df(df: pd.DataFrame):
    scaler, ae, iso, features, threshold = load_all()
    X = df[features].astype(float).values
    Xs = scaler.transform(X)
    recon = ae.predict(Xs)
    recon_err = np.mean(np.power(Xs - recon, 2), axis=1)
    iso_pred = iso.predict(Xs)  # -1 anomaly, 1 normal
    scores = fusion_score(recon_err, iso_pred)
    # choose final label by threshold: if score > 0.5 -> anomaly
    labels = (scores > 0.5).astype(int)
    out = df.copy()
    out['recon_err'] = recon_err
    out['iso_pred'] = iso_pred
    out['fusion_score'] = scores
    out['pred_label'] = labels
    return out

if __name__ == "__main__":
    
