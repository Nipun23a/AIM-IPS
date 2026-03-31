"""
evaluate_cicids2018.py
──────────────────────
Evaluate both the LightGBM network classifier and the Ensemble anomaly
detector against CIC-IDS-2018 (Intrusion Detection Evaluation Dataset).

Usage
─────
    # from project root (activate venv first)
    python -m anomly_detector.src.network_level_attacks_anomality.evaluate_cicids2018 \
        --data data_collector/data_sets/cicids2018/

    # evaluate single CSV
    python -m anomly_detector.src.network_level_attacks_anomality.evaluate_cicids2018 \
        --data data_collector/data_sets/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv

    # limit rows for a quick smoke-test
    python -m anomly_detector.src.network_level_attacks_anomality.evaluate_cicids2018 \
        --data data_collector/data_sets/cicids2018/ --max_rows 200000

Results are saved to:
    models/anomly_detector/network_level_attacks_anomality/cicids2018_eval/
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = (
    PROJECT_ROOT
    / "models"
    / "anomly_detector"
    / "network_level_attacks_anomality"
    / "cicids2018_eval"
)

# ── CIC-IDS-2018 column → unified name mapping ────────────────────────────────
# CIC-IDS-2018 uses slightly different column names than 2017.
# Both variants (with and without spaces/capitalisation) are covered.

CICIDS2018_COL_MAP = {
    # Duration & Volume
    "Flow Duration":               "flow_duration",
    "Tot Fwd Pkts":                "total_fwd_packets",
    "Tot Bwd Pkts":                "total_bwd_packets",
    "TotLen Fwd Pkts":             "total_fwd_bytes",
    "TotLen Bwd Pkts":             "total_bwd_bytes",
    # 2017-style aliases (same dataset, different CICFlowMeter version)
    "Total Fwd Packets":           "total_fwd_packets",
    "Total Backward Packets":      "total_bwd_packets",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    # Rates
    "Flow Byts/s":                 "flow_bytes_per_sec",
    "Flow Pkts/s":                 "flow_pkts_per_sec",
    "Flow Bytes/s":                "flow_bytes_per_sec",
    "Flow Packets/s":              "flow_pkts_per_sec",
    "Fwd Pkts/s":                  "fwd_pkts_per_sec",
    "Bwd Pkts/s":                  "bwd_pkts_per_sec",
    "Fwd Packets/s":               "fwd_pkts_per_sec",
    "Bwd Packets/s":               "bwd_pkts_per_sec",
    # Packet length
    "Pkt Len Mean":                "pkt_len_mean",
    "Pkt Len Std":                 "pkt_len_std",
    "Pkt Len Max":                 "pkt_len_max",
    "Pkt Len Min":                 "pkt_len_min",
    "Packet Length Mean":          "pkt_len_mean",
    "Packet Length Std":           "pkt_len_std",
    "Max Packet Length":           "pkt_len_max",
    "Min Packet Length":           "pkt_len_min",
    # Per-direction packet length
    "Fwd Pkt Len Mean":            "fwd_pkt_len_mean",
    "Bwd Pkt Len Mean":            "bwd_pkt_len_mean",
    "Fwd Pkt Len Std":             "fwd_pkt_len_std",
    "Bwd Pkt Len Std":             "bwd_pkt_len_std",
    "Fwd Packet Length Mean":      "fwd_pkt_len_mean",
    "Bwd Packet Length Mean":      "bwd_pkt_len_mean",
    "Fwd Packet Length Std":       "fwd_pkt_len_std",
    "Bwd Packet Length Std":       "bwd_pkt_len_std",
    # IAT
    "Flow IAT Mean":               "flow_iat_mean",
    "Flow IAT Std":                "flow_iat_std",
    "Flow IAT Max":                "flow_iat_max",
    "Fwd IAT Mean":                "fwd_iat_mean",
    "Bwd IAT Mean":                "bwd_iat_mean",
    # TCP Flags
    "FIN Flag Cnt":                "fin_flag_count",
    "SYN Flag Cnt":                "syn_flag_count",
    "RST Flag Cnt":                "rst_flag_count",
    "PSH Flag Cnt":                "psh_flag_count",
    "ACK Flag Cnt":                "ack_flag_count",
    "FIN Flag Count":              "fin_flag_count",
    "SYN Flag Count":              "syn_flag_count",
    "RST Flag Count":              "rst_flag_count",
    "PSH Flag Count":              "psh_flag_count",
    "ACK Flag Count":              "ack_flag_count",
    # Window sizes
    "Init Fwd Win Byts":           "init_win_fwd",
    "Init Bwd Win Byts":           "init_win_bwd",
    "Init_Win_bytes_forward":      "init_win_fwd",
    "Init_Win_bytes_backward":     "init_win_bwd",
    # Ratio
    "Down/Up Ratio":               "down_up_ratio",
    # Idle (used by ensemble THREAT_FEATURES)
    "Idle Mean":                   "idle mean",
    "Idle Std":                    "idle std",
}

# ── CIC-IDS-2018 label → unified label ────────────────────────────────────────
# Binary mapping for ensemble (benign=0 / attack=1)
# Multiclass mapping for LightGBM

CICIDS2018_LABEL_BINARY = {
    "benign":                    0,
    "Benign":                    0,
    "BENIGN":                    0,
}
# Everything not in the above dict is attack (1)

CICIDS2018_LABEL_MULTICLASS = {
    # Benign
    "benign":                         "benign",
    "Benign":                         "benign",
    "BENIGN":                         "benign",
    # DDoS
    "DDOS attack-HOIC":               "ddos",
    "DDOS attack-LOIC-UDP":           "ddos",
    "DDoS attacks-LOIC-HTTP":         "ddos",
    "DDoS-LOIC-UDP":                  "ddos",
    "DDoS-HOIC":                      "ddos",
    # DoS
    "DoS attacks-Hulk":               "dos",
    "DoS attacks-SlowHTTPTest":       "dos",
    "DoS attacks-Slowloris":          "dos",
    "DoS attacks-GoldenEye":          "dos",
    "DoS-GoldenEye":                  "dos",
    "DoS-Hulk":                       "dos",
    "DoS-Slowloris":                  "dos",
    "DoS-SlowHTTPTest":               "dos",
    # Brute Force / Web
    "FTP-BruteForce":                 "bruteforce",
    "SSH-Bruteforce":                 "bruteforce",
    "Brute Force -Web":               "bruteforce",
    "Brute Force -XSS":               "bruteforce",
    "XSS":                            "bruteforce",
    # Injection
    "SQL Injection":                  "sqli",
    # Infiltration / Botnet
    "Infilteration":                  "infiltration",
    "Infiltration":                   "infiltration",
    "Bot":                            "botnet",
}

# LightGBM only knows these classes from CICIDS2017 training
LGBM_KNOWN_CLASSES = {"benign", "ddos", "dos", "portscan", "botnet"}


# ── UNIFIED_FEATURES (LightGBM — 32 features) ─────────────────────────────────
UNIFIED_FEATURES = [
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes", "total_bwd_bytes",
    "flow_bytes_per_sec", "flow_pkts_per_sec",
    "fwd_pkts_per_sec", "bwd_pkts_per_sec",
    "pkt_len_mean", "pkt_len_std", "pkt_len_max", "pkt_len_min",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max",
    "fwd_iat_mean", "bwd_iat_mean",
    "fin_flag_count", "syn_flag_count", "rst_flag_count",
    "psh_flag_count", "ack_flag_count",
    "init_win_fwd", "init_win_bwd",
    "fwd_ttl_mean", "bwd_ttl_mean",
    "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "fwd_pkt_len_std", "bwd_pkt_len_std",
    "down_up_ratio",
]

# ── THREAT_FEATURES (Ensemble — 16 features, space-separated names) ────────────
THREAT_FEATURES = [
    "flow duration",
    "total fwd packets",
    "total backward packets",
    "total length of fwd packets",
    "total length of bwd packets",
    "fwd packet length mean",
    "bwd packet length mean",
    "flow bytes/s",
    "flow packets/s",
    "syn flag count",
    "ack flag count",
    "psh flag count",
    "packet length mean",
    "packet length std",
    "idle mean",
    "idle std",
]

# Unified → THREAT_FEATURES name mapping
UNIFIED_TO_THREAT = {
    "flow_duration":        "flow duration",
    "total_fwd_packets":    "total fwd packets",
    "total_bwd_packets":    "total backward packets",
    "total_fwd_bytes":      "total length of fwd packets",
    "total_bwd_bytes":      "total length of bwd packets",
    "fwd_pkt_len_mean":     "fwd packet length mean",
    "bwd_pkt_len_mean":     "bwd packet length mean",
    "flow_bytes_per_sec":   "flow bytes/s",
    "flow_pkts_per_sec":    "flow packets/s",
    "syn_flag_count":       "syn flag count",
    "ack_flag_count":       "ack flag count",
    "psh_flag_count":       "psh flag count",
    "pkt_len_mean":         "packet length mean",
    "pkt_len_std":          "packet length std",
    "idle mean":            "idle mean",
    "idle std":             "idle std",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_cicids2018(path: Path, max_rows: int = None) -> pd.DataFrame:
    """Load one or all CICIDS2018 CSV files from a path."""
    if path.is_file():
        csv_files = [path]
    else:
        csv_files = sorted(path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found at: {path}")

    print(f"\n{'='*60}")
    print(f"  Loading CIC-IDS-2018 — {len(csv_files)} file(s)")
    print(f"{'='*60}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df.columns = df.columns.str.strip()
            label_col = next(
                (c for c in df.columns if c.strip().lower() == "label"), None
            )
            if label_col:
                counts = df[label_col].value_counts().to_dict()
                print(f"  {f.name}: {len(df):,} rows | {counts}")
            else:
                print(f"  {f.name}: {len(df):,} rows | (no Label column)")
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Cannot load {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total combined: {len(combined):,} rows")

    if max_rows and len(combined) > max_rows:
        combined = combined.sample(n=max_rows, random_state=42)
        print(f"  Sampled to:    {len(combined):,} rows (--max_rows {max_rows})")

    return combined


def preprocess(df: pd.DataFrame):
    """
    Rename columns, impute missing unified features, clean inf/NaN.
    Returns (df_unified, y_binary, y_multiclass).
    """
    # Rename CICIDS2018 columns → unified names
    df = df.rename(columns=CICIDS2018_COL_MAP)

    # Impute TTL (not produced by CICFlowMeter)
    df["fwd_ttl_mean"] = df.get("fwd_ttl_mean", 64.0)
    df["bwd_ttl_mean"] = df.get("bwd_ttl_mean", 64.0)

    # Impute idle features if missing (not in all CICIDS2018 exports)
    for col in ("idle mean", "idle std"):
        if col not in df.columns:
            df[col] = 0.0

    # Ensure all required features exist
    for feat in UNIFIED_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    for feat in THREAT_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    # Find label column
    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"), None
    )
    if label_col is None:
        raise ValueError("Cannot find Label column — check CSV format.")

    raw_labels = df[label_col].astype(str).str.strip()

    # Binary labels (0 = benign, 1 = attack)
    y_binary = (~raw_labels.str.lower().isin(["benign"])).astype(int).values

    # Multiclass labels
    y_multi_str = raw_labels.map(CICIDS2018_LABEL_MULTICLASS).fillna("other")

    # Numeric conversion
    all_classes = sorted(y_multi_str.unique())
    class_to_int = {c: i for i, c in enumerate(all_classes)}
    y_multi = y_multi_str.map(class_to_int).values

    # Clean features
    feat_cols = list(set(UNIFIED_FEATURES + THREAT_FEATURES))
    for col in feat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in unified features
    valid_mask = df[UNIFIED_FEATURES].notna().all(axis=1)
    print(f"\n  Dropped {(~valid_mask).sum():,} rows with inf/NaN in features")
    df = df[valid_mask].reset_index(drop=True)
    y_binary  = y_binary[valid_mask.values]
    y_multi   = y_multi[valid_mask.values]
    y_multi_str = y_multi_str[valid_mask].reset_index(drop=True)

    # Clip outliers
    for col in ["flow_bytes_per_sec", "flow_pkts_per_sec",
                "fwd_pkts_per_sec", "bwd_pkts_per_sec"]:
        if col in df.columns and df[col].max() > 0:
            cap = df[col].quantile(0.999)
            df[col] = df[col].clip(upper=cap)

    df[UNIFIED_FEATURES] = df[UNIFIED_FEATURES].clip(lower=0).astype(np.float32)

    print(f"  Clean rows:    {len(df):,}")
    print(f"\n  Binary label distribution:")
    unique, counts = np.unique(y_binary, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {'Benign' if u == 0 else 'Attack'}: {c:,}")
    print(f"\n  Multiclass distribution:")
    for cls, cnt in y_multi_str.value_counts().items():
        print(f"    {cls:<20} {cnt:,}")

    return df, y_binary, y_multi, y_multi_str, all_classes


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FEATURE DICTS  (for pipeline inference wrappers)
# ─────────────────────────────────────────────────────────────────────────────

def rows_to_unified_dicts(df: pd.DataFrame) -> list:
    """
    Convert DataFrame rows to feature dicts for LightGBM.
    LGBMNetworkClassifier._vectorize() uses THREAT_FEATURES keys
    (space-separated, e.g. 'flow duration') loaded from features.pkl.
    We supply BOTH unified (underscore) and threat (space) key variants
    so features.pkl resolves correctly regardless of which was used at
    training time.
    """
    records = []
    for row in df[UNIFIED_FEATURES].itertuples(index=False):
        d = dict(zip(UNIFIED_FEATURES, row))
        # Add space-separated aliases for THREAT_FEATURES
        for unified_col, threat_col in UNIFIED_TO_THREAT.items():
            if unified_col in d:
                d[threat_col] = d[unified_col]
        records.append(d)
    return records


def rows_to_threat_dicts(df: pd.DataFrame) -> list:
    """Convert DataFrame rows to THREAT_FEATURES dicts (Ensemble input)."""
    threat_df = pd.DataFrame()
    for unified_col, threat_col in UNIFIED_TO_THREAT.items():
        if unified_col in df.columns:
            threat_df[threat_col] = df[unified_col].values
        elif threat_col in df.columns:
            threat_df[threat_col] = df[threat_col].values
        else:
            threat_df[threat_col] = 0.0
    # Any remaining THREAT_FEATURES not yet covered
    for feat in THREAT_FEATURES:
        if feat not in threat_df.columns:
            threat_df[feat] = df[feat].values if feat in df.columns else 0.0
    return threat_df[THREAT_FEATURES].to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTGBM EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_lgbm(df: pd.DataFrame, y_binary: np.ndarray, y_multi_str: pd.Series):
    print(f"\n{'='*60}")
    print("  LightGBM Network Classifier — Evaluation")
    print(f"{'='*60}")

    from pipeline.network_level.lgbm_network_classifier import LGBMNetworkClassifier
    from pipeline.network_level.feature import LGBM_MODEL_PATH, LGBM_FEATURES_PATH

    clf = LGBMNetworkClassifier(
        model_path    = PROJECT_ROOT / LGBM_MODEL_PATH,
        features_path = PROJECT_ROOT / LGBM_FEATURES_PATH,
    )

    try:
        clf.load()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None

    print(f"  Model classes: {clf.classes}")
    print(f"  Scoring {len(df):,} flows...")

    feature_dicts = rows_to_unified_dicts(df)
    pred_labels  = []
    threat_scores = []

    for i, feat_dict in enumerate(feature_dicts):
        if i % 50000 == 0 and i > 0:
            print(f"    ... {i:,} / {len(feature_dicts):,}")
        label, _, _ = clf.predict(feat_dict)
        score        = clf.threat_score(feat_dict)
        pred_labels.append(label)
        threat_scores.append(score)

    pred_labels   = np.array(pred_labels)
    threat_scores = np.array(threat_scores)

    # Binary predictions (benign=0, anything else=1)
    y_pred_binary = (pred_labels != "benign").astype(int)

    # ── Binary metrics ──────────────────────────────────────────────────────
    acc  = accuracy_score(y_binary, y_pred_binary)
    prec = precision_score(y_binary, y_pred_binary, zero_division=0)
    rec  = recall_score(y_binary, y_pred_binary, zero_division=0)
    f1   = f1_score(y_binary, y_pred_binary, zero_division=0)
    fpr  = (((y_pred_binary == 1) & (y_binary == 0)).sum() /
            max((y_binary == 0).sum(), 1))
    fnr  = (((y_pred_binary == 0) & (y_binary == 1)).sum() /
            max((y_binary == 1).sum(), 1))

    try:
        auc = roc_auc_score(y_binary, threat_scores)
    except Exception:
        auc = float("nan")

    print(f"\n  ── Binary Classification (Benign vs Attack) ──")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  FPR       : {fpr:.4f}  (false alarm rate)")
    print(f"  FNR       : {fnr:.4f}  (miss rate)")

    # Confusion matrix
    cm = confusion_matrix(y_binary, y_pred_binary)
    print(f"\n  Confusion Matrix (binary):")
    print(f"              Pred Benign  Pred Attack")
    print(f"  True Benign  {cm[0,0]:>10,}  {cm[0,1]:>10,}")
    print(f"  True Attack  {cm[1,0]:>10,}  {cm[1,1]:>10,}")

    # Multiclass report for classes the model knows
    known_mask = y_multi_str.isin(LGBM_KNOWN_CLASSES)
    if known_mask.sum() > 0:
        print(f"\n  ── Multiclass Report (known classes only: {known_mask.sum():,} rows) ──")
        print(classification_report(
            y_multi_str[known_mask],
            pred_labels[known_mask],
            zero_division=0,
        ))

    results = {
        "model": "LightGBM",
        "n_samples": len(df),
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "auc_roc":   round(auc,  4) if not np.isnan(auc) else None,
        "fpr":       round(fpr,  4),
        "fnr":       round(fnr,  4),
        "confusion_matrix": cm.tolist(),
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ensemble(df: pd.DataFrame, y_binary: np.ndarray):
    print(f"\n{'='*60}")
    print("  Ensemble Anomaly Detector — Evaluation")
    print("  (AE + VAE + OCC + IsolationForest)")
    print(f"{'='*60}")

    from pipeline.network_level.ensemble_detector import EnsembleDetector

    det = EnsembleDetector()
    try:
        det.load()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None

    print(f"  Weights: { {n: round(float(w),3) for n, w in det.model_summary().items()} }")
    print(f"  Threshold (raw): {det._raw_thr}")
    print(f"  Scoring {len(df):,} flows...")

    threat_dicts  = rows_to_threat_dicts(df)
    ens_scores    = []
    ens_anomalies = []

    for i, feat_dict in enumerate(threat_dicts):
        if i % 50000 == 0 and i > 0:
            print(f"    ... {i:,} / {len(threat_dicts):,}")
        score = det.predict(feat_dict)
        ens_scores.append(score)
        ens_anomalies.append(1 if det.is_anomaly(score) else 0)

    ens_scores    = np.array(ens_scores)
    ens_anomalies = np.array(ens_anomalies)

    # ── Binary metrics ──────────────────────────────────────────────────────
    acc  = accuracy_score(y_binary, ens_anomalies)
    prec = precision_score(y_binary, ens_anomalies, zero_division=0)
    rec  = recall_score(y_binary, ens_anomalies, zero_division=0)
    f1   = f1_score(y_binary, ens_anomalies, zero_division=0)
    fpr  = (((ens_anomalies == 1) & (y_binary == 0)).sum() /
            max((y_binary == 0).sum(), 1))
    fnr  = (((ens_anomalies == 0) & (y_binary == 1)).sum() /
            max((y_binary == 1).sum(), 1))

    try:
        auc = roc_auc_score(y_binary, ens_scores)
    except Exception:
        auc = float("nan")

    print(f"\n  ── Binary Classification (Normal vs Anomaly) ──")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  FPR       : {fpr:.4f}  (false alarm rate)")
    print(f"  FNR       : {fnr:.4f}  (miss rate)")

    # Confusion matrix
    cm = confusion_matrix(y_binary, ens_anomalies)
    print(f"\n  Confusion Matrix:")
    print(f"              Pred Normal  Pred Anomaly")
    print(f"  True Normal  {cm[0,0]:>11,}  {cm[0,1]:>11,}")
    print(f"  True Attack  {cm[1,0]:>11,}  {cm[1,1]:>11,}")

    # Score distribution
    benign_scores = ens_scores[y_binary == 0]
    attack_scores = ens_scores[y_binary == 1]
    print(f"\n  Score distribution:")
    print(f"    Benign  — mean={benign_scores.mean():.4f}  "
          f"std={benign_scores.std():.4f}  "
          f"max={benign_scores.max():.4f}")
    print(f"    Attack  — mean={attack_scores.mean():.4f}  "
          f"std={attack_scores.std():.4f}  "
          f"max={attack_scores.max():.4f}")

    results = {
        "model": "Ensemble (AE+VAE+OCC+IsoForest)",
        "n_samples": len(df),
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "auc_roc":   round(auc,  4) if not np.isnan(auc) else None,
        "fpr":       round(fpr,  4),
        "fnr":       round(fnr,  4),
        "confusion_matrix": cm.tolist(),
        "score_stats": {
            "benign_mean": round(float(benign_scores.mean()), 4),
            "attack_mean": round(float(attack_scores.mean()), 4),
        },
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(lgbm_results: dict, ens_results: dict, data_path: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"eval_{ts}.json"

    payload = {
        "timestamp":   ts,
        "dataset":     "CIC-IDS-2018",
        "data_path":   str(data_path),
        "lightgbm":    lgbm_results,
        "ensemble":    ens_results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(lgbm_results, ens_results):
    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY — CIC-IDS-2018")
    print(f"{'='*60}")
    header = f"  {'Metric':<14} {'LightGBM':>12} {'Ensemble':>12}"
    print(header)
    print(f"  {'-'*14} {'-'*12} {'-'*12}")

    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc", "fpr", "fnr"]
    for m in metrics:
        lv = lgbm_results.get(m, "N/A") if lgbm_results else "N/A"
        ev = ens_results.get(m, "N/A")  if ens_results  else "N/A"
        lv_s = f"{lv:.4f}" if isinstance(lv, float) else str(lv)
        ev_s = f"{ev:.4f}" if isinstance(ev, float) else str(ev)
        print(f"  {m.upper():<14} {lv_s:>12} {ev_s:>12}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM + Ensemble on CIC-IDS-2018"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to CIC-IDS-2018 CSV file or directory containing CSVs",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None,
        help="Maximum rows to evaluate (random sample, for quick tests)",
    )
    parser.add_argument(
        "--skip_lgbm", action="store_true",
        help="Skip LightGBM evaluation",
    )
    parser.add_argument(
        "--skip_ensemble", action="store_true",
        help="Skip Ensemble evaluation",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Path not found: {data_path}")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    raw_df = load_cicids2018(data_path, max_rows=args.max_rows)

    # ── Preprocess ────────────────────────────────────────────────────────────
    df, y_binary, y_multi, y_multi_str, all_classes = preprocess(raw_df)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    lgbm_results = None
    ens_results  = None

    if not args.skip_lgbm:
        lgbm_results = evaluate_lgbm(df, y_binary, y_multi_str)

    if not args.skip_ensemble:
        ens_results = evaluate_ensemble(df, y_binary)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(lgbm_results, ens_results)

    # ── Save ──────────────────────────────────────────────────────────────────
    if lgbm_results or ens_results:
        save_results(lgbm_results or {}, ens_results or {}, str(data_path))


if __name__ == "__main__":
    main()
