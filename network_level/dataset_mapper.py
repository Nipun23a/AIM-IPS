"""
dataset_mapper.py
=================
Maps CICIDS2017 to three output files for training both Network Layer IPS models.
No external dataset needed — CICIDS2017 only.

Outputs:
  - cicids_mapped.csv    → LightGBM (labeled multiclass: benign/ddos/dos/portscan/botnet)
  - cicids_benign.csv    → TCN training (benign traffic only, autoencoder learns normal)
  - cicids_novel.csv     → TCN calibration (Heartbleed + Infiltration + Web attacks)
  - unified_features.json → feature schema mirrored by live Scapy extractor

Attack split logic:
  LightGBM trains on:  DDoS, DoS, PortScan, Botnet  (known, high-volume)
  TCN calibrates on:   Heartbleed, Infiltration,
                       Web Attack (Brute Force, XSS, SQLi),
                       FTP/SSH Patator
                       → all treated as "novel" (not given to LightGBM)

Usage:
  python dataset_mapper.py --cicids path/to/MachineLearningCSV/ --out ./mapped_data
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# 1. UNIFIED FEATURE SCHEMA
#    32 features extractable from CICIDS2017 AND from live Scapy capture
# ---------------------------------------------------------------------------

UNIFIED_FEATURES = [
    # Duration & Volume
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",

    # Rates
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "fwd_pkts_per_sec",
    "bwd_pkts_per_sec",

    # Packet Length Stats
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_max",
    "pkt_len_min",

    # Inter-Arrival Time
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "fwd_iat_mean",
    "bwd_iat_mean",

    # TCP Flags
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",

    # Window & Header
    "init_win_fwd",
    "init_win_bwd",

    # TTL (imputed — not in CICFlowMeter output)
    "fwd_ttl_mean",
    "bwd_ttl_mean",

    # Directional packet length
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_std",

    # Ratio
    "down_up_ratio",
]

# ---------------------------------------------------------------------------
# 2. COLUMN MAPPING  (CICFlowMeter → unified names)
# ---------------------------------------------------------------------------

CICIDS_COLUMN_MAP = {
    "Flow Duration":                "flow_duration",
    "Total Fwd Packets":            "total_fwd_packets",
    "Total Backward Packets":       "total_bwd_packets",
    "Total Length of Fwd Packets":  "total_fwd_bytes",
    "Total Length of Bwd Packets":  "total_bwd_bytes",
    "Flow Bytes/s":                 "flow_bytes_per_sec",
    "Flow Packets/s":               "flow_pkts_per_sec",
    "Fwd Packets/s":                "fwd_pkts_per_sec",
    "Bwd Packets/s":                "bwd_pkts_per_sec",
    "Packet Length Mean":           "pkt_len_mean",
    "Packet Length Std":            "pkt_len_std",
    "Max Packet Length":            "pkt_len_max",
    "Min Packet Length":            "pkt_len_min",
    "Flow IAT Mean":                "flow_iat_mean",
    "Flow IAT Std":                 "flow_iat_std",
    "Flow IAT Max":                 "flow_iat_max",
    "Fwd IAT Mean":                 "fwd_iat_mean",
    "Bwd IAT Mean":                 "bwd_iat_mean",
    "FIN Flag Count":               "fin_flag_count",
    "SYN Flag Count":               "syn_flag_count",
    "RST Flag Count":               "rst_flag_count",
    "PSH Flag Count":               "psh_flag_count",
    "ACK Flag Count":               "ack_flag_count",
    "Init_Win_bytes_forward":       "init_win_fwd",
    "Init_Win_bytes_backward":      "init_win_bwd",
    "Fwd Packet Length Mean":       "fwd_pkt_len_mean",
    "Bwd Packet Length Mean":       "bwd_pkt_len_mean",
    "Fwd Packet Length Std":        "fwd_pkt_len_std",
    "Bwd Packet Length Std":        "bwd_pkt_len_std",
    "Down/Up Ratio":                "down_up_ratio",
    # TTL not produced by CICFlowMeter — imputed below
}

# ---------------------------------------------------------------------------
# 3. LABEL ROUTING
#
#    LIGHTGBM_LABEL_MAP  → cicids_mapped.csv  (known, labeled attacks)
#    NOVEL_LABEL_MAP     → cicids_novel.csv   (TCN calibration)
#    "BENIGN"            → both cicids_mapped.csv AND cicids_benign.csv
# ---------------------------------------------------------------------------

# LightGBM sees these with their specific class labels
LIGHTGBM_LABEL_MAP = {
    "BENIGN":           "benign",
    "DDoS":             "ddos",
    "DoS Hulk":         "dos",
    "DoS GoldenEye":    "dos",
    "DoS slowloris":    "dos",
    "DoS Slowhttptest": "dos",
    "PortScan":         "portscan",
    "Bot":              "botnet",
}

# TCN calibration — all non-DDoS/PortScan/Botnet attacks treated as "novel"
# These are behaviorally unusual enough to serve as zero-day proxies
NOVEL_LABEL_MAP = {
    # Handle both encoding variants CICIDS CSVs ship with
    "Heartbleed":                       "novel",
    "Infiltration":                     "novel",
    "Web Attack \x96 Brute Force":      "novel",
    "Web Attack \x96 XSS":              "novel",
    "Web Attack \x96 Sql Injection":    "novel",
    "Web Attack \xe2\x80\x93 Brute Force":  "novel",
    "Web Attack \xe2\x80\x93 XSS":          "novel",
    "Web Attack \xe2\x80\x93 Sql Injection": "novel",
    "Web Attack – Brute Force":         "novel",
    "Web Attack – XSS":                 "novel",
    "Web Attack – Sql Injection":       "novel",
    "FTP-Patator":                      "novel",
    "SSH-Patator":                      "novel",
}

# ---------------------------------------------------------------------------
# 4. LOADER
# ---------------------------------------------------------------------------

def load_cicids(data_dir: str) -> pd.DataFrame:
    """
    Load all CICIDS2017 CSV files from a directory.
    CICIDS ships as one CSV per weekday — Monday through Friday.
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}\n"
            f"Expected the MachineLearningCSV directory from CICIDS2017."
        )

    print(f"[LOAD] Found {len(csv_files)} CSV file(s):")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df.columns = df.columns.str.strip()  # CICIDS has leading spaces
            dfs.append(df)
            label_col = next(
                (c for c in df.columns if c.lower() == "label"), None
            )
            if label_col:
                counts = df[label_col].value_counts().to_dict()
                print(f"  {f.name}: {len(df):,} rows | {counts}")
            else:
                print(f"  {f.name}: {len(df):,} rows | (no label col found)")
        except Exception as e:
            print(f"  WARNING: Could not load {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n[LOAD] Total combined rows: {len(combined):,}")
    return combined


# ---------------------------------------------------------------------------
# 5. FEATURE MAPPING
# ---------------------------------------------------------------------------

def map_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename CICFlowMeter columns to unified names and impute missing features.
    Returns df with exactly UNIFIED_FEATURES + 'raw_label' columns.
    """
    # Find and normalise label column
    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"), None
    )
    if label_col is None:
        raise ValueError("Cannot find Label column in CICIDS data.")
    df["raw_label"] = df[label_col].str.strip()

    # Rename feature columns
    df = df.rename(columns=CICIDS_COLUMN_MAP)

    # Impute TTL — CICFlowMeter does not extract TTL
    # 64 is the standard Linux default; Windows uses 128
    df["fwd_ttl_mean"] = 64.0
    df["bwd_ttl_mean"] = 64.0

    # Ensure all unified features exist
    for feat in UNIFIED_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    return df[UNIFIED_FEATURES + ["raw_label"]].copy()


# ---------------------------------------------------------------------------
# 6. CLEANING  (shared for all three outputs)
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Remove inf/NaN, clip outliers, enforce float32."""
    print(f"\n[CLEAN] {name}")
    before = len(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=UNIFIED_FEATURES)
    print(f"  Dropped {before - len(df):,} inf/NaN rows")

    # Cap extreme rate values at 99.9th percentile
    rate_cols = [
        "flow_bytes_per_sec", "flow_pkts_per_sec",
        "fwd_pkts_per_sec",   "bwd_pkts_per_sec",
    ]
    for col in rate_cols:
        if col in df.columns and df[col].max() > 0:
            cap = df[col].quantile(0.999)
            df[col] = df[col].clip(upper=cap)

    # All features must be non-negative
    for col in UNIFIED_FEATURES:
        df[col] = df[col].clip(lower=0)

    # Enforce float32 throughout
    for col in UNIFIED_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)

    print(f"  Clean rows: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# 7. SPLIT INTO THREE OUTPUTS
# ---------------------------------------------------------------------------

def split_outputs(df: pd.DataFrame):
    """
    Partition cleaned data into three dataframes:

      lgbm_df    — rows whose raw_label is in LIGHTGBM_LABEL_MAP
                   includes benign + known attacks with specific class labels

      benign_df  — benign rows only (for TCN autoencoder training)

      novel_df   — rows whose raw_label is in NOVEL_LABEL_MAP
                   all labelled "novel" for TCN threshold calibration
    """
    # LightGBM partition
    lgbm_mask = df["raw_label"].isin(LIGHTGBM_LABEL_MAP)
    lgbm_df = df[lgbm_mask].copy()
    lgbm_df["unified_label"] = lgbm_df["raw_label"].map(LIGHTGBM_LABEL_MAP)

    # Benign-only partition (subset of lgbm_df)
    benign_df = lgbm_df[lgbm_df["unified_label"] == "benign"].copy()

    # Novel partition
    novel_mask = df["raw_label"].isin(NOVEL_LABEL_MAP)
    novel_df = df[novel_mask].copy()
    novel_df["unified_label"] = novel_df["raw_label"].map(NOVEL_LABEL_MAP)

    # Summary
    print("\n[SPLIT] Partition summary:")
    print(f"  LightGBM rows : {len(lgbm_df):,}")
    print(f"    {lgbm_df['unified_label'].value_counts().to_dict()}")
    print(f"  Benign (TCN)  : {len(benign_df):,}")
    print(f"  Novel  (TCN)  : {len(novel_df):,}")
    print(f"    {novel_df['raw_label'].value_counts().to_dict()}")

    unlabelled = (~lgbm_mask & ~novel_mask).sum()
    if unlabelled > 0:
        unlabelled_labels = df[~lgbm_mask & ~novel_mask]["raw_label"].value_counts()
        print(f"\n  ⚠  {unlabelled:,} rows not routed to any output:")
        print(f"    {unlabelled_labels.to_dict()}")

    return lgbm_df, benign_df, novel_df


# ---------------------------------------------------------------------------
# 8. BALANCING
# ---------------------------------------------------------------------------

def balance_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance LightGBM training data.
    Strategy:
      1. Undersample benign to 3x total attack count (avoids SMOTE OOM)
      2. SMOTE on minority attack classes
    """
    print("\n[BALANCE] LightGBM data...")
    print(f"  Before: {df['unified_label'].value_counts().to_dict()}")

    le = LabelEncoder()

    # Step 1 — undersample benign
    benign_mask  = df["unified_label"] == "benign"
    attack_count = int((~benign_mask).sum())
    target_benign = min(int(benign_mask.sum()), attack_count * 3)

    benign_idx = df[benign_mask].sample(n=target_benign, random_state=42).index
    attack_idx = df[~benign_mask].index
    df_sub = df.loc[benign_idx.union(attack_idx)].copy()

    X_sub = df_sub[UNIFIED_FEATURES].values
    y_sub = le.fit_transform(df_sub["unified_label"])

    # Step 2 — SMOTE on remaining minority classes
    try:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_sub, y_sub)
        result = pd.DataFrame(X_res, columns=UNIFIED_FEATURES)
        result["unified_label"] = le.inverse_transform(y_res)
        print(f"  After SMOTE: {result['unified_label'].value_counts().to_dict()}")
        return result
    except Exception as e:
        print(f"  WARNING: SMOTE failed ({e}), using undersampled data only")
        df_sub["unified_label"] = le.inverse_transform(y_sub)
        return df_sub[UNIFIED_FEATURES + ["unified_label"]]


def balance_novel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance novel (TCN calibration) data.
    All rows are already "novel" — cap at 50k to keep calibration fast.
    No oversampling needed since TCN uses scores not class labels.
    """
    print("\n[BALANCE] Novel (TCN calibration) data...")
    print(f"  Raw attack breakdown:")
    print(f"  {df['raw_label'].value_counts().to_dict()}")

    if len(df) > 50_000:
        df = df.sample(n=50_000, random_state=42)
        print(f"  Downsampled to 50,000 rows")

    return df[UNIFIED_FEATURES + ["unified_label"]].copy()


# ---------------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------------

def main(cicids_dir: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    raw = load_cicids(cicids_dir)

    # --- Map features ---
    print("\n[MAP] Mapping CICFlowMeter columns to unified features...")
    mapped = map_features(raw)
    print(f"  Mapped rows: {len(mapped):,}")
    print(f"  Raw label distribution:\n{mapped['raw_label'].value_counts()}")

    # --- Clean ---
    cleaned = clean(mapped, "CICIDS2017")

    # --- Split ---
    lgbm_df, benign_df, novel_df = split_outputs(cleaned)

    # --- Balance ---
    lgbm_balanced  = balance_lgbm(lgbm_df)
    benign_out     = benign_df[UNIFIED_FEATURES].copy()   # no label col for TCN train
    novel_balanced = balance_novel(novel_df)

    # --- Save ---
    lgbm_path   = out_dir / "cicids_mapped.csv"
    benign_path = out_dir / "cicids_benign.csv"
    novel_path  = out_dir / "cicids_novel.csv"

    lgbm_balanced.to_csv(lgbm_path,   index=False)
    benign_out.to_csv(benign_path,    index=False)
    novel_balanced.to_csv(novel_path, index=False)

    print(f"\n✅ LightGBM data  → {lgbm_path}   ({len(lgbm_balanced):,} rows)")
    print(f"✅ TCN benign     → {benign_path}  ({len(benign_out):,} rows)")
    print(f"✅ TCN novel      → {novel_path}   ({len(novel_balanced):,} rows)")

    # --- Feature schema for live extractor ---
    schema = {
        "unified_features":   UNIFIED_FEATURES,
        "feature_count":      len(UNIFIED_FEATURES),
        "lightgbm_targets":   ["benign", "ddos", "dos", "portscan", "botnet"],
        "tcn_novel_sources":  list(NOVEL_LABEL_MAP.keys()),
        "lightgbm_label_map": LIGHTGBM_LABEL_MAP,
        "novel_label_map":    NOVEL_LABEL_MAP,
        "notes": {
            "fwd_ttl_mean": "imputed at 64.0 (not in CICFlowMeter)",
            "bwd_ttl_mean": "imputed at 64.0 (not in CICFlowMeter)",
            "dataset":      "CICIDS2017 only",
        }
    }
    schema_path = out_dir / "unified_features.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"✅ Feature schema → {schema_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DATASET MAPPING COMPLETE")
    print("=" * 60)
    print(f"  Source          CICIDS2017 only")
    print(f"  Features        {len(UNIFIED_FEATURES)} unified features")
    print(f"  LightGBM rows   {len(lgbm_balanced):,}  "
          f"({lgbm_balanced['unified_label'].nunique()} classes)")
    print(f"  TCN benign      {len(benign_out):,}  (autoencoder training)")
    print(f"  TCN novel       {len(novel_balanced):,}  (threshold calibration)")
    print(f"  Output dir      {out_dir}")
    print("\nNext steps:")
    print("  python train_lightgbm.py --data ./mapped_data/cicids_mapped.csv")
    print("  python train_tcn.py --benign ./mapped_data/cicids_benign.csv \\")
    print("                      --novel  ./mapped_data/cicids_novel.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map CICIDS2017 to unified feature space for LightGBM + TCN"
    )
    parser.add_argument(
        "--cicids", required=True,
        help="Path to CICIDS2017 MachineLearningCSV directory"
    )
    parser.add_argument(
        "--out", default="./mapped_data",
        help="Output directory (default: ./mapped_data)"
    )
    args = parser.parse_args()
    main(args.cicids, args.out)