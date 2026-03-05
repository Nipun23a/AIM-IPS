"""
extract_real_rows.py
Prints the exact feature values of real DDoS and PortScan rows
from your CICIDS CSVs so we can build perfectly matching synthetic flows.

Run from project root:
    python extract_real_rows.py --data data_collector/data_sets/cicids
"""

import argparse
import pandas as pd
from pathlib import Path

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

def simplify_label(label):
    label = str(label).strip().lower()
    if label == "benign":           return "benign"
    if "dos" in label or "ddos" in label: return "ddos"
    if "portscan" in label:         return "portscan"
    return None

def extract(data_dir: str, n_samples: int = 5):
    csvs = list(Path(data_dir).glob("*.csv"))
    print(f"Found {len(csvs)} CSVs\n")

    collected = {"benign": [], "ddos": [], "portscan": []}

    for csv in csvs:
        if all(len(v) >= n_samples for v in collected.values()):
            break

        df = pd.read_csv(csv, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()

        if "label" not in df.columns:
            continue

        df["_label"] = df["label"].apply(simplify_label)
        df = df[df["_label"].notnull()]

        missing = set(THREAT_FEATURES) - set(df.columns)
        if missing:
            print(f"  skipping {csv.name} — missing: {missing}")
            continue

        for cls in ["benign", "ddos", "portscan"]:
            if len(collected[cls]) >= n_samples:
                continue
            subset = df[df["_label"] == cls][THREAT_FEATURES].dropna()
            # filter out inf values
            subset = subset[~subset.isin([float("inf"), float("-inf")]).any(axis=1)]
            rows = subset.head(n_samples - len(collected[cls]))
            collected[cls].extend(rows.to_dict("records"))

    # Print
    for cls in ["benign", "ddos", "portscan"]:
        rows = collected[cls]
        print("=" * 66)
        print(f"  CLASS: {cls.upper()}  ({len(rows)} rows)")
        print("=" * 66)
        for i, row in enumerate(rows):
            print(f"\n  --- row {i+1} ---")
            for feat in THREAT_FEATURES:
                print(f"    {feat:40s}: {row[feat]}")
        print()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CICIDS CSV folder")
    p.add_argument("--n",    type=int, default=5, help="Rows per class")
    args = p.parse_args()
    extract(args.data, args.n)