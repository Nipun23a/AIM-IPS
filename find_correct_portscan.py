"""
find_correct_portscan.py
Finds portscan rows from your CSVs that the model predicts correctly,
then prints their exact feature values for use in synthetic flows.

Run from project root:
    python find_correct_portscan.py --data data_collector/data_sets/cicids
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

THREAT_FEATURES = [
    "flow duration", "total fwd packets", "total backward packets",
    "total length of fwd packets", "total length of bwd packets",
    "fwd packet length mean", "bwd packet length mean",
    "flow bytes/s", "flow packets/s", "syn flag count",
    "ack flag count", "psh flag count", "packet length mean",
    "packet length std", "idle mean", "idle std",
]

MODEL_PATH    = "models/threat_classifier/lgb_model.pkl"
FEATURES_PATH = "models/threat_classifier/features.pkl"

def simplify_label(label):
    label = str(label).strip().lower()
    if label == "benign":                    return "benign"
    if "dos" in label or "ddos" in label:   return "ddos"
    if "portscan" in label:                  return "portscan"
    return None

def main(data_dir):
    model    = joblib.load(MODEL_PATH)
    saved    = joblib.load(FEATURES_PATH)
    features = saved if isinstance(saved, list) else saved.get("features", THREAT_FEATURES)
    idx_to_label = {0: "benign", 1: "ddos", 2: "portscan"}
    classes = [idx_to_label.get(int(c), str(c)) for c in model.classes_]

    csvs = list(Path(data_dir).glob("*.csv"))
    print(f"Scanning {len(csvs)} CSVs for correctly-predicted portscan rows...\n")

    found_correct   = []
    found_incorrect = []

    for csv in csvs:
        df = pd.read_csv(csv, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        if "label" not in df.columns:
            continue
        df["_label"] = df["label"].apply(simplify_label)
        df = df[df["_label"] == "portscan"]
        if df.empty:
            continue

        missing = set(features) - set(df.columns)
        if missing:
            continue

        df = df[features + ["_label"]].copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
        if df.empty:
            continue

        X = df[features].astype(float)
        probs = model.predict_proba(X)
        preds = [classes[int(np.argmax(p))] for p in probs]

        for i, (pred, row_probs) in enumerate(zip(preds, probs)):
            row = df[features].iloc[i].to_dict()
            entry = {
                "pred":       pred,
                "confidence": float(np.max(row_probs)),
                "probs":      {classes[j]: float(row_probs[j]) for j in range(len(classes))},
                "features":   row,
            }
            if pred == "portscan" and len(found_correct) < 5:
                found_correct.append(entry)
            elif pred != "portscan" and len(found_incorrect) < 3:
                found_incorrect.append(entry)

        if len(found_correct) >= 5:
            break

    # Print correctly classified portscan rows
    print("=" * 66)
    print(f"  CORRECTLY PREDICTED PORTSCAN ROWS: {len(found_correct)}")
    print("=" * 66)
    for i, e in enumerate(found_correct):
        print(f"\n  --- row {i+1} | pred={e['pred']} conf={e['confidence']:.3f} ---")
        for feat in THREAT_FEATURES:
            print(f"    {feat:40s}: {e['features'][feat]}")

    print()
    print("=" * 66)
    print(f"  MISCLASSIFIED PORTSCAN ROWS (predicted as benign/ddos): {len(found_incorrect)}")
    print("=" * 66)
    for i, e in enumerate(found_incorrect):
        print(f"\n  --- row {i+1} | pred={e['pred']} conf={e['confidence']:.3f} ---")
        for feat in THREAT_FEATURES:
            print(f"    {feat:40s}: {e['features'][feat]}")

    if not found_correct:
        print("\n  !! No correctly classified portscan rows found.")
        print("  The model may have low portscan recall — check the")
        print("  classification report from your training script.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    args = p.parse_args()
    main(args.data)