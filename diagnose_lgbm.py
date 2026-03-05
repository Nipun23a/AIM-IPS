"""
diagnose_lgbm.py
═══════════════════════════════════════════════════════════════
Diagnoses why the LGBM model predicts benign=1.000 for all flows.

Checks:
  1. Model internals  — classes, feature importances, leaf counts
  2. Training data    — actual feature ranges per class in CSVs
  3. Boundary probe   — sweeps each feature to find decision boundary
  4. Direct CSV row   — feeds a real DDoS row straight to the model

Run from project root:
    python diagnose_lgbm.py
    python diagnose_lgbm.py --data path/to/cicids/csv
═══════════════════════════════════════════════════════════════
"""

import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg):   print(f"  {RED}✗{RESET}  {msg}")
def warn(msg):   print(f"  {YELLOW}⚠{RESET}  {msg}")
def info(msg):   print(f"  {CYAN}→{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}\n" + "─" * 62)
def dim(msg):    print(f"  {DIM}{msg}{RESET}")


DEFAULT_MODEL_PATH    = "models/threat_classifier/lgb_model.pkl"
DEFAULT_FEATURES_PATH = "models/threat_classifier/features.pkl"

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

LABEL_MAP = {"benign": 0, "ddos": 1, "portscan": 2}

def simplify_label(label):
    label = str(label).strip().lower()
    if label == "benign":       return "benign"
    if "dos" in label or "ddos" in label: return "ddos"
    if "portscan" in label:     return "portscan"
    return None


# ──────────────────────────────────────────────────────────────
# CHECK 1 — Model internals
# ──────────────────────────────────────────────────────────────

def check_model_internals(model, features, classes):
    header("CHECK 1 — Model Internals")

    info(f"Model type     : {type(model).__name__}")
    info(f"Classes stored : {model.classes_.tolist() if hasattr(model,'classes_') else 'N/A'}")
    info(f"Classes mapped : {classes}")
    info(f"Num estimators : {model.n_estimators_}")
    info(f"Num features   : {len(features)}")

    # Feature importances
    print()
    info("Feature importances (gain):")
    importances = model.feature_importances_
    ranked = sorted(zip(features, importances), key=lambda x: -x[1])
    for feat, imp in ranked:
        bar = "█" * int(imp / max(importances) * 30)
        print(f"    {feat:40s} {bar:30s} {imp:.1f}")

    # Warn if top features are unexpected
    top3 = [f for f, _ in ranked[:3]]
    print()
    if any(f in top3 for f in ["flow bytes/s", "flow packets/s", "syn flag count", "total fwd packets"]):
        ok("Top features look reasonable for DDoS/portscan detection")
    else:
        warn(f"Unexpected top features: {top3}")
        warn("Model may have learned from wrong columns")


# ──────────────────────────────────────────────────────────────
# CHECK 2 — Real data ranges per class
# ──────────────────────────────────────────────────────────────

def check_data_ranges(data_dir: str):
    header("CHECK 2 — Real Training Data Feature Ranges")

    data_path = Path(data_dir)
    csvs = list(data_path.glob("*.csv"))

    if not csvs:
        warn(f"No CSVs found in {data_path} — skipping")
        warn("Pass --data <path> pointing to your CICIDS CSVs")
        return None

    info(f"Found {len(csvs)} CSV(s) in {data_path}")

    frames = []
    for csv in csvs:
        df = pd.read_csv(csv, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        if "label" not in df.columns:
            warn(f"  No label column in {csv.name} — skipping")
            continue
        df["label"] = df["label"].apply(simplify_label)
        df = df[df["label"].notnull()]

        missing = set(THREAT_FEATURES) - set(df.columns)
        if missing:
            warn(f"  {csv.name} missing: {missing}")
            continue

        frames.append(df)
        counts = df["label"].value_counts().to_dict()
        info(f"  {csv.name}: {counts}")

    if not frames:
        fail("No usable CSVs loaded")
        return None

    combined = pd.concat(frames, ignore_index=True)

    print()
    info("Feature ranges per class (mean ± std)  [min … max]:")
    print()

    focus = [
        "flow bytes/s",
        "flow packets/s",
        "total fwd packets",
        "syn flag count",
        "flow duration",
        "packet length mean",
    ]

    for feat in focus:
        print(f"  {BOLD}{feat}{RESET}")
        for cls in ["benign", "ddos", "portscan"]:
            subset = combined[combined["label"] == cls][feat].dropna()
            if subset.empty:
                dim(f"    {cls:12s}: no data")
                continue
            print(
                f"    {CYAN}{cls:12s}{RESET}: "
                f"mean={subset.mean():.2f}  std={subset.std():.2f}  "
                f"[{subset.min():.2f} … {subset.max():.2f}]  "
                f"n={len(subset):,}"
            )
        print()

    return combined


# ──────────────────────────────────────────────────────────────
# CHECK 3 — Decision boundary probe
# Sweep one feature at a time; keep others at benign baseline
# ──────────────────────────────────────────────────────────────

def check_boundary_probe(model, features, classes):
    header("CHECK 3 — Decision Boundary Probe")

    info("Sweeping key features from benign baseline to attack values.")
    info("Shows when (if ever) model tips away from benign.\n")

    benign_baseline = {
        "flow duration":                  2.5,
        "total fwd packets":              8,
        "total backward packets":         6,
        "total length of fwd packets":    1200,
        "total length of bwd packets":    8000,
        "fwd packet length mean":         150.0,
        "bwd packet length mean":         1333.0,
        "flow bytes/s":                   3680.0,
        "flow packets/s":                 5.6,
        "syn flag count":                 1,
        "ack flag count":                 12,
        "psh flag count":                 4,
        "packet length mean":             654.0,
        "packet length std":              580.0,
        "idle mean":                      0.1,
        "idle std":                       0.05,
    }

    sweep_specs = [
        ("flow packets/s",    [5, 50, 500, 2000, 5000, 10000, 50000, 100000]),
        ("flow bytes/s",      [3680, 50000, 144000, 500000, 720000, 2000000]),
        ("total fwd packets", [8, 100, 500, 1000, 4800, 9000]),
        ("syn flag count",    [0, 1, 10, 100, 1000, 4800]),
    ]

    tipped = False

    for feat_name, values in sweep_specs:
        print(f"  {BOLD}{feat_name}{RESET}")
        for v in values:
            row = {**benign_baseline, feat_name: v}
            X = pd.DataFrame([[row[f] for f in features]], columns=features)
            probs = model.predict_proba(X)[0]
            pred  = classes[int(np.argmax(probs))]
            b, d, p = probs[0], probs[1], probs[2]

            marker = f"{GREEN}benign{RESET}" if pred == "benign" else f"{RED}{pred}{RESET}"
            print(
                f"    {feat_name}={v:<12}  →  {marker:30s}  "
                f"b={b:.3f}  d={d:.3f}  p={p:.3f}"
            )
            if pred != "benign":
                tipped = True
        print()

    if not tipped:
        fail("Model NEVER left benign — even with extreme attack values.")
        fail("This strongly suggests a column name mismatch between")
        fail("training data columns and the features list saved in features.pkl.")
    else:
        ok("Model does tip to attack class — boundary found.")


# ──────────────────────────────────────────────────────────────
# CHECK 4 — Feed a real CSV row directly to the model
# ──────────────────────────────────────────────────────────────

def check_real_row(model, features, classes, data_dir: str):
    header("CHECK 4 — Real CSV Row Prediction")

    data_path = Path(data_dir)
    csvs = list(data_path.glob("*.csv"))

    if not csvs:
        warn("No CSVs — skipping")
        return

    for csv in csvs:
        df = pd.read_csv(csv, nrows=50000, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()

        if "label" not in df.columns:
            continue

        df["_simple"] = df["label"].apply(simplify_label)
        df = df[df["_simple"].notnull()]

        missing = set(features) - set(df.columns)
        if missing:
            warn(f"  {csv.name} missing cols: {missing}")
            continue

        info(f"Sampling from {csv.name}")

        for cls in ["benign", "ddos", "portscan"]:
            subset = df[df["_simple"] == cls]
            if subset.empty:
                continue

            row = subset.sample(1, random_state=42)
            X = row[features].astype(float)

            probs     = model.predict_proba(X)[0]
            pred_idx  = int(np.argmax(probs))
            pred      = classes[pred_idx]
            expected  = cls

            status = ok if pred == expected else fail
            status(
                f"Real {cls:10s} row → predicted={BOLD}{pred}{RESET}  "
                f"b={probs[0]:.3f}  d={probs[1]:.3f}  p={probs[2]:.3f}"
            )

            # Print the actual feature values for the sampled row
            print(f"  {DIM}  Feature values from CSV row:{RESET}")
            for feat in ["flow bytes/s", "flow packets/s", "total fwd packets",
                         "syn flag count", "flow duration"]:
                if feat in row.columns:
                    print(f"  {DIM}    {feat:40s}: {row[feat].values[0]}{RESET}")
        break   # one CSV is enough


# ──────────────────────────────────────────────────────────────
# CHECK 5 — Column name audit
# ──────────────────────────────────────────────────────────────

def check_column_names(data_dir: str):
    header("CHECK 5 — Column Name Audit")

    data_path = Path(data_dir)
    csvs = list(data_path.glob("*.csv"))

    if not csvs:
        warn("No CSVs — skipping")
        return

    csv = csvs[0]
    df  = pd.read_csv(csv, nrows=1)

    raw_cols      = list(df.columns)
    stripped_cols = [c.strip().lower() for c in raw_cols]

    print(f"  Checking {csv.name}")
    print(f"  {len(raw_cols)} columns total\n")

    all_match = True
    for feat in THREAT_FEATURES:
        if feat in stripped_cols:
            ok(f"{feat}")
        else:
            # Show closest actual column
            close = [c for c in stripped_cols if feat.split()[0] in c]
            fail(f"{feat}  ← NOT FOUND.  Similar: {close[:3]}")
            all_match = False

    print()
    if all_match:
        ok("All 16 feature names match CSV columns after strip+lower")
    else:
        fail("Column name mismatch — this is likely the root cause")
        info("The model was trained on different column names than features.pkl expects")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Diagnose LGBM classifier")
    p.add_argument("--model",    default=DEFAULT_MODEL_PATH)
    p.add_argument("--features", default=DEFAULT_FEATURES_PATH)
    p.add_argument("--data",     default=None,
                   help="Path to folder containing CICIDS CSV files")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{BOLD}{'═' * 62}")
    print("  LGBM DIAGNOSTIC TOOL")
    print(f"{'═' * 62}{RESET}")

    # Load model
    mp = Path(args.model)
    fp = Path(args.features)

    if not mp.exists():
        print(f"\n{RED}ERROR:{RESET} model not found at {mp}")
        sys.exit(1)

    model    = joblib.load(mp)
    saved    = joblib.load(fp) if fp.exists() else THREAT_FEATURES
    features = saved if isinstance(saved, list) else saved.get("features", THREAT_FEATURES)

    idx_to_label = {0: "benign", 1: "ddos", 2: "portscan"}
    classes = [
        idx_to_label.get(int(c), str(c))
        for c in (model.classes_ if hasattr(model, "classes_") else [0, 1, 2])
    ]

    info(f"model    : {mp}")
    info(f"features : {fp}")
    info(f"classes  : {classes}")

    # Run checks
    check_model_internals(model, features, classes)
    check_boundary_probe(model, features, classes)

    if args.data:
        check_column_names(args.data)
        check_data_ranges(args.data)
        check_real_row(model, features, classes, args.data)
    else:
        print(f"\n  {YELLOW}⚠{RESET}  Skipping data checks — pass --data <path/to/cicids/csvs>")
        print(f"  {YELLOW}⚠{RESET}  e.g.  python diagnose_lgbm.py --data data_collector/data_sets/cicids")

    print()


if __name__ == "__main__":
    main()