import pandas as pd

# Path to your CICIDS training file
CSV_PATH = "data_collector/data_sets/cicids_train.csv"

# Possible label column names (robust)
LABEL_COL_CANDIDATES = [
    "Label", "label",
    "Attack", "attack",
    "attack_cat", "Class"
]

# Load CSV
df = pd.read_csv(CSV_PATH)

# Detect label column
label_col = None
for col in LABEL_COL_CANDIDATES:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    raise RuntimeError(
        f"No label column found. Available columns:\n{df.columns.tolist()}"
    )

print(f"[INFO] Using label column: {label_col}")

# Get unique labels (non-repeated)
unique_labels = sorted(df[label_col].dropna().unique())

print("\n[INFO] Unique labels found in dataset:\n")
for lbl in unique_labels:
    print(lbl)

print(f"\n[INFO] Total unique labels: {len(unique_labels)}")
