import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ==========================================================
# Feature list (must match your IPS feature extractor)
# ==========================================================

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

# ==========================================================
# Label map
# ==========================================================

LABEL_MAP = {
    "benign": 0,
    "ddos": 1,
    "portscan": 2,
}

# ==========================================================
# Simplify CICIDS attack labels
# ==========================================================

def simplify_label(label):

    label = str(label).strip().lower()

    if label == "benign":
        return "benign"

    if "dos" in label or "ddos" in label:
        return "ddos"

    if "portscan" in label:
        return "portscan"

    return None


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets" / "cicids"
MODELS_DIR = PROJECT_ROOT / "models" / "threat_classifier"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATHS = list(DATA_DIR.glob("*.csv"))

print("Found datasets:", len(DATA_PATHS))

# ==========================================================
# Load and preprocess datasets
# ==========================================================

Xs = []
Ys = []

for path in DATA_PATHS:

    print("\nProcessing:", path.name)

    df = pd.read_csv(path)

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if "label" not in df.columns:
        print("Skipping file (no label column)")
        continue

    # simplify labels
    df["label"] = df["label"].apply(simplify_label)

    # remove unsupported labels
    df = df[df["label"].notnull()]

    if df.empty:
        print("No supported labels found")
        continue

    # ensure features exist
    missing = set(THREAT_FEATURES) - set(df.columns)

    if missing:
        raise RuntimeError(f"{path.name} missing features: {missing}")

    # extract features
    X = df[THREAT_FEATURES].astype(float)

    # remove NaN rows
    mask = ~X.isnull().any(axis=1)

    X = X[mask]
    y = df["label"].map(LABEL_MAP)[mask]

    Xs.append(X)
    Ys.append(y)

# ==========================================================
# Combine datasets
# ==========================================================

X_all = pd.concat(Xs, ignore_index=True)
y_all = pd.concat(Ys, ignore_index=True)

print("\nFinal dataset shape:", X_all.shape)

print("\nLabel distribution:")
print(y_all.value_counts())

assert len(X_all) == len(y_all)

# ==========================================================
# Train/Test split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    stratify=y_all,
    random_state=42
)

# ==========================================================
# Train LightGBM
# ==========================================================

print("\nTraining LightGBM...")

model = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================================
# Evaluation
# ==========================================================

print("\nEvaluating model...")

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

f1 = f1_score(y_test, y_pred, average="weighted")

print("\nWeighted F1-score:", round(f1, 4))

# ==========================================================
# Save model
# ==========================================================

joblib.dump(model, MODELS_DIR / "lgb_model.pkl")
joblib.dump(THREAT_FEATURES, MODELS_DIR / "features.pkl")

print("\nModel saved to:", MODELS_DIR)