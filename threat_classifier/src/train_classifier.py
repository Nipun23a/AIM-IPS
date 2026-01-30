import os
import joblib
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets" / "cicids"
MODELS_DIR = PROJECT_ROOT / "models" / "threat_classifier"

os.makedirs(MODELS_DIR, exist_ok=True)

DATA_PATHS = [
    DATA_DIR / "cicids_train_part1.csv",
    DATA_DIR / "cicids_train_part2.csv",
    DATA_DIR / "cicids_train_part3.csv",
    DATA_DIR / "cicids_train_part4.csv",
    DATA_DIR / "cicids_train_part5.csv",
    DATA_DIR / "cicids_train_part6.csv",
    DATA_DIR / "cicids_train_part7.csv",
    DATA_DIR / "cicids_train_part8.csv",
]

# =========================
# Imports
# =========================
from threat_classifier.src.features import THREAT_FEATURES
from threat_classifier.src.labels import LABEL_MAP

# =========================
# Load datasets + prepare X,Y
# =========================
print("[train_classifier] Loading datasets and preparing features + labels...")

Xs = []
Ys = []

for path in DATA_PATHS:
    print(f"[train_classifier] Processing {path.name}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Ensure label column exists
    if "label" not in df.columns:
        print(f"[WARN] 'label' column not found in {path.name}, skipping")
        continue

    # Normalize label values
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.replace("�", "-", regex=False)
        .str.lower()
    )

    # Keep only known labels
    df = df[df["label"].isin(LABEL_MAP.keys())].copy()
    if df.empty:
        continue

    # Check required features
    missing = set(THREAT_FEATURES) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing features in {path.name}: {missing}")

    # Extract features + labels TOGETHER
    X = df[THREAT_FEATURES].astype(float)
    Y = df["label"].map(LABEL_MAP)

    # Drop rows with NaN (both X & Y stay aligned)
    mask = ~X.isnull().any(axis=1)
    X = X[mask]
    Y = Y[mask]

    Xs.append(X)
    Ys.append(Y)

# =========================
# Final dataset
# =========================
X_all = pd.concat(Xs, ignore_index=True)
Y_all = pd.concat(Ys, ignore_index=True)

print("\n[train_classifier] Final dataset shape:", X_all.shape)
print("[train_classifier] Label distribution:\n", Y_all.value_counts())

assert len(X_all) == len(Y_all), "X and Y size mismatch"
assert not X_all.isnull().any().any(), "NaN values found in features"

# =========================
# Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    Y_all,
    test_size=0.2,
    stratify=Y_all,
    random_state=42
)

# =========================
# Train LightGBM
# =========================
print("\n[train_classifier] Training LightGBM...")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=15,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)

print("\n[train_classifier] Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\n[train_classifier] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# Save model
# =========================
joblib.dump(model, MODELS_DIR / "lgb_model.pkl")
joblib.dump(THREAT_FEATURES, MODELS_DIR / "features.pkl")

print(f"\n[train_classifier] Model saved to: {MODELS_DIR}")
