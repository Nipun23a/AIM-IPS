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

DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets"
MODELS_DIR = PROJECT_ROOT / "models" / "threat_classifier"

os.makedirs(MODELS_DIR, exist_ok=True)

DATA_PATHS = [
    DATA_DIR / "cicids_train.csv"
]

# =========================
# Imports (shared features + labels)
# =========================
from anomly_detector.src.prepare_features import prepare_from_paths
from threat_classifier.src.labels import LABEL_MAP

# =========================
# Load RAW dataset (for labels)
# =========================
print("[train_classifier] Loading raw dataset...")
df_raw = pd.read_csv(DATA_PATHS[0])

# Keep only rows with labels we know how to classify
df_raw = df_raw[df_raw["label"].isin(LABEL_MAP.keys())].copy()

if df_raw.empty:
    raise RuntimeError("No matching labels found. Check LABEL_MAP vs dataset labels.")

# =========================
# Feature extraction (shared pipeline)
# =========================
print("[train_classifier] Extracting features using shared pipeline...")
X_all, _, features = prepare_from_paths(DATA_PATHS)

# Align features with filtered raw labels
X = X_all.loc[df_raw.index]
Y = df_raw["label"].map(LABEL_MAP)

# =========================
# Safety checks
# =========================
print("\n[train_classifier] Class distribution:")
print(Y.value_counts())

if Y.isna().any():
    raise RuntimeError("Label mapping produced NaN values. Fix LABEL_MAP.")

# =========================
# Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

# =========================
# Train LightGBM
# =========================
print("\n[train_classifier] Training LightGBM classifier...")

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
print(classification_report(y_test, y_pred))

print("\n[train_classifier] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# Save model + feature list
# =========================
joblib.dump(model, MODELS_DIR / "lgb_model.pkl")
joblib.dump(features, MODELS_DIR / "features.pkl")

print(f"\n[train_classifier] Model and features saved to: {MODELS_DIR}")
