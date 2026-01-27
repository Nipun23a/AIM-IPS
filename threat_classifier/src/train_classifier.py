

import sys
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import os
sys.path.append(os.path.abspath("../../anomly_detector/src"))

from prepare_features import prepare_from_paths
from labels import LABEL_MAP

MODEL_DIR = "../../models/threat_classifier"
os.makedirs(MODEL_DIR, exist_ok = True)


DATA_PATHS  = ["../../anomly_detector/data/cicids_train.csv"]

print("[train_classifier] Preparing data...")
X, Y, features  = prepare_from_paths(DATA_PATHS)


mask = Y != -1
X= X[mask]
Y= Y[mask]

Y = Y.map(LABEL_MAP)

print("[train] Class distribution: ")
print(Y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

print("[train] Training LigtGBM classifier...")
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("[train] Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, f"{MODEL_DIR}/lgb_model.pkl")
joblib.dump(features, f"{MODEL_DIR}/features.pkl")

print(f"[train] Model and features saved to {MODEL_DIR}")

