import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = "models"
OUT = Path("data/sample_inference.csv")

def main(n_rows=10):
    feat_path = os.path.join(MODELS_DIR, "fusion_features.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    if not Path(feat_path).exists():
        print("Could not find features file:", feat_path)
        print("Make sure you ran training and models/fusion_features.pkl exists.")
        return

    features = joblib.load(feat_path)
    print(f"Loaded {len(features)} feature names.")

    # If scaler exists, use its data-range (min/max) to sample realistic values
    if Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
        # MinMaxScaler stores min_ and data_min_ attributes (scikit-learn)
        try:
            data_min = scaler.data_min_
            data_max = scaler.data_max_
            # if scaler was fit on full dataset these arrays align with features
            mins = np.array(data_min)
            maxs = np.array(data_max)
            # Handle any degenerate ranges
            ranges = np.maximum(maxs - mins, 1e-6)
            X = np.random.random(size=(n_rows, len(features))) * ranges + mins
            df = pd.DataFrame(X, columns=features)
            print("Sample rows generated using scaler ranges.")
        except Exception as e:
            print("Scaler present but couldn't read min/max:", e)
            df = pd.DataFrame(np.zeros((n_rows, len(features))), columns=features)
    else:
        # fallback: zeros
        df = pd.DataFrame(np.zeros((n_rows, len(features))), columns=features)
        print("Scaler not found — created zero-filled template.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Saved sample inference CSV to:", OUT.resolve())

if __name__ == "__main__":
    main(n_rows=10)