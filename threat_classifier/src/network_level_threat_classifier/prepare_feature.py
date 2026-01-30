import pandas as pd
from threat_classifier.src.features import THREAT_FEATURES

def prepare_threat_features(paths):
    dfs = []

    for path in paths:
        df = pd.read_csv(path)

        # normalize columns
        df.columns = df.columns.str.strip().str.lower()

        # keep only required features
        missing = set(THREAT_FEATURES) - set(df.columns)
        if missing:
            raise RuntimeError(f"Missing features in {path.name}: {missing}")

        X = df[THREAT_FEATURES].copy()
        X = X.dropna()

        dfs.append(X)

    X_all = pd.concat(dfs, ignore_index=True)
    return X_all, THREAT_FEATURES
