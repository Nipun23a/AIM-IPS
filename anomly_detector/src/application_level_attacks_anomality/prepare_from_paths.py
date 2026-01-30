import pandas as pd
import numpy as np
from typing import Tuple, List


def load_csv(path:str) -> pd.DataFrame:
    return pd.read_csv(path)

def intersect_features(dfs: List[pd.DataFrame]) -> List[str]:
    sets = [set(df.columns) for df in dfs]
    inter = set.intersection(*sets)
    blacklist = {
        # identifiers / text
        "payload",
        "attack_type",

        # labels
        "label", "Label", "class", "Class", "attack", "Attack",
        "attack_cat", "attack_category",

        # flow-based leftovers (safe to keep for compatibility)
        "srcip", "dstip", "proto", "service", "state",
        "Timestamp", "Flow ID", "id",
        "start_time", "end_time",
    }

    features = sorted([c for c in inter if c not in blacklist])
    return features

def prepare_from_paths(paths: List[str], label_col_candidates: List[str]=None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if label_col_candidates is None:
        label_col_candidates = ["label", "Label", "class", "Class"]

    # Load datasets
    dfs = [load_csv(p) for p in paths]

    # Identify numeric feature columns
    features = intersect_features(dfs)
    print(f"[prepare] Found {len(features)} intersecting numeric features.")

    Xs, Ys = [], []

    for df in dfs:
        # -------------------------
        # Features
        # -------------------------
        X = df[features].copy()

        # Ensure numeric
        X = X.apply(pd.to_numeric, errors="coerce")

        # -------------------------
        # Labels
        # -------------------------
        label_series = None
        for col in label_col_candidates:
            if col in df.columns:
                label_series = df[col]
                break

        if label_series is None:
            y = pd.Series([-1] * len(X))
        else:
            # Handle BOTH numeric and string labels
            if np.issubdtype(label_series.dtype, np.number):
                y = label_series.astype(int)
            else:
                label_series = label_series.astype(str).str.lower().str.strip()
                y = label_series.map({
                    "0": 0,
                    "norm": 0,
                    "normal": 0,
                    "1": 1,
                    "anom": 1,
                    "anomaly": 1
                }).fillna(-1)

        # -------------------------
        # Drop rows with NaN features
        # -------------------------
        mask = ~X.isnull().any(axis=1)

        Xs.append(X[mask].astype(float))
        Ys.append(y[mask].astype(int))

    # Concatenate all files
    X_all = pd.concat(Xs, ignore_index=True)
    y_all = pd.concat(Ys, ignore_index=True)

    return X_all, y_all, features