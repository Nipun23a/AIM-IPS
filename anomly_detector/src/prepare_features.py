import pandas as pd
import numpy as np
from typing import Tuple, List

def load_csv(path):
    return pd.read_csv(path)

def intersect_features(dfs: List[pd.DataFrame]) -> List[str]:
    # Return intersection of column names across provided DataFrames
    sets = [set(df.columns) for df in dfs]
    inter = set.intersection(*sets)
    inter = list(inter)
    # Remove columns that are obviously non-numeric or identifiers we don't want
    blacklist = {'srcip','proto','service','state','dstip','Timestamp','Flow ID','Label','attack_cat','attack_category','id','start_time','end_time'}
    inter = [c for c in inter if c not in blacklist]
    # Keep stable order
    inter.sort()
    return inter

def prepare_from_paths(paths, label_col_candidates=None):
    # load all datasets
    dfs = [load_csv(p) for p in paths]
    features = intersect_features(dfs)
    print(f"[prepare] Found {len(features)} intersecting features.")
    # For each df, keep only intersecting features; drop rows with NaN in those features
    Xs = []
    Ys = []
    for df in dfs:
        dff = df[features].copy()
        # try to find a label column (common names)
        label = None
        possible = ['Label','label','Attack','attack','class','Class','attack_cat']
        for cand in possible:
            if cand in df.columns:
                label = df[cand]
                break
        # if label not found, mark as unknown (-1)
        if label is None:
            label = pd.Series([-1]*len(dff))
        # drop rows with missing values in features
        mask = ~dff.isnull().any(axis=1)
        Xs.append(dff[mask].astype(float))
        Ys.append(label[mask].astype(int))
    # concat all
    X = pd.concat(Xs, ignore_index=True)
    Y = pd.concat(Ys, ignore_index=True)
    return X, Y, features
