"""
Prepare CICIDS2017 training set by merging multiple CSV files.

Usage:
    # from project root (activate venv first)
    python -m src.prepare_cicids \
        --input data/CIC1.csv data/CIC2.csv data/CIC3.csv data/CIC4.csv data/CIC5.csv \
        --out data/cicids_combined.csv \
        --train_out data/cicids_train.csv \
        --test_out data/cicids_test.csv \
        --test_size 0.2 \
        --balance none

Options for --balance: none, downsample, upsample
"""

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

COMMON_LABEL_NAMES = ['label', 'Label', 'attack', 'Attack', 'class', 'Class', 'flow_label', 'Flow ID']

def normalize_colname(c: str) -> str:
    # Normalize column names to lower_snake for easier matching
    c = str(c).strip()
    c = c.replace(' ', '_').replace('-', '_')
    return c.lower()

def find_label_series(df: pd.DataFrame):
    # Try to find a label column and normalize to 0/1 (0 normal, 1 attack)
    # Common CICIDS values: 'BENIGN' (or 'Normal') and attack names
    for cand in COMMON_LABEL_NAMES:
        if cand in df.columns:
            s = df[cand].copy()
            # map common text labels
            s = s.astype(str).str.strip()
            s_lower = s.str.lower()
            # common normal tokens
            normal_tokens = {'benign','normal','0','no','none','normal flow'}
            # map to 0/1
            mapped = s_lower.apply(lambda v: 0 if v in normal_tokens else (1 if v not in ['', 'nan', 'none', 'na'] else -1))
            # if numeric already (0/1)
            if pd.api.types.is_numeric_dtype(df[cand]):
                mapped = df[cand].fillna(-1).astype(int).apply(lambda x: 0 if x==0 else (1 if x>0 else -1))
            return mapped
    # no label column found
    return pd.Series([-1]*len(df), index=df.index)

def load_and_clean(path):
    print(f"[load] Reading {path}")
    # read with low_memory to avoid dtype inference issues
    df = pd.read_csv(path, low_memory=False)
    # standardize column names
    df.columns = [normalize_colname(c) for c in df.columns]
    # find label column (after normalizing names)
    label_series = find_label_series(df)
    # attach/overwrite a canonical 'label' column (0 normal, 1 attack, -1 unknown)
    df['label'] = label_series
    # drop obviously unwanted identifier columns if present
    drop_candidates = ['flow_id', 'id', 'timestamp', 'start_time', 'end_time', 'time', 'flow_id.1']
    for d in drop_candidates:
        if d in df.columns:
            try:
                df = df.drop(columns=[d])
            except Exception:
                pass
    # Convert numeric-looking columns to numeric where possible
    for col in df.columns:
        if col == 'label':
            continue
        # try to coerce
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def intersect_and_align(dfs):
    # Find intersection of columns (excluding label which we keep)
    col_sets = [set(df.columns) for df in dfs]
    common = set.intersection(*col_sets)
    # always keep 'label' if present in any df
    common = set(common)  # copy
    if not any('label' in df.columns for df in dfs):
        # ensure 'label' present (we created it earlier so likely yes)
        common.add('label')
    # Convert to sorted list for deterministic order
    common = sorted(list(common))
    # Ensure label is last column
    if 'label' in common:
        common.remove('label')
        common.append('label')
    # Align each dataframe to the common columns (if missing, fill NaN)
    aligned = []
    for df in dfs:
        missing = [c for c in common if c not in df.columns]
        if missing:
            # create missing cols filled with NaN
            for m in missing:
                df[m] = np.nan
        aligned.append(df[common].copy())
    return aligned, common

def fill_missing_numeric(df):
    # Fill numeric columns' NaN with median; categorical (all-NaN or object) with mode or -1
    for c in df.columns:
        if c == 'label':
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            median = df[c].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            df[c] = df[c].fillna(median)
        else:
            # object type
            mode = df[c].mode(dropna=True)
            if len(mode) > 0:
                df[c] = df[c].fillna(mode.iloc[0])
            else:
                df[c] = df[c].fillna(-1)
    return df

def balance_dataframe(df, method='none', label_col='label', random_state=42):
    if method == 'none':
        return df
    # filter labelled
    if 'label' not in df.columns:
        print("[balance] No label column found; skipping balancing.")
        return df
    df_labeled = df[df[label_col] != -1].copy()
    df_unlabeled = df[df[label_col] == -1].copy()
    # separate classes
    df_normal = df_labeled[df_labeled[label_col]==0]
    df_attack = df_labeled[df_labeled[label_col]==1]
    if len(df_normal) == 0 or len(df_attack)==0:
        print("[balance] One of classes empty, skipping balancing.")
        return df
    if method == 'downsample':
        # downsample majority to minority
        if len(df_normal) > len(df_attack):
            df_normal_down = resample(df_normal, replace=False, n_samples=len(df_attack), random_state=random_state)
            df_bal = pd.concat([df_normal_down, df_attack, df_unlabeled], ignore_index=True)
        else:
            df_attack_down = resample(df_attack, replace=False, n_samples=len(df_normal), random_state=random_state)
            df_bal = pd.concat([df_normal, df_attack_down, df_unlabeled], ignore_index=True)
    elif method == 'upsample':
        # upsample minority to majority
        if len(df_normal) < len(df_attack):
            df_normal_up = resample(df_normal, replace=True, n_samples=len(df_attack), random_state=random_state)
            df_bal = pd.concat([df_normal_up, df_attack, df_unlabeled], ignore_index=True)
        else:
            df_attack_up = resample(df_attack, replace=True, n_samples=len(df_normal), random_state=random_state)
            df_bal = pd.concat([df_normal, df_attack_up, df_unlabeled], ignore_index=True)
    else:
        df_bal = df
    return df_bal

def main(args):
    paths = args.input
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found")
        dfs.append(load_and_clean(p))

    # align columns by intersection
    aligned, common = intersect_and_align(dfs)
    print(f"[main] {len(common)} common columns including 'label'. Columns: {common[:10]} ...")

    # concat all
    combined = pd.concat(aligned, ignore_index=True)
    print(f"[main] Combined shape before fill: {combined.shape}")

    # Fill missing values
    combined = fill_missing_numeric(combined)

    # Optionally balance
    if args.balance not in ('none', 'downsample', 'upsample'):
        print(f"[main] Unknown balance method {args.balance}, defaulting to 'none'")
        args.balance = 'none'
    combined = balance_dataframe(combined, method=args.balance)

    # Shuffle
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save combined
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"[main] Saved combined dataset to {args.out} (shape={combined.shape})")

    # Create train/test split using labeled rows only (label != -1)
    labeled = combined[combined['label'] != -1].copy()
    unlabeled = combined[combined['label'] == -1].copy()
    if len(labeled) == 0:
        print("[main] No labeled rows found; skipping train/test split.")
        return

    X = labeled.drop(columns=['label'])
    y = labeled['label'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

    train_df = X_train.copy()
    train_df['label'] = y_train.values
    test_df = X_test.copy()
    test_df['label'] = y_test.values

    # Optionally append unlabeled to train set (useful for unsupervised AE)
    if args.append_unlabeled_to_train and len(unlabeled) > 0:
        train_df = pd.concat([train_df, unlabeled], ignore_index=True).sample(frac=1.0, random_state=42)

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)
    print(f"[main] Saved train ({train_df.shape}) -> {args.train_out}")
    print(f"[main] Saved test ({test_df.shape}) -> {args.test_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True,
                        help='Paths to input CSV files (list)')
    parser.add_argument('--out', default='data/cicids_combined.csv', help='Combined output CSV path')
    parser.add_argument('--train_out', default='data/cicids_train.csv', help='Train CSV path')
    parser.add_argument('--test_out', default='data/cicids_test.csv', help='Test CSV path')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split fraction')
    parser.add_argument('--balance', type=str, default='none', help='none/downsample/upsample')
    parser.add_argument('--append_unlabeled_to_train', action='store_true', help='Append unlabeled rows to training set')
    args = parser.parse_args()
    main(args)