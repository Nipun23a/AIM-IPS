import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================================
# CREATE UNBALANCED TEST DATASET FROM CICIDS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets"

# Input: Your 8 CICIDS parts
CICIDS_PARTS = [
    DATA_DIR / "cicids" / f"cicids_train_part{i}.csv" for i in range(1, 9)
]

# Output: Unbalanced test set
OUTPUT_PATH = DATA_DIR / "cicids_test_unbalanced.csv"


def normalize_colname(c: str) -> str:
    """
    Normalize column names to lower_snake for consistency.
    Same normalization used in training script.
    """
    c = str(c).strip()
    c = c.replace(' ', '_').replace('-', '_')
    return c.lower()


def find_label_series(df: pd.DataFrame):
    """
    Find and normalize label column to 0/1 (0=benign, 1=attack)
    Same logic as training script.
    """
    common_label_names = ['label', 'Label', 'attack', 'Attack', 'class', 'Class']
    
    for cand in common_label_names:
        if cand in df.columns:
            s = df[cand].copy()
            # Map common text labels
            s = s.astype(str).str.strip()
            s_lower = s.str.lower()
            # Common normal tokens
            normal_tokens = {'benign', 'normal', '0', 'no', 'none', 'normal flow'}
            # Map to 0/1
            mapped = s_lower.apply(lambda v: 0 if v in normal_tokens else (1 if v not in ['', 'nan', 'none', 'na'] else -1))
            # If numeric already (0/1)
            if pd.api.types.is_numeric_dtype(df[cand]):
                mapped = df[cand].fillna(-1).astype(int).apply(lambda x: 0 if x == 0 else (1 if x > 0 else -1))
            return mapped
    
    # No label column found
    return pd.Series([-1] * len(df), index=df.index)


def load_and_normalize(path):
    """
    Load CSV and normalize column names.
    Same as training script.
    """
    print(f"  Loading {path.name}...")
    df = pd.read_csv(path, low_memory=False)
    
    # Standardize column names (same as training)
    df.columns = [normalize_colname(c) for c in df.columns]
    
    # Find label column (after normalizing names)
    label_series = find_label_series(df)
    
    # Attach/overwrite a canonical 'label' column (0=benign, 1=attack)
    df['label'] = label_series
    
    # Drop obviously unwanted identifier columns if present
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
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def create_unbalanced_dataset(
    benign_ratio=0.95,  # 95% benign, 5% attack (realistic)
    total_samples=100000,  # Total samples in final dataset
    random_state=42
):
    """
    Creates an unbalanced dataset that mimics real-world network traffic.
    Uses EXACT same column normalization as training script.
    
    Args:
        benign_ratio: Proportion of benign traffic (0.9-0.99 is realistic)
        total_samples: Total number of samples in output
        random_state: Random seed for reproducibility
    """
    
    print("="*70)
    print("CREATING UNBALANCED TEST DATASET")
    print("="*70)
    
    all_data = []
    
    # Load all parts
    print("\n📂 Loading CICIDS parts...")
    for i, part_path in enumerate(CICIDS_PARTS, 1):
        if part_path.exists():
            df = load_and_normalize(part_path)
            all_data.append(df)
        else:
            print(f"  ⚠️  Part {i} not found: {part_path}")
    
    if not all_data:
        raise FileNotFoundError("No CICIDS parts found!")
    
    # Find common columns across all parts
    print("\n🔗 Finding common columns...")
    col_sets = [set(df.columns) for df in all_data]
    common_cols = set.intersection(*col_sets)
    common_cols = sorted(list(common_cols))
    
    # Ensure 'label' is last column
    if 'label' in common_cols:
        common_cols.remove('label')
        common_cols.append('label')
    
    print(f"   Found {len(common_cols)} common columns")
    print(f"   Sample columns: {common_cols[:5]}")
    
    # Align all dataframes to common columns
    aligned_dfs = []
    for df in all_data:
        aligned_dfs.append(df[common_cols].copy())
    
    # Combine all data
    print("\n🔗 Combining all parts...")
    combined_df = pd.concat(aligned_dfs, ignore_index=True)
    print(f"   Total samples loaded: {len(combined_df):,}")
    
    # Remove unlabeled samples (label == -1)
    labeled_df = combined_df[combined_df['label'] != -1].copy()
    print(f"   Labeled samples: {len(labeled_df):,}")
    
    # Separate benign and attack samples
    print(f"\n🔍 Separating by label...")
    benign_samples = labeled_df[labeled_df['label'] == 0]
    attack_samples = labeled_df[labeled_df['label'] == 1]
    
    print(f"   Benign samples (0): {len(benign_samples):,}")
    print(f"   Attack samples (1): {len(attack_samples):,}")
    
    # Calculate target counts
    n_benign = int(total_samples * benign_ratio)
    n_attack = total_samples - n_benign
    
    print(f"\n🎯 Target distribution:")
    print(f"   Benign: {n_benign:,} ({benign_ratio*100:.1f}%)")
    print(f"   Attack: {n_attack:,} ({(1-benign_ratio)*100:.1f}%)")
    
    # Sample with replacement if needed
    if n_benign > len(benign_samples):
        print(f"   ⚠️  Not enough benign samples, sampling with replacement")
        sampled_benign = benign_samples.sample(n=n_benign, replace=True, random_state=random_state)
    else:
        sampled_benign = benign_samples.sample(n=n_benign, replace=False, random_state=random_state)
    
    if n_attack > len(attack_samples):
        print(f"   ⚠️  Not enough attack samples, sampling with replacement")
        sampled_attack = attack_samples.sample(n=n_attack, replace=True, random_state=random_state)
    else:
        sampled_attack = attack_samples.sample(n=n_attack, replace=False, random_state=random_state)
    
    # Combine and shuffle
    unbalanced_df = pd.concat([sampled_benign, sampled_attack], ignore_index=True)
    unbalanced_df = unbalanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Fill missing values (same as training)
    print(f"\n🧹 Filling missing values...")
    for col in unbalanced_df.columns:
        if col == 'label':
            continue
        if pd.api.types.is_numeric_dtype(unbalanced_df[col]):
            median = unbalanced_df[col].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            unbalanced_df[col] = unbalanced_df[col].fillna(median)
    
    # Save
    print(f"\n💾 Saving to: {OUTPUT_PATH}")
    unbalanced_df.to_csv(OUTPUT_PATH, index=False)
    
    # Final stats
    final_benign = (unbalanced_df['label'] == 0).sum()
    final_attack = (unbalanced_df['label'] == 1).sum()
    
    print(f"\n✅ DATASET CREATED SUCCESSFULLY")
    print("="*70)
    print(f"Total samples:    {len(unbalanced_df):,}")
    print(f"Total columns:    {len(unbalanced_df.columns)}")
    print(f"Benign (0):       {final_benign:,} ({final_benign/len(unbalanced_df)*100:.2f}%)")
    print(f"Attack (1):       {final_attack:,} ({final_attack/len(unbalanced_df)*100:.2f}%)")
    print(f"Imbalance ratio:  {final_benign/final_attack:.2f}:1")
    print(f"\n📝 Column format: normalized to lowercase_with_underscores")
    print(f"📝 Label format:  0 = Benign, 1 = Attack")
    print(f"📝 Sample columns: {unbalanced_df.columns.tolist()[:5]}")
    print("="*70)
    
    return unbalanced_df


# ==========================================================
# PRESET SCENARIOS
# ==========================================================

def scenario_realistic():
    """Realistic network traffic: 95% benign, 5% attack"""
    return create_unbalanced_dataset(benign_ratio=0.95, total_samples=100000)

def scenario_moderate():
    """Moderate imbalance: 90% benign, 10% attack"""
    return create_unbalanced_dataset(benign_ratio=0.90, total_samples=100000)

def scenario_extreme():
    """Extreme imbalance: 99% benign, 1% attack"""
    return create_unbalanced_dataset(benign_ratio=0.99, total_samples=100000)

def scenario_custom(benign_pct, n_samples):
    """Custom scenario"""
    return create_unbalanced_dataset(
        benign_ratio=benign_pct/100, 
        total_samples=n_samples
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    import sys
    
    print("\n🎛️  UNBALANCED DATASET CREATOR")
    print("="*70)
    print("\nChoose a scenario:")
    print("  1. Realistic (95% benign, 5% attack) - RECOMMENDED")
    print("  2. Moderate (90% benign, 10% attack)")
    print("  3. Extreme (99% benign, 1% attack)")
    print("  4. Custom")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        scenario_realistic()
    elif choice == "2":
        scenario_moderate()
    elif choice == "3":
        scenario_extreme()
    elif choice == "4":
        benign_pct = float(input("Enter benign percentage (e.g., 95): "))
        n_samples = int(input("Enter total samples (e.g., 100000): "))
        scenario_custom(benign_pct, n_samples)
    else:
        print("Invalid choice. Running realistic scenario...")
        scenario_realistic()