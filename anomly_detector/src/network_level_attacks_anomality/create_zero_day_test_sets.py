# create_zero_day_test_sets.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets"
TEST_SETS_DIR = DATA_DIR / "zero_day_tests"

os.makedirs(TEST_SETS_DIR, exist_ok=True)

# Load your combined CICIDS data
CICIDS_TRAIN = DATA_DIR / "cicids_train.csv"


def analyze_attack_distribution():
    """
    Analyze what attack types exist in your dataset
    """
    print("="*70)
    print("ANALYZING CICIDS2017 ATTACK DISTRIBUTION")
    print("="*70)
    
    df = pd.read_csv(CICIDS_TRAIN, low_memory=False)
    
    # Check if there's an attack_type or Label column
    attack_col = None
    for col in ['attack_cat', 'Attack', 'Label', 'attack_category', 'attack_type']:
        if col in df.columns:
            attack_col = col
            break
    
    if attack_col:
        print(f"\n📊 Attack type distribution (column: '{attack_col}'):")
        attack_dist = df[attack_col].value_counts()
        print(attack_dist)
        
        # Save attack types
        attack_types = attack_dist.index.tolist()
        return df, attack_col, attack_types
    else:
        print("\n⚠️  No attack type column found. Only binary label available.")
        print("   We'll create synthetic zero-day scenarios using statistical patterns.")
        return df, None, None


def create_zero_day_scenario_1(df, attack_col, attack_types):
    """
    Scenario 1: Hold out specific attack types for zero-day testing
    """
    print("\n" + "="*70)
    print("SCENARIO 1: ZERO-DAY ATTACK TYPE TESTING")
    print("="*70)
    
    if not attack_col or not attack_types:
        print("❌ Cannot create attack-type holdout without attack labels")
        return None
    
    # Remove 'BENIGN' from attack types
    attack_types_only = [a for a in attack_types if 'benign' not in str(a).lower()]
    
    if len(attack_types_only) < 3:
        print("❌ Not enough attack types for holdout")
        return None
    
    # Hold out 20-30% of attack types for testing
    n_holdout = max(1, len(attack_types_only) // 4)
    
    print(f"\n🎯 Strategy: Hold out {n_holdout} attack type(s) for zero-day testing")
    print(f"   Total attack types: {len(attack_types_only)}")
    
    # Select holdout attacks (e.g., least common ones)
    attack_counts = df[df['label'] == 1][attack_col].value_counts()
    holdout_attacks = attack_counts.nsmallest(n_holdout).index.tolist()
    
    print(f"\n🔒 Held-out (zero-day) attacks:")
    for att in holdout_attacks:
        count = attack_counts[att]
        print(f"   • {att}: {count:,} samples")
    
    # Split data
    seen_mask = ~df[attack_col].isin(holdout_attacks)
    holdout_mask = df[attack_col].isin(holdout_attacks)
    
    train_data = df[seen_mask]
    zero_day_test = df[holdout_mask]
    
    print(f"\n📊 Data split:")
    print(f"   Training (seen attacks): {len(train_data):,}")
    print(f"   Zero-day test: {len(zero_day_test):,}")
    
    # Save datasets
    train_path = TEST_SETS_DIR / "train_without_holdout.csv"
    test_path = TEST_SETS_DIR / "zero_day_holdout_attacks.csv"
    
    train_data.to_csv(train_path, index=False)
    zero_day_test.to_csv(test_path, index=False)
    
    print(f"\n💾 Saved:")
    print(f"   Training: {train_path}")
    print(f"   Zero-day: {test_path}")
    
    # Save metadata
    metadata = {
        'scenario': 'zero_day_attack_holdout',
        'holdout_attacks': holdout_attacks,
        'train_samples': len(train_data),
        'test_samples': len(zero_day_test),
        'train_path': str(train_path),
        'test_path': str(test_path)
    }
    joblib.dump(metadata, TEST_SETS_DIR / "scenario1_metadata.pkl")
    
    return metadata


def create_zero_day_scenario_2(df):
    """
    Scenario 2: Extreme value test (statistical outliers)
    Create samples with extreme statistical characteristics
    """
    print("\n" + "="*70)
    print("SCENARIO 2: STATISTICAL EXTREME VALUE TESTING")
    print("="*70)
    
    # Separate benign and attack
    benign = df[df['label'] == 0]
    attack = df[df['label'] == 1]
    
    # Find attacks with extreme statistical properties
    # (top/bottom 10% of key features)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'label']
    
    print(f"\n🔬 Finding statistically extreme attacks...")
    
    # Calculate feature percentiles
    extreme_mask = pd.Series([False] * len(attack))
    
    for col in numeric_cols[:10]:  # Use first 10 numeric features
        if col in attack.columns:
            p95 = attack[col].quantile(0.95)
            p05 = attack[col].quantile(0.05)
            
            # Mark samples in top/bottom 5%
            extreme_this_col = (attack[col] > p95) | (attack[col] < p05)
            extreme_mask = extreme_mask | extreme_this_col
    
    extreme_attacks = attack[extreme_mask]
    normal_attacks = attack[~extreme_mask]
    
    print(f"\n📊 Attack distribution:")
    print(f"   Normal attacks: {len(normal_attacks):,}")
    print(f"   Extreme attacks: {len(extreme_attacks):,} ({len(extreme_attacks)/len(attack)*100:.1f}%)")
    
    # Create training set: benign + normal attacks
    # Test set: extreme attacks (simulating zero-day with unusual patterns)
    train_data = pd.concat([benign, normal_attacks])
    test_data = extreme_attacks
    
    # Create balanced test set
    test_benign = benign.sample(n=len(test_data), random_state=42)
    balanced_test = pd.concat([test_benign, test_data])
    balanced_test = balanced_test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 Final split:")
    print(f"   Training: {len(train_data):,}")
    print(f"   Extreme test: {len(balanced_test):,} (50% benign, 50% extreme attacks)")
    
    # Save
    train_path = TEST_SETS_DIR / "train_normal_patterns.csv"
    test_path = TEST_SETS_DIR / "zero_day_extreme_patterns.csv"
    
    train_data.to_csv(train_path, index=False)
    balanced_test.to_csv(test_path, index=False)
    
    print(f"\n💾 Saved:")
    print(f"   Training: {train_path}")
    print(f"   Test: {test_path}")
    
    metadata = {
        'scenario': 'extreme_statistical_patterns',
        'train_samples': len(train_data),
        'test_samples': len(balanced_test),
        'train_path': str(train_path),
        'test_path': str(test_path)
    }
    joblib.dump(metadata, TEST_SETS_DIR / "scenario2_metadata.pkl")
    
    return metadata


def create_zero_day_scenario_3():
    """
    Scenario 3: Temporal holdout
    Use different days for training vs testing
    """
    print("\n" + "="*70)
    print("SCENARIO 3: TEMPORAL ZERO-DAY (DIFFERENT TIME PERIODS)")
    print("="*70)
    
    # Check if your data has different parts/days
    parts = list((DATA_DIR / "cicids").glob("cicids_train_part*.csv"))
    
    if len(parts) < 4:
        print("❌ Not enough temporal splits available")
        return None
    
    print(f"\n📅 Found {len(parts)} temporal splits")
    
    # Use first 75% for training, last 25% for zero-day testing
    n_train = int(len(parts) * 0.75)
    
    train_parts = parts[:n_train]
    test_parts = parts[n_train:]
    
    print(f"\n📊 Temporal split:")
    print(f"   Training periods: {[p.name for p in train_parts]}")
    print(f"   Zero-day periods: {[p.name for p in test_parts]}")
    
    # Load and combine
    print("\n🔄 Loading data...")
    train_dfs = [pd.read_csv(p, low_memory=False) for p in train_parts]
    test_dfs = [pd.read_csv(p, low_memory=False) for p in test_parts]
    
    train_data = pd.concat(train_dfs, ignore_index=True)
    test_data = pd.concat(test_dfs, ignore_index=True)
    
    # Create imbalanced test (95:5)
    test_benign = test_data[test_data['label'] == 0]
    test_attack = test_data[test_data['label'] == 1]
    
    n_test = 20000
    n_benign = int(n_test * 0.95)
    n_attack = n_test - n_benign
    
    sampled_benign = test_benign.sample(n=min(n_benign, len(test_benign)), random_state=42)
    sampled_attack = test_attack.sample(n=min(n_attack, len(test_attack)), random_state=42)
    
    balanced_test = pd.concat([sampled_benign, sampled_attack])
    balanced_test = balanced_test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 Final datasets:")
    print(f"   Training: {len(train_data):,}")
    print(f"   Temporal zero-day test: {len(balanced_test):,}")
    
    # Save
    train_path = TEST_SETS_DIR / "train_early_periods.csv"
    test_path = TEST_SETS_DIR / "zero_day_later_periods.csv"
    
    train_data.to_csv(train_path, index=False)
    balanced_test.to_csv(test_path, index=False)
    
    print(f"\n💾 Saved:")
    print(f"   Training: {train_path}")
    print(f"   Test: {test_path}")
    
    metadata = {
        'scenario': 'temporal_holdout',
        'train_periods': [p.name for p in train_parts],
        'test_periods': [p.name for p in test_parts],
        'train_samples': len(train_data),
        'test_samples': len(balanced_test),
        'train_path': str(train_path),
        'test_path': str(test_path)
    }
    joblib.dump(metadata, TEST_SETS_DIR / "scenario3_metadata.pkl")
    
    return metadata


def main():
    """
    Create multiple zero-day test scenarios
    """
    print("\n" + "="*70)
    print("ZERO-DAY / UNSEEN ATTACK TEST SET CREATOR")
    print("="*70)
    print("\nThis will create test sets with attacks/patterns NOT seen during training")
    print("to evaluate model generalization and robustness to zero-day threats.")
    
    # Analyze data
    df, attack_col, attack_types = analyze_attack_distribution()
    
    scenarios_created = []
    
    # Scenario 1: Attack type holdout
    if attack_col and attack_types:
        meta1 = create_zero_day_scenario_1(df, attack_col, attack_types)
        if meta1:
            scenarios_created.append(meta1)
    
    # Scenario 2: Statistical extremes
    meta2 = create_zero_day_scenario_2(df)
    if meta2:
        scenarios_created.append(meta2)
    
    # Scenario 3: Temporal split
    meta3 = create_zero_day_scenario_3()
    if meta3:
        scenarios_created.append(meta3)
    
    # Summary
    print("\n" + "="*70)
    print("✅ ZERO-DAY TEST SETS CREATED")
    print("="*70)
    print(f"\nCreated {len(scenarios_created)} test scenarios:")
    for i, meta in enumerate(scenarios_created, 1):
        print(f"\n{i}. {meta['scenario'].upper()}")
        print(f"   Test samples: {meta['test_samples']:,}")
        print(f"   Test file: {Path(meta['test_path']).name}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Retrain your model on each training set")
    print("2. Evaluate on corresponding zero-day test set")
    print("3. Compare performance: CICIDS standard vs zero-day scenarios")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
