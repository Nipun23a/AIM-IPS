import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets" / "application_level" 

def standardize_attack_types(attack_type):
    """Standardize attack type names to consistent categories"""
    if pd.isna(attack_type):
        return 'norm'
    
    attack_type = str(attack_type).lower().strip()
    
    # SQL Injection variations
    if 'sqli' in attack_type or 'sql' in attack_type:
        return 'sqli'
    
    # XSS variations
    elif 'xss' in attack_type:
        return 'xss'
    
    # Command Injection variations
    elif 'cmd' in attack_type or 'cmdi' in attack_type:
        return 'cmdi'
    
    # Path Traversal variations
    elif 'path' in attack_type or 'traversal' in attack_type:
        return 'path-traversal'
    
    # Auth Bruteforce - SKIP (we don't need this)
    elif 'auth' in attack_type or 'bruteforce' in attack_type:
        return None  # Will be removed
    
    # Normal traffic
    elif attack_type == 'norm' or attack_type == 'normal':
        return 'norm'
    
    # Unknown - treat as normal
    else:
        return 'norm'

def prepare_datasets():
    
    # Load honeypot data
    print("[INFO] Loading honeypot data...")
    honeypot = pd.read_csv(DATA_DIR / "honeypot_data.csv")
    honeypot['payload'] = honeypot["path"] + " " + honeypot["query_params"].fillna("")
    honeypot['attack_type'] = honeypot['tags'].str.split(':').str[0]
    honeypot['attack_type'] = honeypot['attack_type'].apply(standardize_attack_types)
    honeypot['label'] = honeypot['attack_type']
    print(f"  Honeypot samples: {len(honeypot)}")
    
    # Load payload data
    print("[INFO] Loading payload_full data...")
    payload_data = pd.read_csv(DATA_DIR / "payload_full.csv")
    if 'attack_type' in payload_data.columns:
        payload_data['attack_type'] = payload_data['attack_type'].apply(standardize_attack_types)
        payload_data['label'] = payload_data['attack_type']
    print(f"  Payload_full samples: {len(payload_data)}")
    
    # SQL Injection - columns: Query, Label
    print("[INFO] Loading SQL injection data...")
    sql_injection_data = pd.read_csv(DATA_DIR / "sql_injection.csv")
    sql_injection_data = sql_injection_data.rename(columns={'Query': 'payload', 'Label': 'original_label'})
    sql_injection_data['attack_type'] = np.where(sql_injection_data['original_label'] == 1, 'sqli', 'norm')
    sql_injection_data['label'] = sql_injection_data['attack_type']
    print(f"  SQL injection samples: {len(sql_injection_data)}")
    
    # Command Injection - columns: sentence, Label
    print("[INFO] Loading command injection data...")
    cmd_data = pd.read_csv(DATA_DIR / "cmd_injection.csv")
    cmd_data = cmd_data.rename(columns={'sentence': 'payload', 'Label': 'original_label'})
    cmd_data['attack_type'] = np.where(cmd_data['original_label'] == 1, 'cmdi', 'norm')
    cmd_data['label'] = cmd_data['attack_type']
    print(f"  Command injection samples: {len(cmd_data)}")
    
    # XSS - columns: (unnamed index), Sentence, Label
    print("[INFO] Loading XSS data...")
    xss_data = pd.read_csv(DATA_DIR / "xss.csv")
    xss_data = xss_data.rename(columns={'Sentence': 'payload', 'Label': 'original_label'})
    xss_data['attack_type'] = np.where(xss_data['original_label'] == 1, 'xss', 'norm')
    xss_data['label'] = xss_data['attack_type']
    if 'Unnamed: 0' in xss_data.columns:
        xss_data = xss_data.drop('Unnamed: 0', axis=1)
    print(f"  XSS samples: {len(xss_data)}")
    
    # Combine all datasets
    print("\n[INFO] Combining all datasets...")
    all_data = []
    for name, df in [('honeypot', honeypot), 
                     ('payload_full', payload_data), 
                     ('sql_injection', sql_injection_data), 
                     ('cmd_injection', cmd_data), 
                     ('xss', xss_data)]:
        try:
            df_clean = df[['payload', 'attack_type', 'label']].copy()
            all_data.append(df_clean)
            print(f"  ✓ {name}: {len(df_clean)} samples added")
        except KeyError as e:
            print(f"  ✗ {name}: Error - {e}")
            print(f"    Available columns: {df.columns.tolist()}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove auth-bruteforce (only 4 samples) and any None values
    print(f"\n[INFO] Before filtering: {len(combined_df)} samples")
    combined_df = combined_df[combined_df['attack_type'].notna()]
    combined_df = combined_df[combined_df['attack_type'] != 'auth-bruteforce']
    print(f"[INFO] After removing auth-bruteforce: {len(combined_df)} samples")
    
    # Apply standardization one more time to ensure consistency
    combined_df['attack_type'] = combined_df['attack_type'].apply(standardize_attack_types)
    combined_df['label'] = combined_df['attack_type']
    
    # Final cleanup
    combined_df = combined_df.dropna()
    
    return combined_df

# Prepare and display results
combined_data = prepare_datasets()
print(f"\n[INFO] ========== FINAL DATASET SUMMARY ==========")
print(f"[INFO] Total samples: {len(combined_data)}")
print(f"\n[INFO] Attack types distribution:")
print(combined_data['attack_type'].value_counts())

print(f"\n[INFO] Percentage distribution:")
distribution = combined_data['attack_type'].value_counts(normalize=True) * 100
for attack_type, percentage in distribution.items():
    print(f"  {attack_type:20s}: {percentage:6.2f}%")

# Check sample payloads from each attack type
print(f"\n[INFO] Sample payloads:")
for attack_type in sorted(combined_data['attack_type'].unique()):
    sample = combined_data[combined_data['attack_type'] == attack_type]['payload'].iloc[0]
    print(f"  {attack_type:20s}: {sample[:60]}...")

# Save the cleaned dataset
output_path = DATA_DIR / "combined_threat_dataset_clean.csv"
combined_data.to_csv(output_path, index=False)
print(f"\n[INFO] ✅ Clean dataset saved to: {output_path}")

# Dataset quality check
print(f"\n[INFO] ========== DATASET QUALITY CHECK ==========")
print(f"✅ Total samples: {len(combined_data):,} (excellent for LightGBM)")
print(f"✅ Number of classes: {combined_data['attack_type'].nunique()}")
print(f"✅ No missing values: {combined_data.isnull().sum().sum() == 0}")

# Check class balance
min_class = combined_data['attack_type'].value_counts().min()
max_class = combined_data['attack_type'].value_counts().max()
imbalance_ratio = max_class / min_class

print(f"\n[INFO] Class imbalance ratio: {imbalance_ratio:.1f}:1")
if imbalance_ratio > 100:
    print("⚠️  High imbalance - recommend using class weights")
elif imbalance_ratio > 20:
    print("⚠️  Moderate imbalance - class weights recommended")
else:
    print("✅ Acceptable balance")