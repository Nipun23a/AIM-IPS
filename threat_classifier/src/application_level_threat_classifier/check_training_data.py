# threat_classifier/src/application_level_threat_classifier/check_training_data.py

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data_collector" / "data_sets" / "application_level"

# Load dataset
df = pd.read_csv(DATA_DIR / "combined_threat_dataset_clean.csv")

print("\n" + "="*100)
print("TRAINING DATA ANALYSIS - Looking for problematic patterns")
print("="*100)

# Check for UNION SELECT in different classes
print("\n1. Checking UNION SELECT distribution:")
union_select_samples = df[df['payload'].str.contains('UNION', case=False, na=False) & 
                          df['payload'].str.contains('SELECT', case=False, na=False)]
print(f"\nTotal UNION SELECT samples: {len(union_select_samples)}")
print(union_select_samples['attack_type'].value_counts())

print("\n   Sample UNION SELECT payloads by class:")
for attack_type in union_select_samples['attack_type'].unique():
    samples = union_select_samples[union_select_samples['attack_type'] == attack_type]['payload'].head(3)
    print(f"\n   {attack_type}:")
    for s in samples:
        print(f"      {s[:80]}...")

# Check for API paths
print("\n2. Checking /api/ paths:")
api_paths = df[df['payload'].str.contains('/api/', case=False, na=False)]
print(f"\nTotal /api/ samples: {len(api_paths)}")
print(api_paths['attack_type'].value_counts())

print("\n   Sample /api/ payloads by class:")
for attack_type in api_paths['attack_type'].unique():
    samples = api_paths[api_paths['attack_type'] == attack_type]['payload'].head(3)
    print(f"\n   {attack_type}:")
    for s in samples:
        print(f"      {s[:80]}...")

# Check for quotes + SQL keywords labeled as norm
print("\n3. Checking norm samples with quotes + SQL keywords:")
norm_with_sql = df[(df['attack_type'] == 'norm') & 
                   (df['payload'].str.contains("'", na=False)) &
                   (df['payload'].str.contains('SELECT|UNION|INSERT', case=False, na=False))]
print(f"\nTotal norm with quotes + SQL: {len(norm_with_sql)}")
print("\n   Sample payloads:")
for s in norm_with_sql['payload'].head(10):
    print(f"      {s[:80]}...")

# Check sqli samples with UNION SELECT
print("\n4. Checking sqli samples with UNION SELECT:")
sqli_union = df[(df['attack_type'] == 'sqli') & 
                (df['payload'].str.contains('UNION', case=False, na=False))]
print(f"\nTotal sqli with UNION: {len(sqli_union)}")
print("\n   Sample payloads:")
for s in sqli_union['payload'].head(10):
    print(f"      {s[:80]}...")