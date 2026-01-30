import pandas as pd
from anomly_detector.src.application_level_attacks_anomality.payload_features import extract_payload_features

INPUT_CSV = "data_collector/data_sets/application_level/payload_train.csv"
OUTPUT_CSV = "anomly_detector/src/application_level_attacks_anomality/web_features_train.csv"


df = pd.read_csv(INPUT_CSV)

feature_rows = df["payload"].apply(extract_payload_features)
X = pd.DataFrame(list(feature_rows))

label_map = {
    "norm" : 0,
    "anom" : 1
}

y = df["label"].map(label_map).fillna(-1).astype(int)
X["label"] = y
X.to_csv(OUTPUT_CSV, index=False)

print(f"[INFO] Saved extracted features to {OUTPUT_CSV}")