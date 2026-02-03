import json
import joblib
import numpy as np
from pathlib import Path

from threat_classifier.src.application_level_threat_classifier.feature_engineering import (
    ThreatFeatureExtractor
)

class ApplicationThreatClassifier:
    def __init__(self, model_dir: Path):
        # Load model
        self.model = joblib.load(model_dir / "threat_classifier_lgb.pkl")

        # Load feature config
        self.feature_extractor = ThreatFeatureExtractor()
        self.feature_extractor.load_feature_config(
            model_dir / "feature_config.pkl"
        )

        # Load class mapping
        with open(model_dir / "class_mapping.json") as f:
            mapping = json.load(f)
            self.idx_to_class = {
                int(k): v for k, v in mapping["idx_to_class"].items()
            }

    def predict(self, payload: str):
        """
        Returns:
          - max_score (float)
          - predicted_class (str)
          - full_probs (dict)
        """
        features = self.feature_extractor.extract_features(payload)

        # Align feature order
        X = np.array([
            features[name] for name in self.feature_extractor.feature_names
        ]).reshape(1, -1)

        probs = self.model.predict(X)[0]

        max_idx = int(np.argmax(probs))
        max_score = float(np.max(probs))
        predicted_class = self.idx_to_class[max_idx]

        return max_score, predicted_class, {
            self.idx_to_class[i]: float(p)
            for i, p in enumerate(probs)
        }
