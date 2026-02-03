import joblib
import numpy as np
from pathlib import Path

from anomly_detector.src.application_level_attacks_anomality.payload_features import extract_payload_features
from anomly_detector.src.application_level_attacks_anomality.train_tflight import TFLiteFastDetector
from anomly_detector.src.application_level_attacks_anomality.train_lightweight_cnn import mahalanobis_scores

class ApplicationDeepAnomalyDetector:
    """
    Application-level deep anomaly detector
    (TFLite CNN/MLP + optional PCA/Mahalanobis fusion)
    """
    def __init__(self, model_dir: Path, use_fusion=True):
        self.model_dir = model_dir
        self.use_fusion = use_fusion

        # Load TFLite CNN/MLP
        self.detector = TFLiteFastDetector(
            tflite_model_path=model_dir / "fusion_cnn.tflite",
            scaler_path=model_dir / "scaler.pkl",
            features_path=model_dir / "fusion_features.pkl",
            pca_path=(model_dir / "fusion_pca.pkl" if use_fusion else None),
            maha_mean_path=(model_dir / "fusion_maha_mean.pkl" if use_fusion else None),
            maha_inv_cov_path=(model_dir / "fusion_maha_inv_cov.pkl" if use_fusion else None),
        )

        # Load threshold
        self.threshold = joblib.load(
            model_dir / "fusion_cnn_threshold.pkl"
        )

    def predict(self, payload: str):
        """
        Returns:
            fused_score (0–1)
            is_anomaly (bool)
        """
        # Extract lightweight payload features
        features = extract_payload_features(payload)

        # CNN / MLP score (FAST: 2–5ms)
        cnn_score = self.detector.predict_single(features)

        if not self.use_fusion:
            return cnn_score, cnn_score > self.threshold

        # PCA + Mahalanobis
        X = np.array([[features[f] for f in self.detector.features]])
        X_scaled = self.detector.scaler.transform(X)

        maha_score = mahalanobis_scores(
            X_scaled,
            self.detector.pca,
            self.detector.maha_mean,
            self.detector.maha_inv_cov
        )[0]

        # Normalize Mahalanobis
        maha_norm = maha_score / (maha_score + 1e-6)

        # Fusion gate (same as training)
        fused = 0.7 * cnn_score + 0.3 * maha_norm
        fused = float(np.clip(fused, 0.0, 1.0))

        return fused, fused > self.threshold

