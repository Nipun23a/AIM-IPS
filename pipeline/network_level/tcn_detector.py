import logging
import numpy as np
import joblib
from pathlib import Path

from pipeline.network_level.feature import (
    THREAT_FEATURES,
    TCN_TFLITE_PATH, TCN_SCALER_PATH, TCN_FEATURES_PATH,
    LABEL_ZERODAY, LABEL_CLEAN,
)

logger = logging.getLogger(__name__)

# Tunable threshold — scores above this = zero-day attack
# Your TCN is a binary classifier (sigmoid output) — 0.5 is the natural threshold
# Raise to 0.6 to reduce false positives
TCN_ANOMALY_THRESHOLD = 0.5


class TCNDetector:
    """
    TFLite TCN binary classifier for zero-day network attack detection.

    Trained with binary labels:
        0 = BENIGN
        1 = attack (any attack type)

    Output: probability of attack (sigmoid) → 0.0–1.0
    """

    def __init__(
        self,
        tflite_path:   Path = None,
        scaler_path:   Path = None,
        features_path: Path = None,
    ):
        self.tflite_path   = Path(tflite_path)   if tflite_path   else Path(TCN_TFLITE_PATH)
        self.scaler_path   = Path(scaler_path)   if scaler_path   else Path(TCN_SCALER_PATH)
        self.features_path = Path(features_path) if features_path else Path(TCN_FEATURES_PATH)

        self.interpreter    = None
        self.scaler         = None
        self.features: list = THREAT_FEATURES
        self.input_details  = None
        self.output_details = None
        self._loaded        = False

    def load(self) -> "TCNDetector":
        import tensorflow as tf

        # ── TFLite model ─────────────────────────────────────
        if not self.tflite_path.exists():
            raise FileNotFoundError(f"[TCN] TFLite not found: {self.tflite_path}")

        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logger.info(
            f"[TCN] Loaded ← {self.tflite_path} | "
            f"input={self.input_details[0]['shape']} "
            f"dtype={self.input_details[0]['dtype'].__name__}"
        )

        # ── Scaler ────────────────────────────────────────────
        # Shared with CNN detector (saved by preprocess() in train_tcn_tflite.py)
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"[TCN] Scaler loaded ← {self.scaler_path}")
        else:
            logger.warning(f"[TCN] Scaler not found at {self.scaler_path} — using raw features")

        # ── Feature list ──────────────────────────────────────
        # feature_info.pkl saved by preprocess() in train_tcn_tflite.py
        # Format: {'features': [...], 'n_features': N, 'timestamp': '...'}
        if self.features_path.exists():
            info = joblib.load(self.features_path)
            if isinstance(info, dict):
                self.features = info.get("features", THREAT_FEATURES)
            elif isinstance(info, list):
                self.features = info
            logger.info(f"[TCN] {len(self.features)} features loaded")
        else:
            logger.info(f"[TCN] feature_info.pkl not found — using THREAT_FEATURES ({len(self.features)})")

        self._loaded = True
        return self

    def predict(self, features: dict) -> float:
        """
        Run TCN inference on a flow feature dict.

        Returns:
            float 0.0–1.0
            0.0 = benign
            1.0 = attack
        """

        if not self._loaded:
            return 0.0

        try:
            import pandas as pd

            # Build feature dataframe
            X_df = pd.DataFrame(
                [[features.get(f, 0.0) for f in self.features]],
                columns=self.features
            )

            # Apply scaler
            if self.scaler is not None:
                X = self.scaler.transform(X_df).astype(np.float32)
            else:
                X = X_df.values.astype(np.float32)

            # ----------------------------------------------------
            # FIX INPUT SHAPE
            # ----------------------------------------------------
            input_shape = self.input_details[0]['shape']

            if len(input_shape) == 3:
                # expected shape: (1, features, 1)
                X = X.reshape(1, X.shape[1], 1)

            elif len(input_shape) == 2:
                # expected shape: (1, features)
                X = X.reshape(1, X.shape[1])

            else:
                raise RuntimeError(f"[TCN] Unexpected input shape: {input_shape}")

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], X)
            self.interpreter.invoke()

            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            score = float(output.flatten()[0])

            logger.debug("[TCN] score=%.4f", score)

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            logger.error("[TCN] Inference error: %s", e, exc_info=True)
            return 0.0

    def is_anomaly(self, score: float) -> bool:
        return score >= TCN_ANOMALY_THRESHOLD

    def is_ready(self) -> bool:
        return self._loaded and self.interpreter is not None






















