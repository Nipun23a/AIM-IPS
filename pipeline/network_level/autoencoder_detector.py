"""
autoencoder_detector.py
────────────────────────
Drop-in replacement for tcn_detector.py.

Uses a trained deep autoencoder (TFLite FP32) to detect anomalous
network flows via reconstruction error.

How it works:
  • The autoencoder was trained on BENIGN flows only.
  • It reconstructs normal traffic with low MSE.
  • Attack/zero-day flows have high reconstruction MSE → flagged as anomalies.
  • The threshold is the 99th percentile of BENIGN validation MSE,
    saved to ae_threshold.pkl by train_autoencoder.py.

Public API (identical to TCNDetector so NetworkClassifier needs no changes):
    detector = AutoencoderDetector().load()
    score    = detector.predict(features)   # float 0–1 (normalised MSE)
    is_anom  = detector.is_anomaly(score)   # bool
    ready    = detector.is_ready()          # bool
"""

import logging
import numpy as np
import joblib
from pathlib import Path

from pipeline.network_level.feature import (
    THREAT_FEATURES,
    TCN_TFLITE_PATH,    # reuse same config keys — just point to autoencoder files
    TCN_SCALER_PATH,
    TCN_FEATURES_PATH,
    LABEL_ZERODAY,
    LABEL_CLEAN,
)

logger = logging.getLogger(__name__)

# Path to the threshold saved by train_autoencoder.py
# Falls back gracefully if missing (uses a raw MSE cutoff)
_AE_THRESHOLD_PATH = Path(
    "models/anomly_detector/network_level_attacks_anomality/ae_threshold.pkl"
)

# Hard fallback if ae_threshold.pkl is not present yet
_FALLBACK_THRESHOLD = 0.5


class AutoencoderDetector:
    """
    TFLite autoencoder-based anomaly detector.

    Anomaly score = reconstruction MSE normalised to [0, 1]:
        raw_mse  = mean((x - autoencoder(x))^2)
        score    = clip(raw_mse / threshold, 0, 2) / 2
                   → 0.5 exactly at threshold (decision boundary)
                   → approaches 1.0 for large deviations
    """

    def __init__(
        self,
        tflite_path:    Path = None,
        scaler_path:    Path = None,
        features_path:  Path = None,
        threshold_path: Path = None,
    ):
        self.tflite_path    = Path(tflite_path)    if tflite_path    else Path(TCN_TFLITE_PATH)
        self.scaler_path    = Path(scaler_path)    if scaler_path    else Path(TCN_SCALER_PATH)
        self.features_path  = Path(features_path)  if features_path  else Path(TCN_FEATURES_PATH)
        self.threshold_path = Path(threshold_path) if threshold_path else _AE_THRESHOLD_PATH

        self.interpreter    = None
        self.scaler         = None
        self.features: list = THREAT_FEATURES
        self.input_details  = None
        self.output_details = None
        self._threshold     = _FALLBACK_THRESHOLD
        self._loaded        = False

    # ─────────────────────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────────────────────

    def load(self) -> "AutoencoderDetector":
        import tensorflow as tf

        # TFLite model
        if not self.tflite_path.exists():
            raise FileNotFoundError(
                f"[AE] TFLite model not found: {self.tflite_path}\n"
                "Run train_autoencoder.py first."
            )

        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logger.info(
            "[AE] Loaded ← %s | input=%s dtype=%s",
            self.tflite_path,
            self.input_details[0]["shape"],
            self.input_details[0]["dtype"].__name__,
        )

        # Scaler
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            logger.info("[AE] Scaler loaded ← %s", self.scaler_path)
        else:
            logger.warning("[AE] Scaler not found at %s — using raw features", self.scaler_path)

        # Feature list
        if self.features_path.exists():
            info = joblib.load(self.features_path)
            self.features = (
                info.get("features", THREAT_FEATURES)
                if isinstance(info, dict) else info
            )
            logger.info("[AE] %d features loaded", len(self.features))
        else:
            logger.info("[AE] feature_info.pkl not found — using THREAT_FEATURES (%d)", len(self.features))

        # Anomaly threshold
        if self.threshold_path.exists():
            data = joblib.load(self.threshold_path)
            self._threshold = float(data["threshold"])
            logger.info(
                "[AE] Threshold = %.6f  (p%.0f of benign MSE)",
                self._threshold, data.get("percentile", 99),
            )
        else:
            logger.warning(
                "[AE] ae_threshold.pkl not found — using fallback threshold %.4f",
                _FALLBACK_THRESHOLD,
            )
            self._threshold = _FALLBACK_THRESHOLD

        self._loaded = True
        return self

    # ─────────────────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────────────────

    def predict(self, features: dict) -> float:
        """
        Run autoencoder inference on a flow feature dict.

        Returns:
            float in [0, 1]
            ~0.0  → normal traffic (low reconstruction error)
            ~0.5  → right at the decision boundary
            ~1.0  → highly anomalous (large reconstruction error)
        """
        if not self._loaded:
            return 0.0

        try:
            import pandas as pd

            X_df = pd.DataFrame(
                [[features.get(f, 0.0) for f in self.features]],
                columns=self.features,
            )

            # Scale
            if self.scaler is not None:
                X = self.scaler.transform(X_df).astype(np.float32)
            else:
                X = X_df.values.astype(np.float32)

            # Ensure shape is (1, n_features) — autoencoder is Dense-only
            X = X.reshape(1, -1)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]["index"], X)
            self.interpreter.invoke()
            recon = self.interpreter.get_tensor(self.output_details[0]["index"])

            # Reconstruction MSE for this single flow
            raw_mse = float(np.mean((X - recon) ** 2))

            # Normalise: score = 0.5 exactly at threshold
            #   score < 0.5  → below threshold → clean
            #   score > 0.5  → above threshold → anomaly
            score = float(np.clip(raw_mse / (2.0 * self._threshold), 0.0, 1.0))

            logger.debug("[AE] raw_mse=%.6f  score=%.4f  threshold=%.6f",
                         raw_mse, score, self._threshold)
            return score

        except Exception as e:
            logger.error("[AE] Inference error: %s", e, exc_info=True)
            return 0.0

    # ─────────────────────────────────────────────────────────
    # ANOMALY DECISION
    # ─────────────────────────────────────────────────────────

    def is_anomaly(self, score: float) -> bool:
        """Score >= 0.5 means reconstruction error exceeded the threshold."""
        return score >= 0.5

    # ─────────────────────────────────────────────────────────
    # READY CHECK
    # ─────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._loaded and self.interpreter is not None

    # ─────────────────────────────────────────────────────────
    # INFO
    # ─────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "model":     str(self.tflite_path),
            "threshold": self._threshold,
            "features":  len(self.features),
            "ready":     self.is_ready(),
        }