import logging
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from pipeline.network_level.feature import (
    THREAT_FEATURES,
    LABEL_BENIGN,
    LGBM_MODEL_PATH,
    LGBM_FEATURES_PATH,
)

logger = logging.getLogger(__name__)


class LGBMNetworkClassifier:
    """
    Wrapper for the trained LightGBM network classifier.
    Handles feature ordering and label mapping safely.
    """

    def __init__(self, model_path: Path = None, features_path: Path = None):

        self.model_path = Path(model_path) if model_path else Path(LGBM_MODEL_PATH)
        self.features_path = Path(features_path) if features_path else Path(LGBM_FEATURES_PATH)

        self.model = None
        self.features = THREAT_FEATURES
        self.classes = []
        self._loaded = False

    # ----------------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------------

    def load(self) -> "LGBMNetworkClassifier":
        if not self.model_path.exists():
            raise FileNotFoundError(f"[NetLGBM] Model not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        logger.info(f"[NetLGBM] Model loaded ← {self.model_path}")

        # Load feature list used during training
        if self.features_path.exists():
            saved = joblib.load(self.features_path)

            if isinstance(saved, list):
                self.features = saved
            elif isinstance(saved, dict) and "features" in saved:
                self.features = saved["features"]

            logger.info(f"[NetLGBM] Loaded {len(self.features)} features")

        else:
            logger.warning("[NetLGBM] features.pkl not found — using THREAT_FEATURES")

        # -------------------------------------------------------
        # FIXED CLASS LABEL MAPPING
        # -------------------------------------------------------
        if hasattr(self.model, "classes_"):

            idx_to_label = {
                0: "benign",
                1: "ddos",
                2: "portscan"
            }

            self.classes = [
                idx_to_label.get(int(c), str(c))
                for c in self.model.classes_
            ]

        else:
            self.classes = ["benign", "ddos", "portscan"]

        logger.info(f"[NetLGBM] Classes: {self.classes}")

        self._loaded = True
        return self

    # ----------------------------------------------------------
    # VECTORIZE FEATURES
    # ----------------------------------------------------------

    def _vectorize(self, features: dict) -> pd.DataFrame:
        """
        Convert feature dictionary → DataFrame with correct order.
        """

        vec = [float(features.get(f, 0.0)) for f in self.features]

        return pd.DataFrame([vec], columns=self.features)

    # ----------------------------------------------------------
    # PREDICT
    # ----------------------------------------------------------

    def predict(self, features: dict) -> Tuple[str, float, dict]:

        if not self._loaded:
            logger.warning("[NetLGBM] Model not loaded")
            return LABEL_BENIGN, 0.0, {}

        try:

            X = self._vectorize(features)

            probs = self.model.predict_proba(X)[0]

            pred_idx = int(np.argmax(probs))
            pred_label = self.classes[pred_idx]

            confidence = float(probs[pred_idx])

            all_probs = {
                self.classes[i]: float(probs[i])
                for i in range(len(self.classes))
            }

            return pred_label, confidence, all_probs

        except Exception as e:
            logger.error(f"[NetLGBM] Prediction failed: {e}", exc_info=True)
            return LABEL_BENIGN, 0.0, {}

    # ----------------------------------------------------------
    # THREAT SCORE
    # ----------------------------------------------------------

    def threat_score(self, features: dict) -> float:
        """
        Returns threat score between 0–1.
        Score = 1 - P(benign)
        """

        _, _, probs = self.predict(features)

        if not probs:
            return 0.0

        benign_prob = probs.get("benign", 0.0)

        return float(np.clip(1.0 - benign_prob, 0.0, 1.0))

    # ----------------------------------------------------------
    # READY CHECK
    # ----------------------------------------------------------

    def is_ready(self) -> bool:
        return self._loaded and self.model is not None