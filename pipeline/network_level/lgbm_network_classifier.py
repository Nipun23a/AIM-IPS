import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple,Dict

from pipeline.network_level.feature import (
    THREAT_FEATURES,LABEL_BENIGN,LABEL_CLEAN,LGBM_MODEL_PATH,LGBM_FEATURES_PATH
)

logger = logging.getLogger(__name__)

class LGBMNetworkClassifier:
    def __init__(self,model_path : Path = None, feature_path : Path = None):
        self.model_path = Path(model_path) if model_path else Path(LGBM_MODEL_PATH)
        self.feature_path = Path(feature_path) if feature_path else Path(LGBM_FEATURES_PATH)

        self.model = None
        self.features: list = THREAT_FEATURES
        self.idx_to_class : Dict[int,str] = {}
        self._loaded = False

    def load(self) -> "LGBMNetworkClassifier":
        if not self.model_path.exists():
            raise FileNotFoundError(f"[NetLGBM] Model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info(f"[NetLGBM] Loaded <- {self.model_path}")

        if self.feature_path.exists():
            saved = joblib.load(self.feature_path)
            if isinstance(saved,list):
                self.features = saved
            elif isinstance(saved,dict):
                self.features = saved.get("features",THREAT_FEATURES)
            logger.info(f"[NetLGBM] {len(self.features)} features loaded")
        else:
            logger.warning(f"[NetLGBM] features.pkl not found — using THREAT_FEATURES")

        self.idx_to_class = self._load_label_map()
        logger.info(f"[NetLGBM] Classes: {list(self.idx_to_class.values())}")

        self._loaded = True
        return self
    
    def _load_label_map(self) -> Dict[int,str] :
        try:
            from threat_classifier.src.network_level_threat_classifier.labels import LABEL_MAP
            return {int(v):k for k,v in LABEL_MAP.items()}
        except ImportError:
            pass

        if self.model is not None and hasattr(self.model,"classes_"):
            return {int(i) : str(c) for i,c in enumerate(self.model.classes_)}
        
        logger.warning("[NetLGBM] Could not load LABEL_MAP — using fallback")
        return {
            0: "BENIGN",
            1: "DDoS",
            2: "PortScan",
            3: "Bot",
            4: "DoS Hulk",
            5: "DoS GoldenEye",
            6: "FTP-Patator",
            7: "SSH-Patator",
            8: "DoS slowloris",
            9: "DoS Slowhttptest",
            10: "Web Attack",
            11: "Infiltration",
            12: "Heartbleed",
        }
    
    def predict(self,features: dict) -> Tuple[str,float,dict]:
        if not self._loaded:
            return LABEL_BENIGN, 0.0, {}

        try:
            X = np.array(
                [[features.get(f, 0.0) for f in self.features]],
                dtype=np.float32
            )

            # LGBMClassifier.predict_proba returns shape (1, n_classes)
            proba      = self.model.predict_proba(X).flatten()
            pred_idx   = int(np.argmax(proba))
            pred_label = self.idx_to_class.get(pred_idx, f"class_{pred_idx}")
            confidence = float(proba[pred_idx])

            all_probs = {
                self.idx_to_class.get(i, f"class_{i}"): round(float(p), 4)
                for i, p in enumerate(proba)
            }

            return pred_label, confidence, all_probs

        except Exception as e:
            logger.error(f"[NetLGBM] Inference error: {e}", exc_info=True)
            return LABEL_BENIGN, 0.0, {}
        
    def threat_score(self, features: dict) -> float:

        _, _, all_probs = self.predict(features)
        if not all_probs:
            return 0.0

        benign_prob = 0.0
        for key in ("BENIGN", "benign", "normal", "Benign"):
            if key in all_probs:
                benign_prob = all_probs[key]
                break

        return float(np.clip(1.0 - benign_prob, 0.0, 1.0))

    def is_ready(self) -> bool:
        return self._loaded and self.model is not None






















