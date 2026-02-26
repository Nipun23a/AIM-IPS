import json
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

from shared.schemas import RequestContext,LayerScore
from shared.constants import LAYER_2_LGBM,LABEL_CLEAN,LABEL_UNKNOWN,NORMAL_LABELS


logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path("models/application_layer")

class LGBMAppClassifier:
    def __init__(self, model_dir: Path = _DEFAULT_MODEL_DIR):
        self.model_dir   = Path(model_dir)
        self.model       = None
        self.feature_names: list  = []
        self.class_to_idx: dict   = {}
        self.idx_to_class: dict   = {}
        self._loaded     = False
    
    def load(self) -> "LGBMAppClassifier":
        model_path   = self.model_dir / "threat_classifier_lgb.pkl"
        feature_path = self.model_dir / "feature_config.pkl"
        mapping_path = self.model_dir / "class_mapping.json"

        if not model_path.exists():
            raise FileNotFoundError(f"[LGBMApp] Model not found: {model_path}")
        self.model = joblib.load(model_path)
        logger.info(f"[LGBMApp] Loaded modeld <- {model_path}")

        if not feature_path.exists():
            raise FileNotFoundError(f"[LGBMApp] Feature config not found: {feature_path}")
        config = joblib.load(feature_path)
        self.feature_names = config["feature_names"]
        logger.info(f"[LGBMApp] {len(self.feature_names)} feature loaded")

        if not mapping_path.exists():
            raise FileNotFoundError(f"[LGBMApp] Class mapping not found: {mapping_path}")
        with open(mapping_path) as f:
            mapping = json.load(f)
        self.class_to_idx = mapping["class_to_idx"]
        self.idx_to_class = {int(k):v for k,v in mapping["idx_to_class"].items()}
        logger.info(f"[LGBMApp] Classes: {list(self.class_to_idx.keys())}")

        self._loaded = True
        return self
    
    def predict(self,ctx:RequestContext) -> LayerScore:
        if not self._loaded:
            logger.error("[LGBMApp] Model not loaded -class load() first")
            return LayerScore.clean(LAYER_2_LGBM)
        
        try:
            payload = self._build_payload(ctx)
            features = self._extract_features(payload)
            X = np.array(
                [[features.get(f,0.0) for f in self.feature_names]],
                dtype=np.float32
            )

            proba = self.model.predict(X)
            proba = np.array(proba).flatten()

            pred_idx = int(np.argmax(proba))
            pred_label = self.idx_to_class.get(pred_idx,LABEL_UNKNOWN)
            confidence = float(proba[pred_idx])

            normal_idx = self.class_to_idx.get("norm",self.class_to_idx.get("normal",self.class_to_idx.get("benign",None)))
            if normal_idx is not None and normal_idx < len(proba):
                threat_score = float(1.0-proba[normal_idx])
            else:
                threat_score = confidence if pred_label not in NORMAL_LABELS else 0.0
            
            is_attack = pred_label not in NORMAL_LABELS

            logger.debug(
                f"[LGBMApp] {ctx.ip} -> label = {pred_label}"
                f"score = {threat_score:.3f} conf = {confidence:.3f}"
            )

            return LayerScore(
                score = threat_score,
                label = pred_label,
                confidence= confidence,
                layer= LAYER_2_LGBM,
                triggered= False,
                metadata   = {
                    "all_probs":  {self.idx_to_class[i]: round(float(p), 4)
                                   for i, p in enumerate(proba)},
                    "payload_len": len(payload),
                }
            )
        except Exception as e:
            logger.error(f"[LGBMApp] Inference error for {ctx.ip}:{e}",exc_info=True)
            return LayerScore.clean(LAYER_2_LGBM)
        
    def _build_payload(self,ctx:RequestContext) -> str:
        parts = []
        if ctx.body:
            parts.append(ctx.body)
        if ctx.path and ctx.path != "/":
            parts.append(ctx.path)
        if ctx.query_params:
            qs = " ".join(f"{k}={v}" for k, v in ctx.query_params.items())
            parts.append(qs)
        return " ".join(parts)
    
    def _extract_features(self,payload:str) -> dict:
        try:
            from threat_classifier.src.application_level_threat_classifier.feature_engineering import (
                ThreatFeatureExtractor
            )
            extractor = ThreatFeatureExtractor()
            return extractor.extract_features(payload)
        except ImportError:
            logger.warning("[LGBMApp] ThreatFeatureExtractor not found - using fallback extractor")
            return self._fallback_extract(payload)
    
    def _fallback_extract(self,payload:str) -> dict:
        return {f:0.0 for f in self.feature_names}
    
    def is_ready(self) -> bool:
        return self._loaded and self.model is not None