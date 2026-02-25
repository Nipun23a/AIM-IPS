import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Optional
from scipy.spatial.distance import mahalanobis
import tensorflow as tf
import math
import re
from collections import Counter



from shared.schemas import LayerScore, RequestContext
from shared.constants import (
    LAYER_2_CNN, LABEL_ANOMALY, LABEL_ZERODAY, LABEL_CLEAN,
    WEIGHT_CNN_RECON, WEIGHT_CNN_MAHAL,
)


logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path("models/anomaly_detector/application_level_attacks")

ANOMALY_THRESHOLD = 0.5


class CNNAnomalyDetector:
    def __init__(self, model_dir: Path = _DEFAULT_MODEL_DIR):
        self.model_dir    = Path(model_dir)
        self.interpreter  = None
        self.scaler       = None
        self.features: list = []
        self.input_details  = None
        self.output_details = None
        self.pca          = None
        self.maha_mean    = None
        self.maha_inv_cov = None
        self.use_fusion   = False
        self._loaded = False
    
    def load(self) -> "CNNAnomalyDetector":
        tflite_path  = self.model_dir / "fusion_cnn.tflite"
        scaler_path  = self.model_dir / "scaler.pkl"
        features_path = self.model_dir / "fusion_features.pkl"

        if not tflite_path.exists():
            raise FileNotFoundError(f"[CNN] TFLite model not found: {tflite_path}")
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logger.info(
            f"[CNN] Loaded TFLite ← {tflite_path} | "
            f"input={self.input_details[0]['shape']}"
        )

        if not scaler_path.exists():
            raise FileNotFoundError(f"[CNN] Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        logger.info(f"[CNN] Loaded scaler ← {scaler_path}")

        if not features_path.exists():
            raise FileNotFoundError(f"[CNN] Features not found: {features_path}")
        self.features = joblib.load(features_path)
        logger.info(f"[CNN] {len(self.features)} features loaded")

        pca_path      = self.model_dir / "pca.pkl"
        mean_path     = self.model_dir / "maha_mean.pkl"
        inv_cov_path  = self.model_dir / "maha_inv_cov.pkl"

        if pca_path.exists() and mean_path.exists() and inv_cov_path.exists():
            self.pca          = joblib.load(pca_path)
            self.maha_mean    = joblib.load(mean_path)
            self.maha_inv_cov = joblib.load(inv_cov_path)
            self.use_fusion   = True
            logger.info("[CNN] PCA + Mahalanobis fusion gate enabled")
        else:
            logger.info("[CNN] Fusion gate not found — using reconstruction error only")

        self._loaded = True
        return self
    
    def predict(self,ctx: RequestContext)-> LayerScore:
        if not self._loaded:
            logger.error("[CNN] Model not loaded — call load() first")
            return LayerScore.clean(LAYER_2_CNN)
        
        try:
            payload  = self._build_payload(ctx)
            feat_vec = self._extract_features(payload)
            X_raw = np.array(
                [[feat_vec.get(f, 0.0) for f in self.features]],
                dtype=np.float32
            )
            X_scaled = self.scaler.transform(X_raw).astype(np.float32)
            recon_score = self._run_tflite(X_scaled)
            if self.use_fusion:
                maha_score  = self._mahalanobis_score(X_scaled)
                final_score = (
                    WEIGHT_CNN_RECON * recon_score +
                    WEIGHT_CNN_MAHAL * maha_score
                )
                final_score = float(np.clip(final_score, 0.0, 1.0))
            else:
                final_score = float(np.clip(recon_score, 0.0, 1.0))
                maha_score  = 0.0

            label = LABEL_ZERODAY if final_score > ANOMALY_THRESHOLD else LABEL_CLEAN

            logger.debug(
                f"[CNN] {ctx.ip} → score={final_score:.3f} "
                f"recon={recon_score:.3f} maha={maha_score:.3f}"
            )

            return LayerScore(
                score      = final_score,
                label      = label,
                confidence = final_score,   # autoencoder confidence = anomaly score itself
                layer      = LAYER_2_CNN,
                triggered  = False,
                metadata   = {
                    "recon_score": round(recon_score, 4),
                    "maha_score":  round(maha_score, 4),
                    "fusion_used": self.use_fusion,
                }
            )

        except Exception as e:
            logger.error(f"[CNN] Inference error for {ctx.ip}: {e}", exc_info=True)
            return LayerScore.clean(LAYER_2_CNN)
    
    def _run_tflite(self,X_scaled:np.ndarray) -> float:
        self.interpreter.set_tensor(self.input_details[0]['index'], X_scaled)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return float(output[0][0])
    
    def _mahalanobis_score(self,X_scaled:np.ndarray)-> float:
        try:
            X_pca  = self.pca.transform(X_scaled)
            dist   = mahalanobis(X_pca[0], self.maha_mean, self.maha_inv_cov)
            score  = float(1.0 - np.exp(-dist / 10.0))
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"[CNN] Mahalanobis failed: {e}")
            return 0.0
    
    def _build_payload(self,ctx:RequestContext) -> str:
        if ctx.body:
            return ctx.body
        
        if ctx.query_params:
            qs = "&".join(f"{k}={v}" for k, v in ctx.query_params.items())
            return f"{ctx.path}?{qs}"
        return ctx.path
    
    def _extract_features(self,payload:str) -> str :
        try:
            from anomly_detector.src.application_level_attacks_anomality.payload_features import (
                extract_payload_features
            )
            return extract_payload_features(payload)
        except ImportError:
            logger.warning("[CNN] extract_payload_features not found — using inline fallback")
            return self._inline_extract(payload)
        
    def _inline_extract(self,payload : str) -> dict:
        SQL_KW  = re.compile(r"(select|union|insert|drop|or\s+1=1)", re.I)
        XSS_KW  = re.compile(r"(<script|onerror=|alert\()", re.I)
        PATH_KW = re.compile(r"(\.\./|\.\.\\|/etc/passwd)", re.I)
        CMD_KW  = re.compile(r"(;|\|\||&&|\b(cat|ls|whoami)\b)", re.I)

        def shannon_entropy(s:str) -> float:
            if not s:
                return 0.0
            counts = Counter(s)
            probs = [c / len(s) for c in counts.values()]
            return -sum(p * math.log(p) for p in probs)
        
        payload = str(payload)

        return {
            "payload_len":        len(payload),
            "digit_count":        sum(c.isdigit() for c in payload),
            "alpha_count":        sum(c.isalpha() for c in payload),
            "special_char_count": sum(not c.isalnum() for c in payload),
            "slash_count":        payload.count("/"),
            "dot_count":          payload.count("."),
            "percent_count":      payload.count("%"),
            "equals_count":       payload.count("="),
            "question_count":     payload.count("?"),
            "amp_count":          payload.count("&"),
            "has_sql_kw":         int(bool(SQL_KW.search(payload))),
            "has_xss_kw":         int(bool(XSS_KW.search(payload))),
            "has_path_kw":        int(bool(PATH_KW.search(payload))),
            "has_cmd_kw":         int(bool(CMD_KW.search(payload))),
            "entropy":            shannon_entropy(payload),
        }
    
    def is_ready(self) -> bool:
        return self._loaded and self.interpreter is not None










