import logging
import time
import numpy as np
from pathlib import Path


from shared.constants import WEIGHT_NETWORK_LGBM, WEIGHT_NETWORK_TCN
from pipeline.network_level.feature import (
    LABEL_BENIGN,LABEL_ZERODAY,LABEL_CLEAN,
    LGBM_MODEL_PATH,LGBM_FEATURES_PATH,TCN_TFLITE_PATH,TCN_SCALER_PATH,TCN_FEATURES_PATH
)
from pipeline.network_level.lgbm_network_classifier import LGBMNetworkClassifier
from pipeline.network_level.tcn_detector import TCNDetector

logger = logging.getLogger(__name__)


_BENIGN_LABELS = {"BENIGN", "benign", "normal", "Normal"}

class NetworkClassifier:
    def __init__(
        self,
        lgbm_model_path:    Path = None,
        lgbm_features_path: Path = None,
        tcn_tflite_path:    Path = None,
        tcn_scaler_path:    Path = None,
        tcn_features_path:  Path = None,
    ):
        self.lgbm = LGBMNetworkClassifier(
            model_path    = lgbm_model_path    or Path(LGBM_MODEL_PATH),
            features_path = lgbm_features_path or Path(LGBM_FEATURES_PATH)
        )
        self.tcn = TCNDetector(
            tflite_path   = tcn_tflite_path   or Path(TCN_TFLITE_PATH),
            scaler_path   = tcn_scaler_path   or Path(TCN_SCALER_PATH),
            features_path = tcn_features_path or Path(TCN_FEATURES_PATH),
        )

        self._redis = None
        self.total_flows  = 0
        self.threat_flows = 0
        self.redis_writes = 0
    
    @property
    def redis(self):
        """Lazy Redis connection — doesn't fail if Redis is down."""
        if self._redis is None:
            try:
                from utils.redis_client import get_redis
                self._redis = get_redis()
            except ImportError:
                try:
                    from utils.redis_client import RedisClient
                    self._redis = RedisClient.get_instance()
                except Exception as e:
                    logger.warning(f"[NetClassifier] Redis unavailable: {e}")
        return self._redis
    
    def load(self) -> "NetworkClassifier":
        lgbm_ok = False
        tcn_ok = False

        try:
            self.lgbm.load()
            lgbm_ok = True
            logger.info("[NetClassifier] LightGBM loaded ✓")
            logger.info(f"[NetClassifier]   path={self.lgbm.model_path}")
            logger.info(f"[NetClassifier]   classes={self.lgbm.classes}")
        except FileNotFoundError as e:
            logger.error(f"[NetClassifier] LightGBM load failed: {e}", exc_info=True)

        try:
            self.tcn.load()
            tcn_ok = True
            logger.info("[NetClassifier] TCN loaded ✓")
            logger.info(f"[NetClassifier]   path={self.tcn.tflite_path}")
            logger.info(f"[NetClassifier]   features={len(self.tcn.features)}")
        except FileNotFoundError as e:
            logger.error(f"[NetClassifier] TCN not found: {e}")
        except Exception as e:
            logger.error(f"[NetClassifier] TCN load failed: {e}", exc_info=True)

        if not lgbm_ok and not tcn_ok:
            logger.error("[NetClassifier] No models loaded — network scoring disabled")

        return self
    
    def classify_flow(self,src_ip:str,features:dict) -> dict:
        self.total_flows += 1
        lgbm_label  = LABEL_BENIGN
        lgbm_score  = 0.0
        lgbm_probs  = {}

        if self.lgbm.is_ready():
            lgbm_label, lgbm_conf, lgbm_probs = self.lgbm.predict(features)
            lgbm_score = self.lgbm.threat_score(features)
        else:
            logger.debug("[NetClassifier] LightGBM not ready")

        tcn_score = 0.0
        if self.tcn.is_ready():
            tcn_score = self.tcn.predict(features)
        else:
            logger.debug("[NetClassifier] TCN not ready")
        
        if self.lgbm.is_ready() and self.tcn.is_ready():
            fused = WEIGHT_NETWORK_LGBM * lgbm_score + WEIGHT_NETWORK_TCN * tcn_score
        elif self.lgbm.is_ready():
            fused = lgbm_score
        elif self.tcn.is_ready():
            fused = tcn_score
        else:
            fused = 0.0

        fused = float(np.clip(fused, 0.0, 1.0)) if _has_numpy() else min(1.0, max(0.0, fused))


        if lgbm_label not in _BENIGN_LABELS:
            attack_type = lgbm_label   # e.g. "DDoS", "PortScan", "Bot"
        elif self.tcn.is_anomaly(tcn_score):
            attack_type = LABEL_ZERODAY
        else:
            attack_type = LABEL_CLEAN

        is_threat = fused >= 0.35
        if is_threat:
            self.threat_flows += 1

        written = self._write_redis(
            ip          = src_ip,
            score       = fused,
            lgbm_score  = lgbm_score,
            tcn_score   = tcn_score,
            attack_type = attack_type,
            confidence  = max(lgbm_score, tcn_score),
        )

        if is_threat:
            logger.warning(
                "[NetClassifier] THREAT %s → %s "
                "(fused=%.3f lgbm=%.3f tcn=%.3f)",
                src_ip, attack_type, fused, lgbm_score, tcn_score
            )
        else:
            logger.debug("[NetClassifier] CLEAN %s (fused=%.3f)", src_ip, fused)

        return {
            "src_ip":        src_ip,
            "fused_score":   round(fused, 4),
            "lgbm_score":    round(lgbm_score, 4),
            "tcn_score":     round(tcn_score, 4),
            "lgbm_label":    lgbm_label,
            "lgbm_probs":    lgbm_probs,
            "attack_type":   attack_type,
            "is_threat":     is_threat,
            "redis_written": written,
            "timestamp":     time.time(),
        }
    

    def _write_redis(
        self, ip: str, score: float, lgbm_score: float,
        tcn_score: float, attack_type: str, confidence: float,
    ) -> bool:
        """Write threat score to Redis. Returns True on success."""
        r = self.redis
        if r is None:
            return False
        try:
            r.set_network_threat_score(
                ip          = ip,
                score       = score,
                net_lgbm    = lgbm_score,
                tcn         = tcn_score,
                attack_type = attack_type,
                confidence  = confidence,
            )
            self.redis_writes += 1
            return True
        except Exception as e:
            logger.warning(f"[NetClassifier] Redis write failed for {ip}: {e}")
            return False

    def status(self) -> dict:
        redis_ok = False
        try:
            redis_ok = self.redis is not None and self.redis.ping()
        except Exception:
            pass
        return {
            "lgbm_ready":   self.lgbm.is_ready(),
            "tcn_ready":    self.tcn.is_ready(),
            "total_flows":  self.total_flows,
            "threat_flows": self.threat_flows,
            "redis_writes": self.redis_writes,
            "redis_ok":     redis_ok,
        }


def _has_numpy() -> bool:
    try:
        import numpy  # noqa
        return True
    except ImportError:
        return False








