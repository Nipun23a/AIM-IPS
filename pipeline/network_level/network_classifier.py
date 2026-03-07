"""
network_classifier.py
──────────────────────
Fuses LightGBM (multi-class) + Autoencoder (anomaly) scores.

LightGBM  → classifies known attacks (benign / ddos / portscan)
Autoencoder → catches zero-day / novel attacks via reconstruction error
              trained on BENIGN-only traffic, no attack labels needed
"""

import logging
import time
import numpy as np
from pathlib import Path

from shared.constants import WEIGHT_NETWORK_LGBM, WEIGHT_NETWORK_TCN
from pipeline.network_level.feature import (
    LABEL_BENIGN, LABEL_ZERODAY, LABEL_CLEAN,
    LGBM_MODEL_PATH, LGBM_FEATURES_PATH,
    TCN_TFLITE_PATH, TCN_SCALER_PATH, TCN_FEATURES_PATH,
)
from pipeline.network_level.lgbm_network_classifier import LGBMNetworkClassifier
from pipeline.network_level.autoencoder_detector import AutoencoderDetector

logger = logging.getLogger(__name__)

_BENIGN_LABELS = {"BENIGN", "benign", "normal", "Normal"}


class NetworkClassifier:
    """
    Two-stage network threat classifier:

    Stage 1 — LightGBM (known attack taxonomy)
        Output: label ∈ {benign, ddos, portscan} + per-class probabilities
        Threat score = 1 - P(benign)

    Stage 2 — Autoencoder (zero-day / novel anomaly)
        Trained on BENIGN flows only.
        Score = normalised reconstruction MSE:
            < 0.5 → clean  |  ≥ 0.5 → anomalous

    Fusion:
        fused = WEIGHT_NETWORK_LGBM × lgbm_score
              + WEIGHT_NETWORK_TCN  × ae_score

    Attack type decision:
        lgbm label ≠ benign → use lgbm label  (known attack)
        lgbm benign but ae anomaly → "ZeroDay"
        both clean → "clean"
    """

    def __init__(
        self,
        lgbm_model_path:    Path = None,
        lgbm_features_path: Path = None,
        tcn_tflite_path:    Path = None,  # kept for API compat — points to AE model
        tcn_scaler_path:    Path = None,
        tcn_features_path:  Path = None,
    ):
        self.lgbm = LGBMNetworkClassifier(
            model_path    = lgbm_model_path    or Path(LGBM_MODEL_PATH),
            features_path = lgbm_features_path or Path(LGBM_FEATURES_PATH),
        )
        # AutoencoderDetector uses the same path constants as TCNDetector
        # (TCN_TFLITE_PATH etc. should now point to autoencoder_fp32.tflite)
        self.ae = AutoencoderDetector(
            tflite_path   = tcn_tflite_path   or Path(TCN_TFLITE_PATH),
            scaler_path   = tcn_scaler_path   or Path(TCN_SCALER_PATH),
            features_path = tcn_features_path or Path(TCN_FEATURES_PATH),
        )

        # Backwards-compat alias so test code using .tcn still works
        self.tcn = self.ae

        self._redis       = None
        self.total_flows  = 0
        self.threat_flows = 0
        self.redis_writes = 0

    # ─────────────────────────────────────────────────────────
    # REDIS  (lazy)
    # ─────────────────────────────────────────────────────────

    @property
    def redis(self):
        if self._redis is None:
            try:
                from utils.redis_client import get_redis
                self._redis = get_redis()
            except ImportError:
                try:
                    from utils.redis_client import RedisClient
                    self._redis = RedisClient.get_instance()
                except Exception as e:
                    logger.warning("[NetClassifier] Redis unavailable: %s", e)
        return self._redis

    # ─────────────────────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────────────────────

    def load(self) -> "NetworkClassifier":
        lgbm_ok = False
        ae_ok   = False

        try:
            self.lgbm.load()
            lgbm_ok = True
            logger.info("[NetClassifier] LightGBM loaded ✓")
            logger.info("[NetClassifier]   path=%s", self.lgbm.model_path)
            logger.info("[NetClassifier]   classes=%s", self.lgbm.classes)
        except FileNotFoundError as e:
            logger.error("[NetClassifier] LightGBM load failed: %s", e, exc_info=True)

        try:
            self.ae.load()
            ae_ok = True
            logger.info("[NetClassifier] Autoencoder loaded ✓")
            logger.info("[NetClassifier]   path=%s",      self.ae.tflite_path)
            logger.info("[NetClassifier]   threshold=%.6f", self.ae._threshold)
            logger.info("[NetClassifier]   features=%d",  len(self.ae.features))
        except FileNotFoundError as e:
            logger.error("[NetClassifier] Autoencoder not found: %s", e)
        except Exception as e:
            logger.error("[NetClassifier] Autoencoder load failed: %s", e, exc_info=True)

        if not lgbm_ok and not ae_ok:
            logger.error("[NetClassifier] No models loaded — network scoring disabled")

        return self

    # ─────────────────────────────────────────────────────────
    # CLASSIFY
    # ─────────────────────────────────────────────────────────

    def classify_flow(self, src_ip: str, features: dict) -> dict:
        self.total_flows += 1

        # Stage 1 — LightGBM
        lgbm_label = LABEL_BENIGN
        lgbm_score = 0.0
        lgbm_probs = {}

        if self.lgbm.is_ready():
            lgbm_label, lgbm_conf, lgbm_probs = self.lgbm.predict(features)
            lgbm_score = self.lgbm.threat_score(features)
        else:
            logger.debug("[NetClassifier] LightGBM not ready")

        # Stage 2 — Autoencoder
        ae_score = 0.0
        if self.ae.is_ready():
            ae_score = self.ae.predict(features)
        else:
            logger.debug("[NetClassifier] Autoencoder not ready")

        # Fusion
        if self.lgbm.is_ready() and self.ae.is_ready():
            fused = WEIGHT_NETWORK_LGBM * lgbm_score + WEIGHT_NETWORK_TCN * ae_score
        elif self.lgbm.is_ready():
            fused = lgbm_score
        elif self.ae.is_ready():
            fused = ae_score
        else:
            fused = 0.0

        fused = float(np.clip(fused, 0.0, 1.0))

        # Attack type
        if lgbm_label not in _BENIGN_LABELS:
            attack_type = lgbm_label                       # known: ddos, portscan, …
        elif self.ae.is_anomaly(ae_score):
            attack_type = LABEL_ZERODAY                    # novel / zero-day
        else:
            attack_type = LABEL_CLEAN

        is_threat = fused >= 0.35
        if is_threat:
            self.threat_flows += 1

        written = self._write_redis(
            ip          = src_ip,
            score       = fused,
            lgbm_score  = lgbm_score,
            tcn_score   = ae_score,        # field name kept for Redis schema compat
            attack_type = attack_type,
            confidence  = max(lgbm_score, ae_score),
        )

        if is_threat:
            logger.warning(
                "[NetClassifier] THREAT %s → %s (fused=%.3f lgbm=%.3f ae=%.3f)",
                src_ip, attack_type, fused, lgbm_score, ae_score,
            )
        else:
            logger.debug("[NetClassifier] CLEAN %s (fused=%.3f)", src_ip, fused)

        return {
            "src_ip":        src_ip,
            "fused_score":   round(fused,      4),
            "lgbm_score":    round(lgbm_score, 4),
            "tcn_score":     round(ae_score,   4),   # kept for compat
            "ae_score":      round(ae_score,   4),
            "lgbm_label":    lgbm_label,
            "lgbm_probs":    lgbm_probs,
            "attack_type":   attack_type,
            "is_threat":     is_threat,
            "redis_written": written,
            "timestamp":     time.time(),
        }

    # ─────────────────────────────────────────────────────────
    # REDIS WRITE
    # ─────────────────────────────────────────────────────────

    def _write_redis(
        self, ip: str, score: float, lgbm_score: float,
        tcn_score: float, attack_type: str, confidence: float,
    ) -> bool:
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
            logger.warning("[NetClassifier] Redis write failed for %s: %s", ip, e)
            return False

    # ─────────────────────────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        redis_ok = False
        try:
            redis_ok = self.redis is not None and self.redis.ping()
        except Exception:
            pass
        return {
            "lgbm_ready":   self.lgbm.is_ready(),
            "tcn_ready":    self.ae.is_ready(),    # compat key
            "ae_ready":     self.ae.is_ready(),
            "total_flows":  self.total_flows,
            "threat_flows": self.threat_flows,
            "redis_writes": self.redis_writes,
            "redis_ok":     redis_ok,
        }