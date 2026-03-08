"""
pipeline/network_level/network_classifier.py
──────────────────────────────────────────────
Fuses LightGBM (supervised, known attacks) + EnsembleDetector
(semi-supervised anomaly detector) scores.

Stage 1 — LightGBM
    Multi-class: benign / ddos / portscan / bot
    Threat score = 1 - P(benign)

Stage 2 — EnsembleDetector  (AE + VAE + OCC + IsolationForest)
    AE + VAE : BENIGN-only → zero-day via distribution shift
    OCC      : semi-supervised → boundary from real attacks
    IsoForest: semi-supervised → real contamination ratio
    Output: 0.0 (normal) .. 0.5 (boundary) .. 1.0 (anomalous)

Fusion:
    fused = WEIGHT_NETWORK_LGBM * lgbm_score
          + WEIGHT_NETWORK_TCN  * ensemble_score

Attack type:
    lgbm_label != benign         -> known attack (ddos / portscan / bot)
    lgbm benign + ens anomaly    -> ZeroDay
    both clean                   -> clean
"""

import logging
import time
import numpy as np
from pathlib import Path

from shared.constants import WEIGHT_NETWORK_LGBM, WEIGHT_NETWORK_TCN
from pipeline.network_level.feature import (
    LABEL_BENIGN, LABEL_ZERODAY, LABEL_CLEAN,
    LGBM_MODEL_PATH, LGBM_FEATURES_PATH,
)
from pipeline.network_level.lgbm_network_classifier import LGBMNetworkClassifier
from pipeline.network_level.ensemble_detector import EnsembleDetector

logger = logging.getLogger(__name__)

_BENIGN_LABELS = {"BENIGN", "benign", "normal", "Normal"}


class NetworkClassifier:
    """
    Two-stage network threat classifier.

    Backwards-compatible result dict — all existing keys are preserved:
      fused_score, lgbm_score, lgbm_label, lgbm_probs,
      attack_type, is_threat, redis_written, timestamp
      ensemble_score  <- new primary key
      tcn_score       <- legacy alias for ensemble_score
      ae_score        <- legacy alias for ensemble_score
    """

    def __init__(
        self,
        lgbm_model_path:    Path = None,
        lgbm_features_path: Path = None,
        # Legacy kwargs — accepted but ignored; ensemble paths come from feature.py
        tcn_tflite_path:    Path = None,
        tcn_scaler_path:    Path = None,
        tcn_features_path:  Path = None,
    ):
        self.lgbm = LGBMNetworkClassifier(
            model_path    = lgbm_model_path    or Path(LGBM_MODEL_PATH),
            features_path = lgbm_features_path or Path(LGBM_FEATURES_PATH),
        )
        self.ensemble = EnsembleDetector()

        # Backwards-compat aliases
        self.ae  = self.ensemble
        self.tcn = self.ensemble

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
        lgbm_ok = ensemble_ok = False

        try:
            self.lgbm.load()
            lgbm_ok = True
            logger.info("[NetClassifier] LightGBM loaded ✓")
            logger.info("[NetClassifier]   path=%s",    self.lgbm.model_path)
            logger.info("[NetClassifier]   classes=%s", self.lgbm.classes)
        except FileNotFoundError as e:
            logger.error("[NetClassifier] LightGBM load failed: %s", e)

        try:
            self.ensemble.load()
            ensemble_ok = True
            logger.info("[NetClassifier] Ensemble detector loaded ✓")
            logger.info("[NetClassifier]   models=%s",    self.ensemble.model_summary())
            logger.info("[NetClassifier]   threshold=%.6f", self.ensemble._raw_thr or 0.0)
        except FileNotFoundError as e:
            logger.error("[NetClassifier] Ensemble not found: %s", e)
        except Exception as e:
            logger.error("[NetClassifier] Ensemble load failed: %s", e, exc_info=True)

        if not lgbm_ok and not ensemble_ok:
            logger.error("[NetClassifier] No models loaded — scoring disabled")

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
            lgbm_label, _, lgbm_probs = self.lgbm.predict(features)
            lgbm_score = self.lgbm.threat_score(features)
        else:
            logger.debug("[NetClassifier] LightGBM not ready")

        # Stage 2 — Ensemble
        ensemble_score = 0.0
        if self.ensemble.is_ready():
            ensemble_score = self.ensemble.predict(features)
        else:
            logger.debug("[NetClassifier] Ensemble not ready")

        # Fusion
        if self.lgbm.is_ready() and self.ensemble.is_ready():
            fused = (WEIGHT_NETWORK_LGBM * lgbm_score
                   + WEIGHT_NETWORK_TCN  * ensemble_score)
        elif self.lgbm.is_ready():
            fused = lgbm_score
        elif self.ensemble.is_ready():
            fused = ensemble_score
        else:
            fused = 0.0

        fused = float(np.clip(fused, 0.0, 1.0))

        # Attack type
        if lgbm_label not in _BENIGN_LABELS:
            attack_type = lgbm_label                        # ddos / portscan / bot
        elif self.ensemble.is_anomaly(ensemble_score):
            attack_type = LABEL_ZERODAY
        else:
            attack_type = LABEL_CLEAN

        is_threat = fused >= 0.35
        if is_threat:
            self.threat_flows += 1

        written = self._write_redis(
            ip             = src_ip,
            score          = fused,
            lgbm_score     = lgbm_score,
            ensemble_score = ensemble_score,
            attack_type    = attack_type,
            confidence     = max(lgbm_score, ensemble_score),
        )

        if is_threat:
            logger.warning(
                "[NetClassifier] THREAT %s -> %s  "
                "(fused=%.3f lgbm=%.3f ensemble=%.3f)",
                src_ip, attack_type, fused, lgbm_score, ensemble_score,
            )
        else:
            logger.debug("[NetClassifier] CLEAN %s (fused=%.3f)", src_ip, fused)

        return {
            "src_ip":         src_ip,
            "fused_score":    round(fused,          4),
            "lgbm_score":     round(lgbm_score,     4),
            "ensemble_score": round(ensemble_score, 4),
            # ── Legacy aliases ─────────────────────────────────
            "tcn_score":      round(ensemble_score, 4),
            "ae_score":       round(ensemble_score, 4),
            # ───────────────────────────────────────────────────
            "lgbm_label":     lgbm_label,
            "lgbm_probs":     lgbm_probs,
            "attack_type":    attack_type,
            "is_threat":      is_threat,
            "redis_written":  written,
            "timestamp":      time.time(),
        }

    # ─────────────────────────────────────────────────────────
    # REDIS WRITE
    # ─────────────────────────────────────────────────────────

    def _write_redis(
        self, ip: str, score: float, lgbm_score: float,
        ensemble_score: float, attack_type: str, confidence: float,
    ) -> bool:
        r = self.redis
        if r is None:
            return False
        try:
            r.set_network_threat_score(
                ip          = ip,
                score       = score,
                net_lgbm    = lgbm_score,
                tcn         = ensemble_score,   # Redis field kept for schema compat
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
            "lgbm_ready":     self.lgbm.is_ready(),
            "ensemble_ready": self.ensemble.is_ready(),
            # Legacy keys — kept for network_ips.py status checks
            "tcn_ready":      self.ensemble.is_ready(),
            "ae_ready":       self.ensemble.is_ready(),
            "total_flows":    self.total_flows,
            "threat_flows":   self.threat_flows,
            "redis_writes":   self.redis_writes,
            "redis_ok":       redis_ok,
        }