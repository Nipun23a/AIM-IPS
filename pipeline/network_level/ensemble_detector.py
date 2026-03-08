"""
pipeline/network_level/ensemble_detector.py
─────────────────────────────────────────────
Semi-supervised Ensemble Anomaly Detector — inference wrapper.

Loads the four models trained by train_ensemble_detector.py and provides
a single predict() interface that NetworkClassifier calls.

Architecture
────────────
  AE  (TFLite)  — BENIGN-only training  → reconstruction MSE (zero-day)
  VAE (TFLite)  — BENIGN-only training  → ELBO loss          (zero-day)
  OCC (TFLite)  — semi-supervised       → 1 - P(normal)      (known + zero-day)
  IsoForest     — semi-supervised       → inverted depth      (tabular outliers)

  Ensemble score = Σ w_i * normalise(score_i)
  Weights learned by SLSQP AUC maximisation with W_MAX=0.40 per model
  (veto-safe: no single model can silence all others).

  Decision boundary: score >= 0.5 → is_anomaly() = True
  (raw threshold mapped so threshold → 0.5 in output space)

Graceful degradation:
  Missing model files are skipped; remaining weights are renormalised.
  All models missing → predict() returns 0.0 (fail-open / conservative).
"""

import logging
import numpy as np
import joblib
from pathlib import Path

from pipeline.network_level.feature import (
    THREAT_FEATURES,
    ENSEMBLE_SCALER_PATH,
    ENSEMBLE_THRESHOLD_PATH,
    ENSEMBLE_AE_TFLITE_PATH,
    ENSEMBLE_VAE_TFLITE_PATH,
    ENSEMBLE_OCC_TFLITE_PATH,
    ENSEMBLE_ISO_PATH,
    ENSEMBLE_WEIGHTS_PATH,
    ENSEMBLE_SCORE_SCALERS_PATH,
    ENSEMBLE_FEAT_PATH,
)

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    Inference wrapper for the AE + VAE + OCC + IsolationForest ensemble.

    Drop-in replacement for the old TCNDetector — identical public API:

        det = EnsembleDetector().load()
        score   = det.predict(features_dict)   # float 0-1
        is_anom = det.is_anomaly(score)         # score >= 0.5
        ready   = det.is_ready()
    """

    def __init__(self):
        self._ae_interp  = None
        self._vae_interp = None
        self._occ_interp = None
        self._iso        = None

        self._scaler        = None
        self._clip_lo       = None
        self._clip_hi       = None

        self._weights       = None   # np.ndarray shape (n_active,)
        self._weight_names  = None   # list[str]
        self._score_scalers = {}     # {name: (min, max)}
        self._raw_thr       = None   # optimal-F1 threshold in fused-score space
        self.features: list = THREAT_FEATURES
        self._loaded        = False

    # ─────────────────────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────────────────────

    def load(self) -> "EnsembleDetector":
        import tensorflow as tf

        loaded_models = []

        # 1. Scaler + clip bounds
        sp = Path(ENSEMBLE_SCALER_PATH)
        if not sp.exists():
            raise FileNotFoundError(
                f"[Ensemble] Scaler not found: {sp}\n"
                "Run train_ensemble_detector.py first."
            )
        sd = joblib.load(sp)
        self._scaler  = sd["scaler"]
        self._clip_lo = sd["clip_lo"]
        self._clip_hi = sd["clip_hi"]
        logger.info("[Ensemble] Scaler loaded <- %s", sp)

        # 2. Threshold
        tp = Path(ENSEMBLE_THRESHOLD_PATH)
        if tp.exists():
            td = joblib.load(tp)
            self._raw_thr = float(td["threshold"])
            logger.info("[Ensemble] Threshold=%.6f (method=%s)",
                        self._raw_thr, td.get("method", "unknown"))
        else:
            self._raw_thr = None
            logger.warning("[Ensemble] Threshold file missing — will use 0.5")

        # 3. Score scalers (per-model min/max)
        ssp = Path(ENSEMBLE_SCORE_SCALERS_PATH)
        if ssp.exists():
            self._score_scalers = joblib.load(ssp)

        # 4. Ensemble weights
        wp = Path(ENSEMBLE_WEIGHTS_PATH)
        if wp.exists():
            wd = joblib.load(wp)
            self._weights      = np.array(wd["weights"], dtype=np.float32)
            self._weight_names = wd["names"]
            if "score_scalers" in wd and not self._score_scalers:
                self._score_scalers = wd["score_scalers"]
            logger.info("[Ensemble] Weights: %s",
                " | ".join(f"{n}={w:.3f}" for n, w in
                           zip(self._weight_names, self._weights)))
        else:
            logger.warning("[Ensemble] ensemble_weights.pkl missing — equal weights")
            self._weight_names = ["AE", "VAE", "OCC", "IsoForest"]
            self._weights      = np.ones(4, dtype=np.float32) / 4

        # 5. Feature list
        fp = Path(ENSEMBLE_FEAT_PATH)
        if fp.exists():
            info = joblib.load(fp)
            self.features = (info.get("features", THREAT_FEATURES)
                             if isinstance(info, dict) else info)

        # 6. TFLite models
        def _load_tflite(path_str, name):
            p = Path(path_str)
            if not p.exists():
                logger.warning("[Ensemble] %s not found: %s", name, p)
                return None
            try:
                # Disable XNNPack delegate for VAE — its dynamic batch/feature
                # shape causes "failed to reshape runtimeNode" on some builds.
                # AE and OCC have static shapes and work fine with XNNPack.
                if name == "VAE":
                    interp = tf.lite.Interpreter(
                        model_path=str(p),
                        experimental_delegates=[],   # CPU-only, no XNNPack
                        num_threads=1,
                    )
                else:
                    interp = tf.lite.Interpreter(model_path=str(p))
                interp.allocate_tensors()
                logger.info("[Ensemble] %s loaded <- %s", name, p)
                return interp
            except Exception as e:
                logger.error("[Ensemble] Failed to load %s: %s", name, e)
                return None

        self._ae_interp  = _load_tflite(ENSEMBLE_AE_TFLITE_PATH,  "AE")
        self._vae_interp = _load_tflite(ENSEMBLE_VAE_TFLITE_PATH, "VAE")
        self._occ_interp = _load_tflite(ENSEMBLE_OCC_TFLITE_PATH, "OCC")
        if self._ae_interp:  loaded_models.append("AE")
        if self._vae_interp: loaded_models.append("VAE")
        if self._occ_interp: loaded_models.append("OCC")

        # 7. IsolationForest
        ip = Path(ENSEMBLE_ISO_PATH)
        if ip.exists():
            try:
                self._iso = joblib.load(ip)
                loaded_models.append("IsoForest")
                logger.info("[Ensemble] IsolationForest loaded <- %s", ip)
            except Exception as e:
                logger.error("[Ensemble] IsolationForest load failed: %s", e)

        # 8. Renormalise weights for available models only
        if loaded_models:
            self._renormalise_weights(loaded_models)
            self._loaded = True
            logger.info("[Ensemble] Ready — %d/4 models: %s",
                        len(loaded_models), loaded_models)
        else:
            logger.error("[Ensemble] No models loaded — detector disabled")

        return self

    def _renormalise_weights(self, available: list):
        name_to_w = dict(zip(self._weight_names, self._weights))
        active_w  = np.array([name_to_w.get(n, 0.0) for n in available],
                             dtype=np.float32)
        total = active_w.sum()
        active_w = active_w / total if total > 0 else np.ones(len(available)) / len(available)
        self._weights      = active_w
        self._weight_names = available

    # ─────────────────────────────────────────────────────────
    # PREPROCESSING
    # ─────────────────────────────────────────────────────────

    def _preprocess(self, features: dict) -> np.ndarray:
        """Build (1,16) float32 array with clip + StandardScaler applied."""
        x = np.array([features.get(f, 0.0) for f in self.features],
                     dtype=np.float32).reshape(1, -1)
        if self._clip_lo is not None:
            x = np.clip(x, self._clip_lo, self._clip_hi)
        if self._scaler is not None:
            x = self._scaler.transform(x).astype(np.float32)
        return x

    # ─────────────────────────────────────────────────────────
    # PER-MODEL INFERENCE
    # ─────────────────────────────────────────────────────────

    def _ae_score(self, x: np.ndarray) -> float:
        """Reconstruction MSE — higher = more anomalous."""
        inp = self._ae_interp.get_input_details()
        out = self._ae_interp.get_output_details()
        self._ae_interp.set_tensor(inp[0]["index"], x)
        self._ae_interp.invoke()
        recon = self._ae_interp.get_tensor(out[0]["index"])
        return float(np.mean((x - recon) ** 2))

    def _vae_score(self, x: np.ndarray) -> float:
        """Reconstruction MSE from VAE — higher = more anomalous."""
        inp = self._vae_interp.get_input_details()
        out = self._vae_interp.get_output_details()
        self._vae_interp.set_tensor(inp[0]["index"], x)
        self._vae_interp.invoke()
        recon = self._vae_interp.get_tensor(out[0]["index"])
        return float(np.mean((x - recon) ** 2))

    def _occ_score(self, x: np.ndarray) -> float:
        """1 - P(normal) — higher = more anomalous."""
        inp = self._occ_interp.get_input_details()
        out = self._occ_interp.get_output_details()
        self._occ_interp.set_tensor(inp[0]["index"], x)
        self._occ_interp.invoke()
        p_normal = float(self._occ_interp.get_tensor(out[0]["index"]).flatten()[0])
        return 1.0 - p_normal

    def _iso_score(self, x: np.ndarray) -> float:
        """Inverted isolation depth — higher = more anomalous."""
        return float(-self._iso.score_samples(x)[0])

    # ─────────────────────────────────────────────────────────
    # SCORE NORMALISATION
    # ─────────────────────────────────────────────────────────

    def _normalise(self, name: str, raw: float) -> float:
        if name in self._score_scalers:
            mn, mx = self._score_scalers[name]
            return float(np.clip((raw - mn) / (mx - mn + 1e-9), 0.0, 1.0))
        return float(np.clip(raw, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────────────────

    def predict(self, features: dict) -> float:
        """
        Run the full ensemble on one flow feature dict.

        Returns float in [0, 1]:
            0.0 → very normal
            0.5 → decision boundary (is_anomaly threshold)
            1.0 → very anomalous
        """
        if not self._loaded:
            return 0.0
        try:
            x = self._preprocess(features)

            raw_scores = {}
            if self._ae_interp  and "AE"        in self._weight_names:
                raw_scores["AE"]        = self._ae_score(x)
            if self._vae_interp and "VAE"       in self._weight_names:
                raw_scores["VAE"]       = self._vae_score(x)
            if self._occ_interp and "OCC"       in self._weight_names:
                raw_scores["OCC"]       = self._occ_score(x)
            if self._iso        and "IsoForest" in self._weight_names:
                raw_scores["IsoForest"] = self._iso_score(x)

            if not raw_scores:
                return 0.0

            norm  = {n: self._normalise(n, s) for n, s in raw_scores.items()}
            w_map = dict(zip(self._weight_names, self._weights))
            fused = sum(w_map[n] * norm[n] for n in norm)

            # Map so that raw_thr → 0.5 in output space
            if self._raw_thr and self._raw_thr > 0:
                final = float(np.clip(fused / (2.0 * self._raw_thr), 0.0, 1.0))
            else:
                final = float(np.clip(fused, 0.0, 1.0))

            logger.debug("[Ensemble] %s  fused=%.4f  final=%.4f",
                         {n: f"{s:.3f}" for n, s in norm.items()}, fused, final)
            return final

        except Exception as e:
            logger.error("[Ensemble] Inference error: %s", e, exc_info=True)
            return 0.0

    # ─────────────────────────────────────────────────────────
    # PUBLIC HELPERS
    # ─────────────────────────────────────────────────────────

    def predict_detailed(self, features: dict) -> dict:
        """
        Same as predict() but also returns per-model raw + normalised scores
        and individual anomaly verdicts.

        Returns
        -------
        dict with keys:
            final_score   : float  — overall ensemble score (same as predict())
            is_anomaly    : bool   — final_score >= 0.5
            models        : dict   — per-model breakdown:
                {
                  "AE":        {"raw": float, "norm": float, "weight": float, "anomaly": bool},
                  "VAE":       {...},
                  "OCC":       {...},
                  "IsoForest": {...},
                }
            fused_raw     : float  — weighted sum before threshold mapping
            threshold     : float  — raw_thr used for mapping (or None)
        """
        if not self._loaded:
            return {"final_score": 0.0, "is_anomaly": False,
                    "models": {}, "fused_raw": 0.0, "threshold": None}
        try:
            x = self._preprocess(features)

            raw_scores = {}
            if self._ae_interp  and "AE"        in self._weight_names:
                raw_scores["AE"]        = self._ae_score(x)
            if self._vae_interp and "VAE"       in self._weight_names:
                raw_scores["VAE"]       = self._vae_score(x)
            if self._occ_interp and "OCC"       in self._weight_names:
                raw_scores["OCC"]       = self._occ_score(x)
            if self._iso        and "IsoForest" in self._weight_names:
                raw_scores["IsoForest"] = self._iso_score(x)

            if not raw_scores:
                return {"final_score": 0.0, "is_anomaly": False,
                        "models": {}, "fused_raw": 0.0, "threshold": None}

            norm  = {n: self._normalise(n, s) for n, s in raw_scores.items()}
            w_map = dict(zip(self._weight_names, self._weights))
            fused = sum(w_map[n] * norm[n] for n in norm)

            if self._raw_thr and self._raw_thr > 0:
                final = float(np.clip(fused / (2.0 * self._raw_thr), 0.0, 1.0))
            else:
                final = float(np.clip(fused, 0.0, 1.0))

            # Per-model anomaly verdict: norm score > 0.5 means that model
            # individually thinks it's anomalous relative to its own scale
            model_breakdown = {
                n: {
                    "raw":     round(raw_scores[n], 6),
                    "norm":    round(norm[n], 4),
                    "weight":  round(float(w_map[n]), 4),
                    "anomaly": norm[n] >= 0.5,
                }
                for n in raw_scores
            }

            return {
                "final_score": round(final, 4),
                "is_anomaly":  final >= 0.5,
                "models":      model_breakdown,
                "fused_raw":   round(fused, 6),
                "threshold":   self._raw_thr,
            }

        except Exception as e:
            logger.error("[Ensemble] predict_detailed error: %s", e, exc_info=True)
            return {"final_score": 0.0, "is_anomaly": False,
                    "models": {}, "fused_raw": 0.0, "threshold": None}

    def is_anomaly(self, score: float) -> bool:
        """score >= 0.5 → anomaly (maps to the optimal-F1 threshold)."""
        return score >= 0.5

    def is_ready(self) -> bool:
        return self._loaded and bool(self._weight_names)

    def model_summary(self) -> dict:
        if not self._loaded:
            return {}
        return {n: float(w) for n, w in zip(self._weight_names, self._weights)}