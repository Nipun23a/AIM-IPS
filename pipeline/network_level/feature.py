"""
pipeline/network_level/feature.py
───────────────────────────────────
Feature definitions and model path constants for the network layer IPS.

Ensemble detector paths updated to point at the semi-supervised
AE + VAE + OCC-Net + IsolationForest ensemble trained by
train_ensemble_detector.py.
"""

THREAT_FEATURES = [
    "flow duration",
    "total fwd packets",
    "total backward packets",
    "total length of fwd packets",
    "total length of bwd packets",
    "fwd packet length mean",
    "bwd packet length mean",
    "flow bytes/s",
    "flow packets/s",
    "syn flag count",
    "ack flag count",
    "psh flag count",
    "packet length mean",
    "packet length std",
    "idle mean",
    "idle std",
]

NUM_FEATURES = len(THREAT_FEATURES)

# ── Flow accumulator tuning ────────────────────────────────────────────────
FLOW_TIMEOUT_SECONDS = 30
MIN_PACKETS_PER_FLOW = 4
MAX_FLOW_DURATION    = 120

# ── Classification labels ──────────────────────────────────────────────────
LABEL_BENIGN   = "benign"
LABEL_DDOS     = "ddos"
LABEL_PORTSCAN = "portscan"
LABEL_BOTNET   = "bot"        # lowercase — matches CICIDS "Bot" normalised
LABEL_ZERODAY  = "ZeroDay"
LABEL_CLEAN    = "clean"

# ── LightGBM paths (unchanged) ─────────────────────────────────────────────
LGBM_MODEL_PATH    = "models/threat_classifier/lgb_model.pkl"
LGBM_FEATURES_PATH = "models/threat_classifier/features.pkl"

# ── Ensemble anomaly detector paths ────────────────────────────────────────
#
#  Directory layout produced by train_ensemble_detector.py:
#
#  models/anomly_detector/network_level_attacks_anomality/
#      ensemble_scaler.pkl          <- StandardScaler + [p1,p99] clip bounds
#      ensemble_threshold.pkl       <- optimal-F1 decision threshold
#      feature_info.pkl             <- feature list + metadata
#      ensemble/
#          models/
#              ae_fp32.tflite       <- Deep Autoencoder  (benign-only)
#              vae_fp32.tflite      <- Variational AE    (benign-only)
#              occ_fp32.tflite      <- OCC-Net           (semi-supervised)
#              iso_forest.pkl       <- Isolation Forest  (semi-supervised)
#              ensemble_weights.pkl <- learned SLSQP fusion weights (W_MAX=0.40)
#              score_scalers.pkl    <- per-model [min,max] normalisers
#
_ENS_BASE   = "models/anomly_detector/network_level_attacks_anomality"
_ENS_MODELS = f"{_ENS_BASE}/ensemble/models"

ENSEMBLE_SCALER_PATH        = f"{_ENS_BASE}/ensemble_scaler.pkl"
ENSEMBLE_THRESHOLD_PATH     = f"{_ENS_BASE}/ensemble_threshold.pkl"
ENSEMBLE_FEAT_PATH          = f"{_ENS_BASE}/feature_info.pkl"

ENSEMBLE_AE_TFLITE_PATH     = f"{_ENS_MODELS}/ae_fp32.tflite"
ENSEMBLE_VAE_TFLITE_PATH    = f"{_ENS_MODELS}/vae_fp32.tflite"
ENSEMBLE_OCC_TFLITE_PATH    = f"{_ENS_MODELS}/occ_fp32.tflite"
ENSEMBLE_ISO_PATH           = f"{_ENS_MODELS}/iso_forest.pkl"
ENSEMBLE_WEIGHTS_PATH       = f"{_ENS_MODELS}/ensemble_weights.pkl"
ENSEMBLE_SCORE_SCALERS_PATH = f"{_ENS_MODELS}/score_scalers.pkl"

# ── Legacy aliases ─────────────────────────────────────────────────────────
# These pointed at the old single-model TCN.  New code should use the
# ENSEMBLE_* constants above.  Kept so any un-migrated imports don't crash.
TCN_TFLITE_PATH   = ENSEMBLE_AE_TFLITE_PATH   # was tcn/models/tcn_fp32.tflite
TCN_SCALER_PATH   = ENSEMBLE_SCALER_PATH       # was anomality/scaler.pkl
TCN_FEATURES_PATH = ENSEMBLE_FEAT_PATH         # was anomality/feature_info.pkl