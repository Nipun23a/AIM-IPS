
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

FLOW_TIMEOUT_SECONDS = 30
MIN_PACKETS_PER_FLOW = 4
MAX_FLOW_DURATION = 120

LABEL_BENIGN   = "BENIGN"
LABEL_DDOS     = "DDoS"
LABEL_PORTSCAN = "PortScan"
LABEL_BOTNET   = "Bot"
LABEL_ZERODAY  = "ZeroDay"
LABEL_CLEAN    = "clean"

LGBM_MODEL_PATH    = "models/threat_classifier/lgb_model.pkl"
LGBM_FEATURES_PATH = "models/threat_classifier/features.pkl"

TCN_TFLITE_PATH    = "models/anomly_detector/network_level_attacks_anomality/tcn/models/tcn_fp32.tflite"
TCN_SCALER_PATH    = "models/anomly_detector/network_level_attacks_anomality/scaler.pkl"
TCN_FEATURES_PATH  = "models/anomly_detector/network_level_attacks_anomality/feature_info.pkl"