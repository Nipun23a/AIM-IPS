"""

Single source of truth for All thresholds, weights, TTLs, and config used across the AIM-IPS system.
"""

# Layer 3 - Score Thresholds Values
SCORE_ALLOW_MAX = 0.35 # score < 0.35 -> ALLOW
SCORE_DELAY_MIN = 0.35 # 0.35 <= score < 0.50 -> DELAY
SCORE_DEALY_MAX = 0.50
SCORE_THROTTLE_MIN = 0.50 # 0.50 <= score < 0.65 -> THROTTLE
SCORE_THROTTLE_MAX = 0.65
SCORE_CAPTCHA_MIN = 0.65 # 0.65 <= score < 0.85 -> CAPTCHA
SCORE_CAPTCHA_MAX = 0.85
SCORE_BLOCK_MIN = 0.85 # score >= 0.85 -> BLOCK


# Layer 3 - Fusion Weights


WEIGHT_LAYER1_REGEX = 0.20  # Regex pre-filter confidence 
WEIGHT_LAYER2_LGBM = 0.35   # LightGBM application classifier
WEIGHT_LAYER2_CNN = 0.25    # CNN Autoencoder anomaly score
WEIGHT_NETWORK = 0.20       # Network layer (LightGBM + TCN fused) 

_WEIGHT_SUM = WEIGHT_LAYER1_REGEX + WEIGHT_LAYER2_LGBM + WEIGHT_LAYER2_CNN + WEIGHT_NETWORK
assert abs(_WEIGHT_SUM - 1.0) < 1e-6, f"Weights must sum to 1.0, but got {_WEIGHT_SUM}"


# Network Layer - internal fusion weights (LightGBM + TCN)
WEIGHT_NETWORK_LGBM = 0.55  # LightGBM network classifier (known: DDos, PortScan, Botnet)
WEIGHT_NETWORK_TCN = 0.45   # TCN TFLite (zero-day network attacks)

# Layer 2 - CNN Autoencoder internal fusion
WEIGHT_CNN_RECON = 0.50  # Reconstruction error score
WEIGHT_CNN_MAHAL = 0.50  # Mahalanobis distance score

# Layer 0 - Rate Limiting
MAX_REQUESTS_PER_MINUTE = 100  # Max requests per minute before rate limiting is applied

# Redis TTLs (seconds)
TTL_NETWORK_THREAT = 60     # network layer score freshness
TTL_BLACKLIST_TEMP = 3600   # 1 hour temporary block
TTL_BLACKLIST_PERM = -1     # sentinel: no expiry (permanent)
TTL_CAPTCHA = 300             # 5 minutes to solve CAPTCHA
TTL_RATE_LIMIT = 60           # sliding rate limit window
TTL_REQUEST_LOG = 120         # short-term per-request score log for dashboard


# Redis Key Templates
KEY_NETWORK_THREAT = "threat:ip:{ip}"
KEY_BLACKLIST = "blacklist:ip:{ip}"
KEY_RATE_LIMIT = "ratelimit:ip:{ip}"
KEY_CAPTCHA_SESSION = "session:ip:{ip}:captcha"
KEY_REQUEST_LOG = "reqlog:id:{req_id}"


# Pipeline Actions 

ACTION_ALLOW = "ALLOW"
ACTION_DELAY = "DELAY"
ACTION_THROTTLE = "THROTTLE"
ACTION_CAPTCHA = "CAPTCHA"
ACTION_BLOCK = "BLOCK"

# Layer Identifiers
LAYER_0 = "layer0_firewall"
LAYER_1 = "layer1_regex"
LAYER_2_LGBM = "layer2_lgbm"
LAYER_2_CNN = "layer2_cnn"
LAYER_NETWORK = "network_ips"
LAYER_3 = "layer3_response"

# Attack Label Strings

LABEL_NORM          = "norm"           
LABEL_CLEAN         = "clean"          
LABEL_SQLI          = "sqli"
LABEL_XSS           = "xss"
LABEL_PATH_TRAV     = "path-traversal"  
LABEL_CMD_INJECT    = "cmdi"            
LABEL_DDOS          = "ddos"
LABEL_PORTSCAN      = "portscan"
LABEL_BOTNET        = "botnet"
LABEL_ANOMALY       = "anomaly"
LABEL_ZERODAY       = "zeroday"
LABEL_UNKNOWN       = "unknown"

ATTACK_LABELS = {LABEL_SQLI, LABEL_XSS, LABEL_PATH_TRAV, LABEL_CMD_INJECT,
                 LABEL_DDOS, LABEL_PORTSCAN, LABEL_BOTNET, LABEL_ANOMALY, LABEL_ZERODAY}


NORMAL_LABELS = {LABEL_NORM, LABEL_CLEAN, "normal", "benign"}