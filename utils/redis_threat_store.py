"""
utils/redis_threat_store.py
───────────────────────────
Single source of truth for the network-layer threat score Redis schema.

Both sides import from here:
  • Network Layer IPS  → write()  (after every completed flow)
  • Application Layer  → read() / get_score()  (on every HTTP request)

Key:    threat:ip:{ip}          (KEY_NETWORK_THREAT from shared/constants.py)
TTL:    60 s                    (TTL_NETWORK_THREAT  from shared/constants.py)

Schema
──────
  score       float  fused threat score 0-1
  net_lgbm    float  LightGBM component score
  ensemble    float  Ensemble detector component score
  attack_type str    clean | ddos | portscan | bot | ZeroDay
  confidence  float  max(lgbm_score, ensemble_score)
  timestamp   float  unix epoch of write
"""

import json
import logging
import time
from typing import Optional

import redis as _redis

from shared.constants import KEY_NETWORK_THREAT, TTL_NETWORK_THREAT

logger = logging.getLogger(__name__)

# ── Schema field names ─────────────────────────────────────────────────────
# Single definition — change here and both writer and reader stay in sync.
F_SCORE       = "score"
F_NET_LGBM    = "net_lgbm"
F_ENSEMBLE    = "ensemble"
F_ATTACK_TYPE = "attack_type"
F_CONFIDENCE  = "confidence"
F_TIMESTAMP   = "timestamp"


def write(
    r:           _redis.Redis,
    ip:          str,
    score:       float,
    net_lgbm:    float = 0.0,
    ensemble:    float = 0.0,
    attack_type: str   = "unknown",
    confidence:  float = 0.0,
    ttl:         int   = TTL_NETWORK_THREAT,
) -> bool:
    """
    Write a threat score for *ip*. Returns True on success.

    Called by NetworkClassifier after every completed flow.
    The raw redis.Redis client is expected (use RedisClient.raw if you
    hold a RedisClient wrapper instance).
    """
    key = KEY_NETWORK_THREAT.format(ip=ip)
    payload = json.dumps({
        F_SCORE:       round(float(score),      4),
        F_NET_LGBM:    round(float(net_lgbm),   4),
        F_ENSEMBLE:    round(float(ensemble),    4),
        F_ATTACK_TYPE: attack_type,
        F_CONFIDENCE:  round(float(confidence), 4),
        F_TIMESTAMP:   round(time.time(),        3),
    })
    try:
        r.setex(key, ttl, payload)
        return True
    except _redis.RedisError as e:
        logger.warning("[ThreatStore] write failed for %s: %s", ip, e)
        return False


def read(r: _redis.Redis, ip: str) -> Optional[dict]:
    """
    Read the threat record for *ip*. Returns None if missing or expired.

    Called on every HTTP request by the application layer middleware.
    Benchmarks at ~0.1 ms on a local Redis instance.
    """
    key = KEY_NETWORK_THREAT.format(ip=ip)
    try:
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except (_redis.RedisError, json.JSONDecodeError) as e:
        logger.warning("[ThreatStore] read failed for %s: %s", ip, e)
        return None


def get_score(r: _redis.Redis, ip: str, default: float = 0.0) -> float:
    """
    Return just the fused score float. Returns *default* (0.0) if no entry.

    Minimal allocation — optimised for the middleware hot path.
    """
    data = read(r, ip)
    if data is None:
        return default
    return float(data.get(F_SCORE, default))
