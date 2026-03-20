"""
pipeline/correlation/cross_pipeline.py
────────────────────────────────────────
Cross-Pipeline Temporal Correlation Engine

Detects multi-stage attacks where the same IP performs network-level
reconnaissance followed by application-level exploitation.  Each attack
alone may be below the block threshold, but the sequence indicates
coordinated attack intent and the final score is amplified accordingly.

Redis schema
────────────
  Key     : corr:hist:{ip}          (sorted set)
  ZSET    : score  = unix timestamp   (enables ZRANGEBYSCORE time windows)
            member = JSON event       {"ts", "layer", "type", "score"}
  TTL     : HISTORY_TTL seconds      (refreshed on every write)

Amplification table
───────────────────
  Distinct attack types in window   Multiplier
  ─────────────────────────────────────────────
  1                                  1.0   (no change)
  2                                  1.3
  3                                  1.5
  4+                                 1.7

  Multiplier is applied only when BOTH "network" AND "application" layers
  are present in the correlation window.

Integration
───────────
  Network pipeline  → call record_and_correlate() after classify_flow()
  Application layer → call record_and_correlate() inside middleware after
                       all layer scores are known

Performance
───────────
  4 Redis commands via a single pipeline → ~0.4–0.8 ms on localhost
  Zero local state → fully thread-safe and multi-process safe
  Fails open on Redis error (never blocks the pipeline)
"""

import json
import logging
import time
from typing import Optional

import redis as _redis

logger = logging.getLogger(__name__)

# ── Layer identifiers (import these in both pipelines) ──────────────────────
LAYER_NETWORK     = "network"
LAYER_APPLICATION = "application"

# ── Redis key ────────────────────────────────────────────────────────────────
KEY_CORR_HISTORY = "corr:hist:{ip}"

# ── Configuration ────────────────────────────────────────────────────────────
HISTORY_TTL       = 60    # seconds — how long attack events are retained
DEFAULT_WINDOW    = 10    # seconds — correlation detection window
MAX_EVENTS_PER_IP = 50    # cap ZSET size to prevent memory bloat

# ── Normal / ignored labels (not counted as distinct attack types) ────────────
_NORMAL_TYPES = frozenset({"clean", "norm", "normal", "benign", "unknown", ""})

# ── Amplification table ───────────────────────────────────────────────────────
_MULTIPLIER_TABLE = {1: 1.0, 2: 1.3, 3: 1.5}
_MULTIPLIER_MAX   = 1.7   # for 4+ distinct attack types


def _multiplier(n_distinct: int) -> float:
    """Return amplification multiplier for n_distinct attack types."""
    if n_distinct >= 4:
        return _MULTIPLIER_MAX
    return _MULTIPLIER_TABLE.get(n_distinct, 1.0)


# ── Public API ────────────────────────────────────────────────────────────────

def record_and_correlate(
    r:           _redis.Redis,
    ip:          str,
    layer:       str,
    attack_type: str,
    base_score:  float,
    window:      int  = DEFAULT_WINDOW,
    record:      bool = True,
) -> dict:
    """
    Record an attack event for *ip* and check for cross-pipeline correlation.

    Call this from both pipelines whenever a non-benign score is produced:

        # Network pipeline (NetworkClassifier.classify_flow)
        from pipeline.correlation import record_and_correlate, LAYER_NETWORK
        corr = record_and_correlate(r, src_ip, LAYER_NETWORK, attack_type, fused_score)
        final_score = corr["amplified_score"]

        # Application middleware (after layer scores fused)
        from pipeline.correlation import record_and_correlate, LAYER_APPLICATION
        corr = record_and_correlate(r, client_ip, LAYER_APPLICATION, label, app_score)
        final_score = corr["amplified_score"]

    Args:
        r:           Raw redis.Redis client  (use RedisClient.raw)
        ip:          Source IP address
        layer:       LAYER_NETWORK or LAYER_APPLICATION
        attack_type: Attack label from the calling model (e.g. "portscan", "sqli")
        base_score:  Raw threat score 0.0–1.0 from the calling pipeline
        window:      Seconds to look back for correlated events (default 10)
        record:      Write this event to history (False = read-only correlation check)

    Returns:
        {
          "correlated":      bool,   True if cross-layer attack sequence detected
          "multiplier":      float,  Amplification factor applied (1.0 – 1.7)
          "amplified_score": float,  min(base_score × multiplier, 1.0)
          "distinct_types":  int,    Distinct attack types seen in window
          "layers_seen":     list,   Layers observed in the correlation window
          "history_count":   int,    Total events in the ZSET for this IP
        }

    Never raises — returns passthrough result (multiplier=1.0) on Redis error.
    Overhead: ~0.4–0.8 ms (4 pipelined Redis commands).
    Thread-safe: all state is in Redis, no local mutable state.
    """
    now    = time.time()
    key    = KEY_CORR_HISTORY.format(ip=ip)
    cutoff = now - HISTORY_TTL

    try:
        pipe = r.pipeline(transaction=False)

        if record:
            # Unique member: include nanosecond timestamp so two events from the
            # same IP at the same second are not treated as duplicates by Redis.
            member = json.dumps(
                {
                    "ts":    round(now, 4),
                    "layer": layer,
                    "type":  attack_type,
                    "score": round(float(base_score), 4),
                },
                separators=(",", ":"),
            )
            pipe.zadd(key, {member: now})
            # Prune events older than the retention window
            pipe.zremrangebyscore(key, "-inf", cutoff)
            # Cap cardinality to prevent runaway memory from flood attacks
            pipe.zremrangebyrank(key, 0, -(MAX_EVENTS_PER_IP + 1))
            # Refresh TTL so the key lives as long as the IP is active
            pipe.expire(key, HISTORY_TTL)

        # Always read events inside the correlation window
        pipe.zrangebyscore(key, now - window, "+inf")

        results = pipe.execute()
        # zrangebyscore is the last command → last element of results
        recent_raw: list = results[-1] if results else []

    except _redis.RedisError as e:
        logger.warning("[Correlator] Redis error for %s: %s — failing open", ip, e)
        return _passthrough(base_score)

    # ── Parse recent events ───────────────────────────────────────────────────
    layers_seen: set[str] = set()
    types_seen:  set[str] = set()

    for raw_entry in recent_raw:
        try:
            ev = json.loads(raw_entry)
        except (json.JSONDecodeError, TypeError):
            continue
        layer_val = ev.get("layer", "")
        type_val  = ev.get("type",  "")
        if layer_val:
            layers_seen.add(layer_val)
        if type_val and type_val not in _NORMAL_TYPES:
            types_seen.add(type_val)

    # ── Correlation decision ──────────────────────────────────────────────────
    correlated = (LAYER_NETWORK in layers_seen) and (LAYER_APPLICATION in layers_seen)
    n_distinct  = max(len(types_seen), 1)
    mult        = _multiplier(n_distinct) if correlated else 1.0
    amp_score   = min(float(base_score) * mult, 1.0)

    if correlated:
        logger.info(
            "[Correlator] CORRELATION DETECTED ip=%s layers=%s types=%s "
            "multiplier=%.1f base=%.3f amplified=%.3f",
            ip, sorted(layers_seen), sorted(types_seen), mult, base_score, amp_score,
        )

    return {
        "correlated":      correlated,
        "multiplier":      round(mult, 2),
        "amplified_score": round(amp_score, 4),
        "distinct_types":  n_distinct,
        "layers_seen":     sorted(layers_seen),
        "history_count":   len(recent_raw),
    }


def get_history(
    r:      _redis.Redis,
    ip:     str,
    window: int = HISTORY_TTL,
) -> list:
    """
    Return all recorded attack events for *ip* within the last *window* seconds.

    Useful for dashboard, debugging, and incident investigation.
    Returns an empty list on Redis error.
    """
    key    = KEY_CORR_HISTORY.format(ip=ip)
    cutoff = time.time() - window
    try:
        raw_entries = r.zrangebyscore(key, cutoff, "+inf")
        result = []
        for entry in raw_entries:
            try:
                result.append(json.loads(entry))
            except (json.JSONDecodeError, TypeError):
                pass
        return result
    except _redis.RedisError as e:
        logger.warning("[Correlator] get_history failed for %s: %s", ip, e)
        return []


def clear_history(r: _redis.Redis, ip: str) -> bool:
    """
    Delete all correlation history for *ip*.

    Call this after permanently blacklisting an IP so the ZSET does not
    linger until TTL expiry.  Returns True on success.
    """
    try:
        r.delete(KEY_CORR_HISTORY.format(ip=ip))
        return True
    except _redis.RedisError as e:
        logger.warning("[Correlator] clear_history failed for %s: %s", ip, e)
        return False


# ── Internal helpers ─────────────────────────────────────────────────────────

def _passthrough(base_score: float) -> dict:
    """Safe default returned when Redis is unavailable."""
    return {
        "correlated":      False,
        "multiplier":      1.0,
        "amplified_score": round(float(base_score), 4),
        "distinct_types":  0,
        "layers_seen":     [],
        "history_count":   0,
    }
