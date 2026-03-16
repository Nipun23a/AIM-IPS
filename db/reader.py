"""
db/reader.py
============
Async query functions for the dashboard API.
Replaces Redis-based _read_admin_events() / _compute_stats() in main.py.

Falls back gracefully when PostgreSQL is unavailable (returns None /
empty list so main.py can fall back to Redis or mock data).
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/aimips",
)

_pool: Optional[asyncpg.Pool] = None


async def init_pool() -> None:
    """Create the connection pool — call once from FastAPI lifespan."""
    global _pool
    try:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=10,
        )
        logger.info("[DBReader] PostgreSQL pool created ✓")
    except Exception as e:
        logger.warning("[DBReader] Could not create pool: %s — API will fall back to Redis/mock", e)


async def _pool_ok() -> bool:
    return _pool is not None and not _pool._closed


# ── Events query ──────────────────────────────────────────────────────────────

async def fetch_events(
    limit: int = 100,
    action: str = "",
    ip: str = "",
    attack_type: str = "",
) -> Optional[list]:
    """
    Returns a list of event dicts shaped identically to the Redis/mock format
    so the existing dashboard frontend requires zero changes.
    Returns None if PostgreSQL is unavailable.
    """
    if not await _pool_ok():
        return None

    conditions = ["1=1"]
    params: list = []
    i = 1

    if action:
        conditions.append(f"action = ${i}")
        params.append(action.upper())
        i += 1
    if ip:
        conditions.append(f"ip = ${i}")
        params.append(ip)
        i += 1
    if attack_type:
        conditions.append(f"(best_label ILIKE ${i} OR block_reason ILIKE ${i})")
        params.append(f"%{attack_type}%")
        i += 1

    params.append(limit)
    query = f"""
        SELECT  request_id, ip, method, path, timestamp,
                final_score, action, block_reason, short_circuited,
                network_score, latency_ms, best_label,
                regex_conf, app_lgbm_score, net_lgbm_score, deep_anomaly,
                layer_scores
        FROM    attack_events
        WHERE   {' AND '.join(conditions)}
        ORDER   BY timestamp DESC
        LIMIT   ${i}
    """

    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
    except Exception as e:
        logger.error("[DBReader] fetch_events error: %s", e)
        return None

    events = []
    for r in rows:
        ls = r["layer_scores"]
        if isinstance(ls, str):
            try:
                ls = json.loads(ls)
            except Exception:
                ls = []

        events.append({
            "request_id":     str(r["request_id"]),
            "ip":             r["ip"],
            "method":         r["method"],
            "path":           r["path"],
            "timestamp":      r["timestamp"],
            "final_score":    r["final_score"],
            "action":         r["action"],
            "block_reason":   r["block_reason"] or "",
            "short_circuited": r["short_circuited"],
            "network_score":  r["network_score"],
            "latency_ms":     r["latency_ms"],
            "best_label":     r["best_label"],
            "scores": {
                "regex_conf":     r["regex_conf"],
                "app_lgbm_score": r["app_lgbm_score"],
                "net_lgbm_score": r["net_lgbm_score"],
                "deep_anomaly":   r["deep_anomaly"],
                "final_risk":     r["final_score"],
            },
            "layer_scores": ls,
        })

    return events


# ── Stats query ───────────────────────────────────────────────────────────────

async def fetch_stats() -> Optional[dict]:
    """
    Computes all dashboard stats directly in PostgreSQL.
    Returns None if DB unavailable or no data in last 24 h.
    """
    if not await _pool_ok():
        return None

    try:
        async with _pool.acquire() as conn:
            # ── Summary counts (last 24 h) ──────────────────────────────
            summary = await conn.fetchrow("""
                SELECT
                    COUNT(*)                                                AS total,
                    COUNT(*) FILTER (WHERE action = 'BLOCK')               AS blocked,
                    COUNT(*) FILTER (WHERE action = 'ALLOW')               AS allowed,
                    COUNT(*) FILTER (WHERE action IN ('THROTTLE','CAPTCHA','DELAY'))
                                                                           AS throttled,
                    AVG(latency_ms)                                        AS avg_latency
                FROM attack_events
                WHERE timestamp >= EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours')
            """)

            if not summary or summary["total"] == 0:
                return None

            total     = int(summary["total"])
            blocked   = int(summary["blocked"])
            allowed   = int(summary["allowed"])
            throttled = int(summary["throttled"])
            avg_lat   = round(float(summary["avg_latency"] or 0), 1)

            # ── RPM buckets — last 30 minutes ──────────────────────────
            rpm_rows = await conn.fetch("""
                SELECT
                    FLOOR((EXTRACT(EPOCH FROM NOW()) - timestamp) / 60)::INT AS mins_ago,
                    COUNT(*)                                                  AS cnt,
                    COUNT(*) FILTER (WHERE action = 'BLOCK')                 AS blocked_cnt
                FROM attack_events
                WHERE timestamp >= EXTRACT(EPOCH FROM NOW()) - 1800
                GROUP BY 1
            """)

            rpm_total   = [0] * 30
            rpm_blocked = [0] * 30
            for row in rpm_rows:
                idx = int(row["mins_ago"])
                if 0 <= idx < 30:
                    rpm_total[29 - idx]   = int(row["cnt"])
                    rpm_blocked[29 - idx] = int(row["blocked_cnt"])

            # ── Attack type distribution ────────────────────────────────
            attack_rows = await conn.fetch("""
                SELECT best_label, COUNT(*) AS cnt
                FROM attack_events
                WHERE timestamp >= EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours')
                  AND best_label NOT IN ('clean','norm','normal','unknown','')
                GROUP BY best_label
                ORDER BY cnt DESC
                LIMIT 10
            """)
            attack_types = {r["best_label"]: int(r["cnt"]) for r in attack_rows}

            # ── Layer detection counts ──────────────────────────────────
            layer_rows = await conn.fetch("""
                SELECT ls->>'layer' AS layer, COUNT(*) AS cnt
                FROM attack_events,
                     jsonb_array_elements(layer_scores) AS ls
                WHERE timestamp >= EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours')
                  AND (ls->>'triggered')::boolean = TRUE
                GROUP BY 1
                ORDER BY cnt DESC
            """)
            layer_counts = {
                r["layer"]: int(r["cnt"])
                for r in layer_rows if r["layer"]
            }

            # ── Top threat IPs ─────────────────────────────────────────
            ip_rows = await conn.fetch("""
                SELECT
                    ip,
                    COUNT(*)                                            AS requests,
                    COUNT(*) FILTER (WHERE action = 'BLOCK')           AS blocked,
                    MAX(final_score)                                    AS max_score,
                    MAX(timestamp) FILTER (WHERE action = 'BLOCK')     AS last_attack_ts
                FROM attack_events
                WHERE timestamp >= EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours')
                GROUP BY ip
                ORDER BY blocked DESC, max_score DESC
                LIMIT 20
            """)

            top_ips = []
            for r in ip_rows:
                la = r["last_attack_ts"]
                top_ips.append({
                    "ip":             r["ip"],
                    "requests":       int(r["requests"]),
                    "blocked":        int(r["blocked"]),
                    "max_score":      round(float(r["max_score"] or 0), 4),
                    "last_attack_iso": (
                        datetime.fromtimestamp(float(la), tz=timezone.utc).isoformat()
                        if la else ""
                    ),
                })

    except Exception as e:
        logger.error("[DBReader] fetch_stats error: %s", e)
        return None

    return {
        "total_requests":    total,
        "blocked_attacks":   blocked,
        "allowed_requests":  allowed,
        "throttled_captcha": throttled,
        "block_rate_pct":    round(blocked / total * 100, 1) if total else 0,
        "avg_latency_ms":    avg_lat,
        "rpm_total":         rpm_total,
        "rpm_blocked":       rpm_blocked,
        "attack_types":      attack_types,
        "layer_counts":      layer_counts,
        "top_ips":           top_ips,
    }
