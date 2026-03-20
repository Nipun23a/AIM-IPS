"""
db/writer.py
============
Background async worker — reads events from Redis queue and batch-inserts
into PostgreSQL.  Runs as a FastAPI lifespan task, completely decoupled
from the request pipeline.

Data flow
---------
  IPSMiddleware._log_async()
      └─ LPUSH "db:queue"  (1 extra Redis cmd in existing fire-and-forget)
              ↓
  DBWriter (this module) — runs independently
      └─ RPOP "db:queue" → accumulate → batch INSERT → PostgreSQL
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/aimips",
)
REDIS_QUEUE           = "db:queue"
REDIS_BLACKLIST_QUEUE = "db:blacklist:queue"
BATCH_SIZE   = 50       # flush when buffer reaches this many events
FLUSH_EVERY  = 5.0      # flush at most every N seconds regardless of batch size
POLL_SLEEP   = 0.2      # seconds between Redis polls when queue is empty


# ── PostgreSQL helpers ────────────────────────────────────────────────────────

async def _connect(retries: int = 6) -> Optional[asyncpg.Connection]:
    for attempt in range(retries):
        try:
            conn = await asyncpg.connect(DATABASE_URL)
            logger.info("[DBWriter] PostgreSQL connected ✓")
            return conn
        except Exception as e:
            wait = 2 ** attempt
            logger.warning("[DBWriter] Connect attempt %d/%d failed (%s) — retrying in %ds",
                           attempt + 1, retries, e, wait)
            await asyncio.sleep(wait)
    logger.error("[DBWriter] Could not connect to PostgreSQL after %d attempts", retries)
    return None


def _extract_best_label(event: dict) -> str:
    for ls in event.get("layer_scores", []):
        lbl = ls.get("label", "")
        if lbl and lbl not in ("clean", "norm", "normal", ""):
            return lbl
    return event.get("block_reason") or (
        "clean" if event.get("action") == "ALLOW" else "unknown"
    )


def _to_row(event: dict) -> tuple:
    """Convert a raw event dict (from Redis) into a DB row tuple."""
    scores = event.get("scores") or {}
    best_label = (event.get("best_label") or _extract_best_label(event) or "unknown")[:100]
    return (
        str(event.get("request_id", ""))[:36],
        str(event.get("ip", "0.0.0.0"))[:45],
        str(event.get("method", "GET"))[:10],
        str(event.get("path", "/"))[:500],
        float(event.get("timestamp") or time.time()),
        float(event.get("final_score") or 0),
        str(event.get("action", "ALLOW"))[:20],
        str(event.get("block_reason") or "")[:500],
        bool(event.get("short_circuited", False)),
        float(event.get("network_score") or 0),
        float(event["latency_ms"]) if event.get("latency_ms") is not None else None,
        best_label,
        float(scores.get("regex_conf") or 0),
        float(scores.get("app_lgbm_score") or 0),
        float(scores.get("net_lgbm_score") or 0),
        float(scores.get("deep_anomaly") or 0),
        json.dumps(event.get("layer_scores") or []),
    )


_INSERT_SQL = """
    INSERT INTO attack_events (
        request_id, ip, method, path, timestamp,
        final_score, action, block_reason, short_circuited,
        network_score, latency_ms, best_label,
        regex_conf, app_lgbm_score, net_lgbm_score, deep_anomaly,
        layer_scores
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
    ON CONFLICT (request_id) DO NOTHING
"""


async def _flush(conn: asyncpg.Connection, buffer: list) -> None:
    if not buffer:
        return
    try:
        rows = []
        for event in buffer:
            try:
                rows.append(_to_row(event))
            except Exception as e:
                logger.debug("[DBWriter] Skipping malformed event: %s", e)
        if rows:
            await conn.executemany(_INSERT_SQL, rows)
            logger.debug("[DBWriter] Inserted %d events into PostgreSQL", len(rows))
    except asyncpg.PostgresError as e:
        logger.error("[DBWriter] Batch insert failed: %s", e)
        raise


# ── Main worker loop ──────────────────────────────────────────────────────────

async def run_db_writer(redis_raw) -> None:
    """
    Infinite background loop.

    - Polls Redis `db:queue` with RPOP (non-blocking, no executor needed).
    - Accumulates events and flushes to PostgreSQL in batches.
    - Reconnects to PostgreSQL automatically on connection loss.
    - On PostgreSQL unavailability, drains the Redis queue silently so it
      does not grow unbounded.
    """
    conn = await _connect()
    buffer: list = []
    last_flush = time.monotonic()

    logger.info("[DBWriter] Started — polling Redis queue '%s'", REDIS_QUEUE)

    while True:
        try:
            # ── Pop one event from Redis (non-blocking) ──────────────────
            raw = redis_raw.rpop(REDIS_QUEUE)

            if raw:
                try:
                    buffer.append(json.loads(raw))
                except Exception:
                    pass  # skip unparseable entries
            else:
                # Queue empty — small sleep to avoid busy-spinning
                await asyncio.sleep(POLL_SLEEP)

            # ── Flush condition: batch full OR timeout ───────────────────
            now = time.monotonic()
            should_flush = (
                len(buffer) >= BATCH_SIZE
                or (buffer and (now - last_flush) >= FLUSH_EVERY)
            )

            if should_flush:
                if conn is None or conn.is_closed():
                    logger.warning("[DBWriter] Reconnecting to PostgreSQL…")
                    conn = await _connect()

                if conn:
                    await _flush(conn, buffer)
                else:
                    logger.warning("[DBWriter] PostgreSQL unavailable — %d events dropped", len(buffer))

                buffer.clear()
                last_flush = now

        except asyncpg.PostgresConnectionStatusError:
            logger.warning("[DBWriter] PostgreSQL connection lost — reconnecting…")
            conn = await _connect()

        except Exception as e:
            logger.error("[DBWriter] Unexpected error: %s", e, exc_info=True)
            await asyncio.sleep(1)


# ── Blacklist writer ───────────────────────────────────────────────────────────

_BLACKLIST_SQL = """
    INSERT INTO blacklisted_ips (ip, reason, permanent, source, blocked_at, expires_at)
    VALUES ($1, $2, $3, $4, to_timestamp($5), $6)
    ON CONFLICT (ip) DO UPDATE
        SET reason       = EXCLUDED.reason,
            permanent    = EXCLUDED.permanent,
            source       = EXCLUDED.source,
            blocked_at   = EXCLUDED.blocked_at,
            expires_at   = EXCLUDED.expires_at,
            unblocked_at = NULL
"""


async def run_blacklist_writer(redis_raw) -> None:
    """
    Background loop: pops blacklist events from Redis queue and upserts
    into the blacklisted_ips PostgreSQL table for audit trail.

    Queue key:  db:blacklist:queue
    Payload:    {"ip", "reason", "permanent", "source", "timestamp"}
    """
    conn = await _connect()
    logger.info("[BlacklistWriter] Started — polling '%s'", REDIS_BLACKLIST_QUEUE)

    while True:
        try:
            raw = redis_raw.rpop(REDIS_BLACKLIST_QUEUE)
            if not raw:
                await asyncio.sleep(POLL_SLEEP)
                continue

            try:
                event = json.loads(raw)
            except Exception:
                continue

            if conn is None or conn.is_closed():
                conn = await _connect()
            if conn is None:
                logger.warning("[BlacklistWriter] PostgreSQL unavailable — entry dropped")
                continue

            permanent  = bool(event.get("permanent", False))
            ts         = float(event.get("timestamp") or time.time())
            ttl        = event.get("ttl_seconds")
            expires_at = None
            if not permanent and ttl:
                from datetime import datetime, timezone, timedelta
                expires_at = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(seconds=int(ttl))

            await conn.execute(
                _BLACKLIST_SQL,
                str(event.get("ip", ""))[:45],
                str(event.get("reason", ""))[:500],
                permanent,
                str(event.get("source", "auto"))[:50],
                ts,
                expires_at,
            )
            logger.debug("[BlacklistWriter] Upserted blacklist entry for %s", event.get("ip"))

        except asyncpg.PostgresConnectionStatusError:
            logger.warning("[BlacklistWriter] PostgreSQL connection lost — reconnecting…")
            conn = await _connect()

        except Exception as e:
            logger.error("[BlacklistWriter] Unexpected error: %s", e, exc_info=True)
            await asyncio.sleep(1)
