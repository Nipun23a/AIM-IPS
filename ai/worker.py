"""
ai/worker.py
============
Asyncio background worker — continuously processes the AI threat analysis queue.

Started in main.py lifespan alongside run_db_writer / run_blacklist_writer:

    ai_worker_task = asyncio.ensure_future(run_ai_analysis_worker(r.raw))

The worker runs in the same process but offloads the blocking Claude HTTP call
to a thread-pool executor so it never stalls the event loop.

Environment variables (add to .env):
    ANTHROPIC_API_KEY   — required
    HONEYDB_API_ID      — optional, enables honeypot reputation data
    HONEYDB_API_KEY     — optional
    ABUSEIPDB_API_KEY   — optional, enables crowd-sourced abuse scores
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

POLL_SLEEP  = 0.5   # seconds between queue polls when empty
RETRY_SLEEP = 2.0   # seconds to wait after an unexpected error


async def run_ai_analysis_worker(redis_raw, db_pool=None) -> None:
    """
    Infinite asyncio coroutine.  Cancel to stop.

    Args:
        redis_raw  : raw redis-py client (same one used by db_writer)
        db_pool    : optional asyncpg pool — if provided, analyses are also
                     written to PostgreSQL threat_analyses table
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        logger.warning(
            "[AIWorker] ANTHROPIC_API_KEY not set — "
            "AI threat analysis worker is disabled"
        )
        return

    # Lazy import so startup succeeds even if `anthropic` is not installed yet
    try:
        from ai.threat_analysis import ThreatAnalysisQueue, AIReasoningAgent
    except ImportError as e:
        logger.error("[AIWorker] Import failed: %s", e)
        return

    queue = ThreatAnalysisQueue(redis_raw)

    try:
        agent = AIReasoningAgent(
            anthropic_api_key = anthropic_key,
            honeydb_api_id    = os.getenv("HONEYDB_API_ID",    ""),
            honeydb_api_key   = os.getenv("HONEYDB_API_KEY",   ""),
            abuseipdb_api_key = os.getenv("ABUSEIPDB_API_KEY", ""),
        )
    except Exception as e:
        logger.error("[AIWorker] Agent init failed: %s", e)
        return

    logger.info("[AIWorker] Started — polling '%s'", "ai:threat:queue")

    loop = asyncio.get_event_loop()

    while True:
        try:
            threat = queue.dequeue_threat()

            if threat is None:
                await asyncio.sleep(POLL_SLEEP)
                continue

            event_id = threat.get("event_id", "?")
            src_ip   = threat.get("ip_address", "?")
            logger.info("[AIWorker] Analyzing threat %s  src=%s", event_id, src_ip)

            t_start = time.monotonic()

            # ── Run blocking Claude + HTTP calls in thread pool ───────────
            try:
                analysis = await loop.run_in_executor(
                    None,                    # default ThreadPoolExecutor
                    agent.analyze_threat,
                    threat,
                )
            except Exception as e:
                logger.error("[AIWorker] Analysis failed for %s: %s", event_id, e)
                # Store error result so frontend doesn't spin forever
                queue.store_result(event_id, {
                    "status":   "error",
                    "event_id": event_id,
                    "error":    str(e),
                    "analysis_timestamp": time.time(),
                })
                continue

            elapsed = round(time.monotonic() - t_start, 2)

            # ── Store in Redis (immediate dashboard access) ───────────────
            queue.store_result(event_id, analysis)

            # ── Push adaptive regex rules back into Layer 1 ───────────────
            try:
                from ai.adaptive_rules import AdaptiveRuleStore
                tc       = analysis.get("threat_classification") or {}
                severity = tc.get("severity", "MEDIUM")
                patterns = (analysis.get("mitigation_recommendations") or {}).get("regex_patterns") or []
                if patterns and severity in ("CRITICAL", "HIGH"):
                    store = AdaptiveRuleStore(redis_raw)
                    accepted = store.push_patterns(
                        patterns    = patterns,
                        attack_type = tc.get("attack_type", "unknown"),
                        event_id    = event_id,
                        severity    = severity,
                        confidence  = float(tc.get("confidence", 0)),
                    )
                    if accepted:
                        logger.info("[AIWorker] Adaptive rules added: %s", accepted)
            except Exception as _ae:
                logger.debug("[AIWorker] Adaptive rule push error: %s", _ae)

            # ── Store in PostgreSQL (persistent audit trail) ──────────────
            if db_pool:
                try:
                    async with db_pool.acquire() as conn:
                        await _store_analysis_db(conn, event_id, analysis)
                except Exception as e:
                    logger.error("[AIWorker] DB write failed for %s: %s", event_id, e)

            severity = (
                (analysis.get("threat_classification") or {})
                .get("severity", "?")
            )
            logger.info(
                "[AIWorker] Complete: %s  severity=%s  elapsed=%.1fs",
                event_id, severity, elapsed,
            )

        except asyncio.CancelledError:
            logger.info("[AIWorker] Shutting down gracefully")
            break
        except Exception as e:
            logger.error("[AIWorker] Unexpected error: %s", e, exc_info=True)
            await asyncio.sleep(RETRY_SLEEP)


# ── PostgreSQL persistence ─────────────────────────────────────────────────────

_INSERT_ANALYSIS_SQL = """
    INSERT INTO threat_analyses (
        event_id, analysis_timestamp, attack_type, owasp_category,
        severity, confidence, is_novel_variant, mitre_id, mitre_name,
        mitre_tactic, kill_chain_phase, root_cause_analysis,
        attack_sophistication, evasion_techniques,
        waf_rules, regex_patterns, threshold_adjustments,
        immediate_actions, long_term_fixes,
        analyst_summary, iocs,
        global_prevalence, honeydb_count, abuseipdb_score,
        full_analysis
    ) VALUES (
        $1,  to_timestamp($2), $3,  $4,  $5,  $6,  $7,  $8,  $9,
        $10, $11, $12, $13, $14::jsonb, $15::jsonb, $16::jsonb,
        $17::jsonb, $18::jsonb, $19::jsonb, $20, $21::jsonb,
        $22, $23, $24, $25::jsonb
    )
    ON CONFLICT (event_id) DO NOTHING
"""


async def _store_analysis_db(conn, event_id: str, analysis: dict) -> None:
    tc    = analysis.get("threat_classification") or {}
    mitre = tc.get("mitre_technique") or {}
    rec   = analysis.get("mitigation_recommendations") or {}
    intel = analysis.get("threat_intelligence") or {}

    await conn.execute(
        _INSERT_ANALYSIS_SQL,
        str(event_id)[:36],
        float(analysis.get("analysis_timestamp") or time.time()),
        str(tc.get("attack_type",   ""))[:100],
        str(tc.get("owasp_category",""))[:200],
        str(tc.get("severity",      "MEDIUM"))[:20],
        float(tc.get("confidence",  0)),
        bool(tc.get("is_novel_variant", False)),
        str(mitre.get("id",   ""))[:20],
        str(mitre.get("name", ""))[:200],
        str(mitre.get("tactic", ""))[:100],
        str(tc.get("kill_chain_phase", ""))[:100],
        str(analysis.get("root_cause_analysis", ""))[:2000],
        str(analysis.get("attack_sophistication", ""))[:50],
        json.dumps(analysis.get("evasion_techniques")        or []),
        json.dumps(rec.get("waf_rules")                      or []),
        json.dumps(rec.get("regex_patterns")                 or []),
        json.dumps(rec.get("threshold_adjustments")          or {}),
        json.dumps(rec.get("immediate_actions")              or []),
        json.dumps(rec.get("long_term_fixes")                or []),
        str(analysis.get("analyst_summary", ""))[:1000],
        json.dumps(analysis.get("iocs")                      or []),
        str(intel.get("global_prevalence", "isolated"))[:20],
        int((intel.get("honeydb")  or {}).get("count",                   0) or 0),
        int((intel.get("abuseipdb") or {}).get("abuse_confidence_score", 0) or 0),
        json.dumps(analysis),
    )
