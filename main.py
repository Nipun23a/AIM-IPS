import json
import logging
import time
import asyncio
import uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()   # loads .env from the project root before anything else
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from pipeline.application_level.layer2 import Layer2MLOrchestrator
from api.middleware import IPSMiddleware
from utils.redis_client import RedisClient
import db.reader as db_reader
from db.writer import run_db_writer, run_blacklist_writer
from firewall.engine import StaticFirewall
from firewall.regex_filter import RegexFilter
from firewall.decisions import FirewallDecision
from shared.schemas import RequestContext
from shared.constants import (
    MAX_REQUESTS_PER_MINUTE,
    WEIGHT_LAYER1_REGEX, WEIGHT_LAYER2_LGBM, WEIGHT_LAYER2_CNN, WEIGHT_NETWORK,
    SCORE_BLOCK_MIN, SCORE_CAPTCHA_MIN, SCORE_THROTTLE_MIN, SCORE_DELAY_MIN,
    KEY_BLACKLIST, KEY_RATE_LIMIT, KEY_NETWORK_THREAT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

LGBM_MODEL_DIR = Path("models/application_layer")
CNN_MODEL_DIR = Path("models/anomaly_detector/application_level_attacks")


@asynccontextmanager
async def lifespan(app:FastAPI):
    logger.info("=" * 60)
    logger.info("AIM-IPS Starting up")
    logger.info("=" * 60)

    logger.info("[Startup] Loading Layer 2 ML models...")
    logger.info(f"[Startup] LightGBM dir : {LGBM_MODEL_DIR}")
    logger.info(f"[Startup] CNN di : {CNN_MODEL_DIR}")

    try:
        layer2 = Layer2MLOrchestrator(
            lgbm_model_dir= LGBM_MODEL_DIR,
            cnn_model_dir= CNN_MODEL_DIR,
            run_lgbm= True,
            run_cnn = True
        ).load()

        status = layer2.status()
        logger.info(f"[Startup] Layer 2 loaded:")
        logger.info(f"[Startup]   LightGBM ready : {status['lgbm_ready']}")
        logger.info(f"[Startup]   CNN ready      : {status['cnn_ready']}")

        if not status["lgbm_ready"]:
            logger.warning("[Startup] LightGBM not ready — app-layer known attacks won't be scored")
        if not status["cnn_ready"]:
            logger.warning("[Startup] CNN not ready - zero day anomality detecton disabled")

    except Exception as e:
        logger.error(f"[Startup] Layer 2 load failed: {e}", exc_info=True)
        logger.warning("[Startup] Running without ML layer — only Layer 0/1 active")
        layer2 = Layer2MLOrchestrator(run_lgbm=False, run_cnn=False)

    app.state.layer2 = layer2

    logger.info("[Startup] Connecting to Redis...")
    try:
        r = RedisClient.get_redis()
        if r.ping():
            stats = r.get_stats()
            logger.info(f"[Startup] Redis Connected | {stats}")
        else:
            logger.warning("[Startup] Redis ping failed")
    except Exception as e:
        logger.warning(
            f"[Startup] Redis unavailable: {e}\n"
            "         Pipeline will run without:\n"
            "           - Network layer threat scores\n"
            "           - Redis-backed rate limiting (falls back to in-memory)\n"
            "           - Blacklist checks\n"
            "           - Captcha sessions"
        )

    # ── PostgreSQL reader pool ─────────────────────────────────────
    logger.info("[Startup] Connecting to PostgreSQL (reader pool)...")
    await db_reader.init_pool()

    # ── Background DB writer task ──────────────────────────────────
    # Pops events from Redis db:queue and batch-inserts to PostgreSQL.
    # Completely decoupled from the request pipeline — zero latency impact.
    writer_task = None
    blacklist_writer_task = None
    try:
        r = RedisClient.get_redis()
        writer_task           = asyncio.ensure_future(run_db_writer(r.raw))
        blacklist_writer_task = asyncio.ensure_future(run_blacklist_writer(r.raw))
        logger.info("[Startup] DB writer task started ✓")
        logger.info("[Startup] Blacklist writer task started ✓")
    except Exception as e:
        logger.warning("[Startup] DB writer could not start: %s", e)

    logger.info("[Startup] AIM-IPS Ready ✓")
    logger.info("=" * 60)

    yield

    logger.info("[Shutdown] AIM-IPS shutting down")
    if writer_task:
        writer_task.cancel()
    if blacklist_writer_task:
        blacklist_writer_task.cancel()


app = FastAPI(
    title="AIM-IPS",
    description="AI-Powered Intrusion Prevention System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    IPSMiddleware,
    skip_paths=[
        "/health", "/docs", "/openapi.json", "/redoc", "/status", "/demo",
        "/api/inspect", "/api/stats", "/api/events", "/api/admin", "/api/myip",
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aimips.tech",
        "https://www.aimips.tech",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/myip")
async def my_ip(request: Request):
    """Returns the real client IP — used by the inspector to auto-fill the IP field."""
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.client.host
    return {"ip": ip}


@app.post("/api/probe")
async def probe(request: Request):
    """Target endpoint for the demo page — middleware scores every request to this."""
    return {"ok": True, "path": str(request.url.path)}


@app.get("/demo", response_class=HTMLResponse)
async def demo():
    return HTMLResponse(content= "This is demo page")


# ── /api/inspect — Pipeline Layer Inspector ───────────────────────────────────

class _LayersEnabled(BaseModel):
    layer0: bool = True
    layer1: bool = True
    network: bool = True
    layer2_lgbm: bool = True
    layer2_cnn: bool = True


class InspectRequest(BaseModel):
    ip: str = "127.0.0.1"
    method: str = "GET"
    path: str = "/api/probe"
    headers: Dict[str, str] = {}
    query_params: Dict[str, str] = {}
    body: str = ""
    layers_enabled: _LayersEnabled = _LayersEnabled()


@app.post("/api/inspect")
async def inspect_pipeline(req: InspectRequest, request: Request):
    """
    Run every pipeline layer independently and return per-layer scores.
    Disabled layers (enabled=false) are still executed but excluded from fusion.
    Never modifies Redis state (read-only).
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    req_dict = {
        "ip":           req.ip,
        "method":       req.method,
        "path":         req.path,
        "headers":      req.headers,
        "query_params": req.query_params,
        "body":         req.body,
    }
    le = req.layers_enabled.model_dump()

    # ── Read-only Redis snapshot (no INCR, no writes) ─────────────
    is_blacklisted = False
    rate_count = 0
    net_data = None
    try:
        r = RedisClient.get_redis()
        pipe = r.raw.pipeline(transaction=False)
        pipe.exists(KEY_BLACKLIST.format(ip=req.ip))
        pipe.get(KEY_RATE_LIMIT.format(ip=req.ip))
        pipe.get(KEY_NETWORK_THREAT.format(ip=req.ip))
        results = pipe.execute()
        is_blacklisted = bool(results[0])
        rate_count = int(results[1] or 0)
        net_data = json.loads(results[2]) if results[2] else None
    except Exception:
        pass

    # ── Layer 0 — Static Firewall ─────────────────────────────────
    firewall = StaticFirewall()
    fw_decision, fw_reason = firewall.inspect(req_dict)

    l0_rate_limited = rate_count > MAX_REQUESTS_PER_MINUTE
    l0_triggered = False

    if is_blacklisted:
        l0_decision, l0_score, l0_triggered = "BLOCK", 1.0, True
        l0_label, l0_reason, l0_fw_match = "blacklisted", "IP is blacklisted", None
    elif l0_rate_limited:
        l0_decision, l0_score, l0_triggered = "BLOCK", 1.0, True
        l0_label, l0_reason, l0_fw_match = "rate_limited", "Rate limit exceeded", None
    elif fw_decision == FirewallDecision.MITIGATE:
        l0_decision, l0_score, l0_triggered = "BLOCK", 1.0, True
        l0_label = str(fw_reason) if isinstance(fw_reason, str) else "pattern_match"
        l0_reason = str(fw_reason)
        l0_fw_match = l0_label
    elif fw_decision == FirewallDecision.FORWARD_TO_ML:
        susp = 0.0
        if isinstance(fw_reason, dict):
            susp = min(0.4, fw_reason.get("suspicion_score", 0) * 0.10)
        l0_decision, l0_score = "SUSPICIOUS", susp
        l0_label, l0_reason, l0_fw_match = "suspicious", str(fw_reason), None
    else:
        l0_decision, l0_score = "CLEAN", 0.0
        l0_label, l0_reason, l0_fw_match = "clean", "Normal traffic", None

    short_circuited = False
    short_circuit_at = None
    if le.get("layer0", True) and l0_triggered:
        short_circuited = True
        short_circuit_at = "layer0"

    # ── Layer 1 — Regex Filter ────────────────────────────────────
    regex_filter = RegexFilter()
    r1_decision, r1_reason = regex_filter.inspect(req_dict)

    l1_triggered = False
    if r1_decision == FirewallDecision.MITIGATE:
        l1_decision, l1_score, l1_triggered = "BLOCK", 1.0, True
        l1_label      = r1_reason.get("pattern_group", "regex_match") if isinstance(r1_reason, dict) else "regex_match"
        l1_pattern    = r1_reason.get("pattern",       None)          if isinstance(r1_reason, dict) else None
        l1_confidence = r1_reason.get("confidence",    1.0)           if isinstance(r1_reason, dict) else 1.0
        l1_group      = r1_reason.get("pattern_group", None)          if isinstance(r1_reason, dict) else None
        l1_matched_in = r1_reason.get("matched_in",    None)          if isinstance(r1_reason, dict) else None
        l1_matched_text = r1_reason.get("matched_text", None)         if isinstance(r1_reason, dict) else None
    elif r1_decision == FirewallDecision.FORWARD_TO_ML:
        conf = r1_reason.get("confidence", 0.65) if isinstance(r1_reason, dict) else 0.65
        l1_decision, l1_score = "SUSPICIOUS", regex_filter.confidence_to_score(conf)
        l1_label      = r1_reason.get("pattern_group", "regex_suspicious") if isinstance(r1_reason, dict) else "regex_suspicious"
        l1_pattern    = r1_reason.get("pattern",       None) if isinstance(r1_reason, dict) else None
        l1_confidence = conf
        l1_group      = r1_reason.get("pattern_group", None) if isinstance(r1_reason, dict) else None
        l1_matched_in = r1_reason.get("matched_in",    None) if isinstance(r1_reason, dict) else None
        l1_matched_text = r1_reason.get("matched_text", None) if isinstance(r1_reason, dict) else None
    else:
        l1_decision, l1_score = "CLEAN", 0.0
        l1_label, l1_pattern, l1_confidence = "clean", None, 0.0
        l1_group = l1_matched_in = l1_matched_text = None

    if not short_circuited and le.get("layer1", True) and l1_triggered:
        short_circuited = True
        short_circuit_at = "layer1"

    # ── Network Layer — Redis Threat Score (read-only) ────────────
    net_score = 0.0
    net_found = False
    net_attack_type = net_lgbm_score = net_ensemble_score = None
    if net_data:
        net_found = True
        net_score = float(net_data.get("score", 0.0))
        net_attack_type    = net_data.get("attack_type", None)
        net_lgbm_score     = net_data.get("net_lgbm",   None)
        net_ensemble_score = net_data.get("ensemble",   None)

    # ── Layer 2 — CNN gate → LightGBM (mirrors pipeline order) ────
    # CNN runs first. LightGBM only runs if CNN flags an anomaly.
    lgbm_ran = False
    lgbm_score_val = 0.0
    lgbm_label = None
    lgbm_all_probs: dict = {}
    lgbm_skipped_reason = None

    cnn_ran = False
    cnn_score_val = 0.0
    cnn_is_anomaly = False
    cnn_recon_score = cnn_maha_score = None
    cnn_skipped_reason = None

    if short_circuited:
        lgbm_skipped_reason = "short_circuited"
        cnn_skipped_reason  = "short_circuited"
    else:
        layer2 = getattr(request.app.state, "layer2", None)
        if layer2:
            ctx2 = RequestContext(
                ip=req.ip, method=req.method, path=req.path,
                headers=req.headers, query_params=req.query_params, body=req.body,
            )
            # Step 1: CNN Autoencoder (always runs if available)
            if layer2.cnn and layer2.cnn.is_ready():
                ls = layer2.cnn.predict(ctx2)
                cnn_ran, cnn_score_val = True, ls.score
                cnn_is_anomaly  = ls.label in {"anomaly", "zeroday"}
                cnn_recon_score = ls.metadata.get("recon_score", None)
                cnn_maha_score  = ls.metadata.get("maha_score",  None)
            else:
                cnn_skipped_reason = "model_not_ready"

            # Step 2: LightGBM — only if CNN flagged anomaly
            cnn_flagged = cnn_is_anomaly or cnn_score_val > 0.5
            if cnn_flagged:
                if layer2.lgbm and layer2.lgbm.is_ready():
                    ls = layer2.lgbm.predict(ctx2)
                    lgbm_ran, lgbm_score_val, lgbm_label = True, ls.score, ls.label
                    lgbm_all_probs = ls.metadata.get("all_probs", {})
                else:
                    lgbm_skipped_reason = "model_not_ready"
            else:
                lgbm_skipped_reason = "cnn_gate_clean"
        else:
            lgbm_skipped_reason = cnn_skipped_reason = "layer2_not_loaded"

    # ── Layer 3 — Weighted Fusion (enabled layers only) ───────────
    BASE_WEIGHTS = {
        "layer1":     WEIGHT_LAYER1_REGEX,
        "layer2_lgbm": WEIGHT_LAYER2_LGBM,
        "layer2_cnn": WEIGHT_LAYER2_CNN,
        "network":    WEIGHT_NETWORK,
    }

    if short_circuited:
        fused_score  = 1.0
        final_action = "BLOCK"
        eff_weights  = {"layer0": 1.0, "layer1": 0.0, "network": 0.0, "layer2_lgbm": 0.0, "layer2_cnn": 0.0}
    else:
        enabled_keys = [k for k in BASE_WEIGHTS if le.get(k, True)]
        total_w = sum(BASE_WEIGHTS[k] for k in enabled_keys)
        norm_w  = {k: BASE_WEIGHTS[k] / total_w for k in enabled_keys} if total_w > 0 else {}

        scores_map = {
            "layer1":     l1_score,
            "layer2_lgbm": lgbm_score_val if lgbm_ran else 0.0,
            "layer2_cnn": cnn_score_val   if cnn_ran  else 0.0,
            "network":    net_score,
        }
        fused_score = sum(norm_w.get(k, 0.0) * scores_map.get(k, 0.0) for k in BASE_WEIGHTS)
        fused_score = round(min(1.0, max(0.0, fused_score)), 4)

        if fused_score >= SCORE_BLOCK_MIN:
            final_action = "BLOCK"
        elif fused_score >= SCORE_CAPTCHA_MIN:
            final_action = "CAPTCHA"
        elif fused_score >= SCORE_THROTTLE_MIN:
            final_action = "THROTTLE"
        elif fused_score >= SCORE_DELAY_MIN:
            final_action = "DELAY"
        else:
            final_action = "ALLOW"

        eff_weights = {
            "layer0":     0.0,
            "layer1":     norm_w.get("layer1",     0.0),
            "network":    norm_w.get("network",    0.0),
            "layer2_lgbm": norm_w.get("layer2_lgbm", 0.0),
            "layer2_cnn": norm_w.get("layer2_cnn", 0.0),
        }

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # ── Log inspect result to DB queue (fire-and-forget) ──────────
    # The inspect endpoint is read-only for Redis state, but we still
    # want every scored request persisted so the dashboard shows real data.
    async def _push_inspect_event():
        try:
            r = RedisClient.get_redis()
            best_lbl = (
                lgbm_label if lgbm_ran and lgbm_label not in (None, "norm", "normal")
                else (l1_label if l1_label not in ("clean", None) else
                      (l0_label if l0_label not in ("clean", None) else "unknown"))
            )
            event = {
                "request_id":    request_id,
                "ip":            req.ip,
                "method":        req.method,
                "path":          req.path,
                "timestamp":     start_time,
                "final_score":   fused_score,
                "action":        final_action,
                "block_reason":  l0_reason if short_circuit_at == "layer0" else (
                                 l1_label  if short_circuit_at == "layer1" else ""),
                "short_circuited": short_circuited,
                "network_score": net_score,
                "latency_ms":    latency_ms,
                "best_label":    best_lbl,
                "scores": {
                    "regex_conf":     l1_score,
                    "app_lgbm_score": lgbm_score_val,
                    "net_lgbm_score": net_lgbm_score or 0.0,
                    "deep_anomaly":   cnn_score_val,
                    "final_risk":     fused_score,
                },
                "layer_scores": [
                    {"layer": "layer0_firewall", "score": l0_score,  "label": l0_label,
                     "confidence": l0_score, "triggered": l0_triggered, "metadata": {}},
                    {"layer": "layer1_regex",    "score": l1_score,  "label": l1_label,
                     "confidence": l1_confidence, "triggered": l1_triggered,
                     "metadata": {"pattern_group": l1_group, "matched_in": l1_matched_in}},
                    {"layer": "layer2_lgbm",     "score": lgbm_score_val, "label": lgbm_label or "norm",
                     "confidence": lgbm_score_val, "triggered": lgbm_score_val > 0.5, "metadata": {}},
                    {"layer": "layer2_cnn",      "score": cnn_score_val, "label": "anomaly" if cnn_is_anomaly else "norm",
                     "confidence": cnn_score_val, "triggered": cnn_is_anomaly, "metadata": {}},
                    {"layer": "network",         "score": net_score, "label": net_attack_type or "clean",
                     "confidence": net_score, "triggered": net_score > 0.5, "metadata": {}},
                ],
                "source": "inspector",
            }
            raw = json.dumps(event)
            pipe = r.raw.pipeline(transaction=False)
            pipe.lpush("admin:events", raw)
            pipe.ltrim("admin:events", 0, 9999)
            pipe.expire("admin:events", 86400)
            pipe.lpush("db:queue", raw)
            pipe.ltrim("db:queue", 0, 49999)
            pipe.execute()
        except Exception as e:
            logger.debug("[Inspect] DB push failed: %s", e)

    asyncio.ensure_future(_push_inspect_event())

    result = {
        "request_id":      request_id,
        "latency_ms":      latency_ms,
        "final_action":    final_action,
        "final_score":     fused_score,
        "short_circuited": short_circuited,
        "short_circuit_at": short_circuit_at,
        "layers": {
            "layer0": {
                "enabled":  le.get("layer0", True),
                "ran":      True,
                "decision": l0_decision,
                "score":    l0_score,
                "label":    l0_label,
                "reason":   l0_reason,
                "detail": {
                    "blacklisted":    is_blacklisted,
                    "rate_limited":   l0_rate_limited,
                    "rate_count":     rate_count,
                    "firewall_match": l0_fw_match,
                },
            },
            "layer1": {
                "enabled":      le.get("layer1", True),
                "ran":          True,
                "decision":     l1_decision,
                "score":        l1_score,
                "label":        l1_label,
                "pattern":      l1_pattern,
                "confidence":   l1_confidence,
                "group":        l1_group,
                "matched_in":   l1_matched_in,
                "matched_text": l1_matched_text,
            },
            "network": {
                "enabled":        le.get("network", True),
                "ran":            True,
                "found":          net_found,
                "score":          net_score,
                "attack_type":    net_attack_type,
                "lgbm_score":     net_lgbm_score,
                "ensemble_score": net_ensemble_score,
            },
            "layer2_lgbm": {
                "enabled":        le.get("layer2_lgbm", True),
                "ran":            lgbm_ran,
                "skipped_reason": lgbm_skipped_reason,
                "score":          lgbm_score_val,
                "label":          lgbm_label,
                "all_probs":      lgbm_all_probs,
            },
            "layer2_cnn": {
                "enabled":        le.get("layer2_cnn", True),
                "ran":            cnn_ran,
                "skipped_reason": cnn_skipped_reason,
                "score":          cnn_score_val,
                "is_anomaly":     cnn_is_anomaly,
                "recon_score":    cnn_recon_score,
                "maha_score":     cnn_maha_score,
            },
            "layer3": {
                "ran":        True,
                "fused_score": fused_score,
                "action":     final_action,
                "weights":    eff_weights,
            },
        },
    }
    return result

@app.get("/status")
async def status():
    redis_ok = False
    redis_stats = {}
    try:
        r = RedisClient.get_redis()
        redis_ok = r.ping()
        redis_stats = r.get_stats()
    except Exception:
        pass

    layer2 = getattr(app.state, "layer2",None)
    layer2_status = layer2.status() if layer2 else {
        "lgbm_ready" : False,
        "cnn_ready" : False,
    }

    return {
        "status": "ok",
        "pipeline": {
            "layer0_firewall":  True,
            "layer1_regex":     True,
            "layer2_lgbm":      layer2_status["lgbm_ready"],
            "layer2_cnn":       layer2_status["cnn_ready"],
            "layer3_response":  True,
            "redis":            redis_ok,
            "network_ips":      redis_ok,  # network layer writes to Redis
        },
        "models": {
            "lgbm": {
                "ready":    layer2_status["lgbm_ready"],
                "path":     str(LGBM_MODEL_DIR),
                "classes":  ["sqli", "xss", "cmdi", "path-traversal", "norm"],
                "features": "ThreatFeatureExtractor (70+)",
            },
            "cnn": {
                "ready":    layer2_status["cnn_ready"],
                "path":     str(CNN_MODEL_DIR),
                "classes":  ["norm", "anom"],
                "features": "extract_payload_features (15)",
            },
        },
        "redis": redis_stats,
    }


# ── Admin / Stats helpers ──────────────────────────────────────────────────────

from datetime import datetime, timezone
from fastapi import Query


def _read_admin_events() -> list:
    try:
        r = RedisClient.get_redis()
        raw_list = r.raw.lrange("admin:events", 0, 9999)
        if not raw_list:
            return []
        result = []
        for raw in raw_list:
            try:
                result.append(json.loads(raw))
            except Exception:
                pass
        return result
    except Exception:
        return []


def _compute_stats(events: list) -> dict:
    if not events:
        return None
    now = time.time()
    day_ago = now - 86400

    today = [e for e in events if e.get("timestamp", 0) >= day_ago]
    total  = len(today)
    blocked   = sum(1 for e in today if e.get("action") == "BLOCK")
    allowed   = sum(1 for e in today if e.get("action") == "ALLOW")
    throttled = sum(1 for e in today if e.get("action") in ("THROTTLE", "CAPTCHA", "DELAY"))
    block_rate = round(blocked / total * 100, 1) if total else 0
    latencies = [e.get("latency_ms", 0) for e in today if e.get("latency_ms")]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0

    # Requests per minute — last 30 minutes (30 buckets)
    rpm_buckets = [0] * 30
    rpm_blocked = [0] * 30
    for e in events:
        mins_ago = (now - e.get("timestamp", now)) / 60
        if 0 <= mins_ago < 30:
            idx = int(mins_ago)
            rpm_buckets[29 - idx] += 1
            if e.get("action") == "BLOCK":
                rpm_blocked[29 - idx] += 1

    # Attack type distribution
    attack_counts: dict = {}
    for e in today:
        label = e.get("block_reason") or ""
        for ls in e.get("layer_scores", []):
            if ls.get("triggered") or ls.get("score", 0) > 0.5:
                lbl = ls.get("label") or label
                if lbl and lbl not in ("clean","norm","normal",""):
                    attack_counts[lbl] = attack_counts.get(lbl, 0) + 1
                    break

    # Layer distribution
    layer_counts: dict = {}
    for e in today:
        for ls in e.get("layer_scores", []):
            if ls.get("triggered"):
                lyr = ls.get("layer","unknown")
                layer_counts[lyr] = layer_counts.get(lyr, 0) + 1
                break

    # Top threat IPs
    ip_stats: dict = {}
    for e in today:
        ip = e.get("ip","")
        if not ip:
            continue
        if ip not in ip_stats:
            ip_stats[ip] = {"ip": ip, "requests": 0, "blocked": 0,
                            "max_score": 0.0, "last_attack": 0}
        ip_stats[ip]["requests"] += 1
        if e.get("action") == "BLOCK":
            ip_stats[ip]["blocked"] += 1
        sc = e.get("final_score", 0)
        if sc > ip_stats[ip]["max_score"]:
            ip_stats[ip]["max_score"] = sc
            ip_stats[ip]["last_attack"] = e.get("timestamp", 0)

    top_ips = sorted(ip_stats.values(), key=lambda x: x["blocked"], reverse=True)[:20]
    for rec in top_ips:
        rec["max_score"] = round(rec["max_score"], 4)
        rec["last_attack_iso"] = (
            datetime.fromtimestamp(rec["last_attack"], tz=timezone.utc).isoformat()
            if rec["last_attack"] else ""
        )

    return {
        "total_requests":  total,
        "blocked_attacks": blocked,
        "allowed_requests": allowed,
        "throttled_captcha": throttled,
        "block_rate_pct":  block_rate,
        "avg_latency_ms":  avg_latency,
        "rpm_total":   rpm_buckets,
        "rpm_blocked": rpm_blocked,
        "attack_types":  attack_counts,
        "layer_counts":  layer_counts,
        "top_ips":       top_ips,
    }


# ── /api/stats ─────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    # 1. Try PostgreSQL (real persistent data)
    stats = await db_reader.fetch_stats()
    if stats:
        stats["source"] = "postgresql"
        return stats

    # 2. Fall back to Redis in-memory list
    events = _read_admin_events()
    stats  = _compute_stats(events)
    if stats:
        stats["source"] = "redis"
        return stats

    # No data yet — return empty stats
    return {
        "total_requests": 0, "blocked_attacks": 0,
        "allowed_requests": 0, "throttled_captcha": 0,
        "block_rate_pct": 0, "avg_latency_ms": 0,
        "rpm_total": [0] * 30, "rpm_blocked": [0] * 30,
        "attack_types": {}, "layer_counts": {}, "top_ips": [],
        "source": "empty",
    }


# ── /api/events ────────────────────────────────────────────────────────────────

@app.get("/api/events")
async def api_events(
    limit:       int = Query(100, le=500),
    action:      str = Query(""),
    attack_type: str = Query(""),
    ip:          str = Query(""),
):
    # 1. Try PostgreSQL
    events = await db_reader.fetch_events(
        limit=limit, action=action, ip=ip, attack_type=attack_type
    )

    # 2. Fall back to Redis
    if events is None:
        events = _read_admin_events()
        if action:
            events = [e for e in events if e.get("action", "").upper() == action.upper()]
        if ip:
            events = [e for e in events if e.get("ip", "") == ip]
        if attack_type:
            def _has_type(e):
                if attack_type.lower() in (e.get("block_reason", "") or "").lower():
                    return True
                for ls in e.get("layer_scores", []):
                    if attack_type.lower() in (ls.get("label", "") or "").lower():
                        return True
                return False
            events = [e for e in events if _has_type(e)]
        events = events[:limit]

    # Ensure best_label is set on every event
    for e in events:
        if not e.get("best_label"):
            for ls in e.get("layer_scores", []):
                lbl = ls.get("label", "")
                if lbl and lbl not in ("clean", "norm", "normal"):
                    e["best_label"] = lbl
                    break
            else:
                e["best_label"] = e.get("block_reason") or (
                    "clean" if e.get("action") == "ALLOW" else "unknown"
                )

    return {"events": events, "total": len(events)}


# ── /api/admin/block-ip ────────────────────────────────────────────────────────

class _BlockIPRequest(BaseModel):
    ip: str
    reason: str = "Manually blocked via admin dashboard"
    permanent: bool = False


@app.post("/api/admin/block-ip")
async def admin_block_ip(req: _BlockIPRequest):
    try:
        r = RedisClient.get_redis()
        r.blacklist_ip(req.ip, reason=req.reason, permanent=req.permanent)
        return {"ok": True, "ip": req.ip, "message": f"{req.ip} added to blacklist"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


# ── /api/admin/blocked-ips ─────────────────────────────────────────────────────

@app.get("/api/admin/blocked-ips")
async def admin_blocked_ips():
    try:
        r = RedisClient.get_redis()
        keys = r.raw.keys("blacklist:ip:*")
        result = []
        for k in keys:
            ip = k.replace("blacklist:ip:", "")
            entry = r.get_blacklist_entry(ip) or {}
            result.append({
                "ip":        ip,
                "reason":    entry.get("reason", ""),
                "permanent": entry.get("permanent", False),
                "timestamp": entry.get("timestamp", 0),
            })
        return {"blocked_ips": result, "count": len(result)}
    except Exception as e:
        return {"blocked_ips": [], "count": 0, "error": str(e)}
