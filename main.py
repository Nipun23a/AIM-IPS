import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pipeline.layer2 import Layer2MLOrchestrator
from api.middleware import IPSMiddleware
from utils.redis_client import RedisClient

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

    logger.info("[Startup] AIM-IPS Ready ✓")
    logger.info("=" * 60)

    yield

    logger.info("[Shutdown] AIM-IPS shutting down")


app = FastAPI(
    title="AIM-IPS",
    description="AI-Powered Intrusion Prevention System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    IPSMiddleware,
    skip_paths=["/health", "/docs", "/openapi.json", "/redoc", "/status"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

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







