import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse

from pipeline.application_level.layer2 import Layer2MLOrchestrator
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
    skip_paths=["/health", "/docs", "/openapi.json", "/redoc", "/status", "/demo"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/probe")
async def probe(request: Request):
    """Target endpoint for the demo page — middleware scores every request to this."""
    return {"ok": True, "path": str(request.url.path)}


@app.get("/demo", response_class=HTMLResponse)
async def demo():
    return HTMLResponse(content=_DEMO_HTML)


_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AIM-IPS Live Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 32px; }
  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: #64748b; font-size: 0.9rem; margin-bottom: 32px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; max-width: 1100px; }
  .card { background: #1e2130; border-radius: 12px; padding: 20px; border: 1px solid #2d3148; }
  .card h2 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 14px; }
  .btn-group { display: flex; flex-direction: column; gap: 8px; }
  button {
    padding: 10px 16px; border: none; border-radius: 8px; font-size: 0.88rem;
    font-weight: 600; cursor: pointer; text-align: left; transition: opacity 0.15s;
  }
  button:hover { opacity: 0.85; }
  .btn-clean   { background: #1d4ed8; color: #fff; }
  .btn-sqli    { background: #7c3aed; color: #fff; }
  .btn-xss     { background: #b45309; color: #fff; }
  .btn-path    { background: #b91c1c; color: #fff; }
  .btn-cmdi    { background: #0f766e; color: #fff; }
  .btn-ua      { background: #6b21a8; color: #fff; }
  .btn-zeroday { background: #9f1239; color: #fff; }
  .result-panel { grid-column: 1 / -1; background: #1e2130; border-radius: 12px; padding: 20px; border: 1px solid #2d3148; min-height: 160px; }
  .result-panel h2 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 14px; }
  .badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 1rem; font-weight: 700; margin-bottom: 12px;
  }
  .badge-BLOCK    { background: #7f1d1d; color: #fca5a5; }
  .badge-ALLOW    { background: #14532d; color: #86efac; }
  .badge-CAPTCHA  { background: #713f12; color: #fcd34d; }
  .badge-THROTTLE { background: #1e3a5f; color: #93c5fd; }
  .badge-DELAY    { background: #312e81; color: #c4b5fd; }
  .badge-PENDING  { background: #1e2130; color: #64748b; }
  .score-bar-wrap { background: #0f1117; border-radius: 6px; height: 10px; margin-bottom: 14px; overflow: hidden; }
  .score-bar { height: 10px; border-radius: 6px; transition: width 0.4s; }
  table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
  td { padding: 6px 10px; border-bottom: 1px solid #2d3148; }
  td:first-child { color: #64748b; width: 40%; }
  td:last-child  { font-family: monospace; word-break: break-all; }
  .spinner { display: none; width: 18px; height: 18px; border: 3px solid #2d3148; border-top-color: #6366f1; border-radius: 50%; animation: spin 0.7s linear infinite; margin-left: auto; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .status-dots { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
  .dot { font-size: 0.78rem; display: flex; align-items: center; gap: 5px; }
  .dot span { width: 8px; height: 8px; border-radius: 50%; background: #64748b; }
  .dot.ok span  { background: #22c55e; }
  .dot.bad span { background: #ef4444; }
</style>
</head>
<body>

<h1>AIM-IPS &nbsp;·&nbsp; Live Demo</h1>
<p class="subtitle">AI-Powered Multi-Layer Intrusion Prevention System</p>

<div id="statusDots" class="status-dots">
  <div class="dot" id="dot-app"><span></span>App Layer</div>
  <div class="dot" id="dot-lgbm"><span></span>LightGBM</div>
  <div class="dot" id="dot-cnn"><span></span>CNN AE</div>
  <div class="dot" id="dot-redis"><span></span>Redis</div>
  <div class="dot" id="dot-net"><span></span>Network IPS</div>
</div>

<div class="grid">

  <div class="card">
    <h2>Clean Traffic</h2>
    <div class="btn-group">
      <button class="btn-clean" onclick="probe('GET', '', {})">GET /api/probe — normal request</button>
      <button class="btn-clean" onclick="probe('POST', '{&quot;username&quot;:&quot;alice&quot;,&quot;page&quot;:1}', {})">POST — valid JSON body</button>
    </div>
  </div>

  <div class="card">
    <h2>SQL Injection</h2>
    <div class="btn-group">
      <button class="btn-sqli" onclick="probe('POST', &quot;username=admin'--&password=x&quot;, {})">Classic OR comment bypass</button>
      <button class="btn-sqli" onclick="probe('GET', '', {'q': &quot;' UNION SELECT NULL,username,password FROM users--&quot;})">UNION SELECT via query param</button>
      <button class="btn-sqli" onclick="probe('GET', '', {'id': '1; SELECT SLEEP(5)--'})">Time-based blind injection</button>
    </div>
  </div>

  <div class="card">
    <h2>Cross-Site Scripting</h2>
    <div class="btn-group">
      <button class="btn-xss" onclick="probe('POST', '<script>alert(document.cookie)</script>', {})">Script tag injection</button>
      <button class="btn-xss" onclick="probe('POST', '<img src=x onerror=alert(1)>', {})">onerror event handler</button>
      <button class="btn-xss" onclick="probe('GET', '', {'url': 'javascript:alert(1)'})">javascript: protocol</button>
    </div>
  </div>

  <div class="card">
    <h2>Path Traversal &amp; Command Injection</h2>
    <div class="btn-group">
      <button class="btn-path" onclick="probe('GET', '', {'file': '../../../etc/passwd'})">../../../etc/passwd</button>
      <button class="btn-path" onclick="probe('GET', '', {'path': '%2e%2e%2fetc%2fpasswd'})">URL-encoded traversal</button>
      <button class="btn-cmdi" onclick="probe('POST', 'host=127.0.0.1; cat /etc/passwd', {})">Semicolon + cat command</button>
    </div>
  </div>

  <div class="card">
    <h2>Scanner / Bad User-Agent</h2>
    <div class="btn-group">
      <button class="btn-ua" onclick="probeUA('sqlmap/1.7')">sqlmap scanner</button>
      <button class="btn-ua" onclick="probeUA('Nikto/2.1.6')">Nikto scanner</button>
    </div>
  </div>

  <div class="card">
    <h2>Zero-Day / Anomaly</h2>
    <div class="btn-group">
      <button class="btn-zeroday" onclick="probe('POST', btoa('AAAA'.repeat(200)), {})">Base64-encoded large payload</button>
      <button class="btn-zeroday" onclick="probe('POST', JSON.stringify({a:Array(80).fill('x').join('')}), {})">Unusual structure — ML layer</button>
    </div>
  </div>

  <div class="result-panel">
    <h2>Result &nbsp;<div class="spinner" id="spinner"></div></h2>
    <div id="badge" class="badge badge-PENDING">—</div>
    <div class="score-bar-wrap"><div class="score-bar" id="scoreBar" style="width:0%;background:#6366f1"></div></div>
    <table id="resultTable">
      <tr><td>Action</td><td id="r-action">—</td></tr>
      <tr><td>Fused Score</td><td id="r-score">—</td></tr>
      <tr><td>Pipeline Latency</td><td id="r-latency">—</td></tr>
      <tr><td>HTTP Status</td><td id="r-status">—</td></tr>
      <tr><td>Reason / Label</td><td id="r-reason">—</td></tr>
      <tr><td>Request ID</td><td id="r-rid">—</td></tr>
      <tr><td>Layer triggered</td><td id="r-layer">—</td></tr>
    </table>
  </div>

</div>

<script>
async function loadStatus() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    const p = d.pipeline || {};
    setDot('dot-app',   true);
    setDot('dot-lgbm',  p.layer2_lgbm);
    setDot('dot-cnn',   p.layer2_cnn);
    setDot('dot-redis', p.redis);
    setDot('dot-net',   p.network_ips);
  } catch(e) { setDot('dot-app', false); }
}

function setDot(id, ok) {
  const el = document.getElementById(id);
  el.className = 'dot ' + (ok ? 'ok' : 'bad');
}

async function probe(method, body, params) {
  const spinner = document.getElementById('spinner');
  spinner.style.display = 'inline-block';

  let url = '/api/probe';
  if (Object.keys(params).length) {
    url += '?' + new URLSearchParams(params).toString();
  }

  const opts = { method, headers: {} };
  if (method === 'POST' && body) {
    opts.body = body;
    opts.headers['Content-Type'] = 'text/plain';
  }

  const t0 = performance.now();
  try {
    const resp = await fetch(url, opts);
    const latencyMs = (performance.now() - t0).toFixed(1);
    const action   = resp.headers.get('X-IPS-Action') || (resp.status === 403 ? 'BLOCK' : resp.status === 429 ? 'CAPTCHA' : 'ALLOW');
    const score    = resp.headers.get('X-IPS-Score')  || '—';
    const rid      = resp.headers.get('X-IPS-RequestID') || '—';
    let   bodyData = {};
    try { bodyData = await resp.json(); } catch(_) {}

    const reason = bodyData.reason || bodyData.error || '—';
    const scoreNum = parseFloat(score) || 0;

    document.getElementById('badge').textContent  = action;
    document.getElementById('badge').className    = 'badge badge-' + action;
    document.getElementById('r-action').textContent  = action;
    document.getElementById('r-score').textContent   = score;
    document.getElementById('r-latency').textContent = latencyMs + ' ms';
    document.getElementById('r-status').textContent  = resp.status + ' ' + resp.statusText;
    document.getElementById('r-reason').textContent  = reason;
    document.getElementById('r-rid').textContent     = rid;
    document.getElementById('r-layer').textContent   = bodyData.label || (resp.status === 403 ? 'Layer 0/1 (hard block)' : 'Layer 3 (score fusion)');

    const bar = document.getElementById('scoreBar');
    bar.style.width = (scoreNum * 100) + '%';
    bar.style.background = scoreNum >= 0.7 ? '#ef4444' : scoreNum >= 0.5 ? '#f59e0b' : scoreNum >= 0.35 ? '#6366f1' : '#22c55e';

  } catch(e) {
    document.getElementById('r-latency').textContent = (performance.now() - t0).toFixed(1) + ' ms';
    document.getElementById('r-reason').textContent  = 'Request failed: ' + e.message;
  } finally {
    spinner.style.display = 'none';
  }
}

async function probeUA(ua) {
  const spinner = document.getElementById('spinner');
  spinner.style.display = 'inline-block';
  const t0 = performance.now();
  try {
    const resp = await fetch('/api/probe', { headers: { 'User-Agent': ua } });
    const latencyMs = (performance.now() - t0).toFixed(1);
    const action = resp.headers.get('X-IPS-Action') || (resp.status === 403 ? 'BLOCK' : 'ALLOW');
    const score  = resp.headers.get('X-IPS-Score') || '—';
    const rid    = resp.headers.get('X-IPS-RequestID') || '—';
    let   bodyData = {};
    try { bodyData = await resp.json(); } catch(_) {}

    document.getElementById('badge').textContent     = action;
    document.getElementById('badge').className       = 'badge badge-' + action;
    document.getElementById('r-action').textContent  = action;
    document.getElementById('r-score').textContent   = score;
    document.getElementById('r-latency').textContent = latencyMs + ' ms';
    document.getElementById('r-status').textContent  = resp.status;
    document.getElementById('r-reason').textContent  = bodyData.reason || '—';
    document.getElementById('r-rid').textContent     = rid;
    document.getElementById('r-layer').textContent   = 'Layer 0 (static firewall — bad UA)';

    const scoreNum = parseFloat(score) || 0;
    const bar = document.getElementById('scoreBar');
    bar.style.width = (scoreNum * 100 || (resp.status === 403 ? 100 : 0)) + '%';
    bar.style.background = resp.status === 403 ? '#ef4444' : '#22c55e';
  } catch(e) {
    document.getElementById('r-latency').textContent = (performance.now() - t0).toFixed(1) + ' ms';
    document.getElementById('r-reason').textContent  = e.message;
  } finally {
    spinner.style.display = 'none';
  }
}

loadStatus();
</script>
</body>
</html>"""

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







