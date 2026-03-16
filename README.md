# AIM-IPS — AI-Powered Intrusion Prevention System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-19-cyan?style=flat-square&logo=react)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-blue?style=flat-square&logo=postgresql)
![Redis](https://img.shields.io/badge/Redis-6+-red?style=flat-square&logo=redis)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-orange?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?style=flat-square&logo=tensorflow)

**A modular, multi-layer AI-driven Intrusion Prevention System that analyses every HTTP request through a 5-stage machine learning pipeline — detecting, scoring, and blocking cyber threats in real time.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Detection Pipeline](#detection-pipeline)
- [Network Layer IPS](#network-layer-ips)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Database Setup](#database-setup)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Frontend Pages](#frontend-pages)
- [Attack Presets](#attack-presets)
- [Redis Key Schema](#redis-key-schema)
- [Scoring & Thresholds](#scoring--thresholds)
- [Performance](#performance)

---

## Overview

**AIM-IPS** (Adaptive Intelligent Machine Learning-based Intrusion Prevention System) is a research-grade web security middleware that sits in front of your application and analyses every HTTP request through a multi-stage AI pipeline.

Unlike traditional WAFs that rely solely on static rules, AIM-IPS combines:

- **Static firewall rules** — instant IP blacklist, rate limiting, bad user-agent detection
- **Regex pattern matching** — high-confidence attack signature detection
- **Network packet analysis** — raw traffic flow classification via Scapy + ML
- **CNN Autoencoder** — zero-day anomaly detection via reconstruction error
- **LightGBM Classifier** — supervised classification of known attack categories
- **Weighted fusion engine** — all layer scores combined into a single graduated response

Every request is scored, logged to PostgreSQL, and visible in the real-time admin dashboard.

---

## Architecture

```
                     ┌─────────────────────────────────────┐
                     │          Incoming HTTP Request        │
                     └──────────────────┬──────────────────┘
                                        │
                     ┌──────────────────▼──────────────────┐
                     │         IPSMiddleware (FastAPI)       │
                     │                                       │
                     │  Redis Bulk Read (1 round-trip)       │
                     │  • blacklist:ip:{ip}                  │
                     │  • ratelimit:ip:{ip}                  │
                     │  • threat:ip:{ip}  ◄──────────────┐  │
                     │                                    │  │
                     │  Layer 0 → Layer 1 → Layer 2 → L3 │  │
                     │                                    │  │
                     │  _log_async() → db:queue ──────────┘  │
                     └───────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────┼─────────────────────────┐
  │  Network Layer IPS (separate sudo   │  DB Writer (async task)  │
  │  process)                           │                          │
  │                                     │  RPOP db:queue           │
  │  Scapy → Flows → LightGBM + TCN     │    → batch INSERT        │
  │    → Redis SETEX threat:ip:{ip}     │      → PostgreSQL        │
  └─────────────────────────────────────┴─────────────────────────┘
```

---

## Detection Pipeline

### Layer 0 — Static Firewall
**File:** `firewall/engine.py`

Fastest line of defense — zero ML inference, pure in-memory checks.

| Check | Action |
|---|---|
| IP in Redis blacklist | BLOCK immediately |
| > 100 requests/minute | BLOCK (rate limit) |
| Bad User-Agent (sqlmap, nikto, curl, python) | BLOCK |
| Hard-coded SQLi / XSS / path traversal patterns | BLOCK |
| Suspicious heuristics (encoding, long payload, special chars) | FORWARD_TO_ML |

### Layer 1 — Regex Filter
**File:** `firewall/regex_filter.py`

Pattern-based detection across URL, headers, body, and query params.

| Attack Type | Examples |
|---|---|
| SQL Injection | `UNION SELECT`, `' OR 1=1`, `SLEEP(5)`, stacked queries |
| XSS | `<script>`, `onerror=`, `javascript:`, SVG injections |
| Path Traversal | `../`, `/etc/passwd`, `/proc/`, null bytes, encoded variants |
| Command Injection | `;`, `\|`, `&&`, backtick substitution, reverse shells |
| XXE / LFI / RFI | XML entity injection, file inclusion |
| Malicious UA | Scanner and exploit framework fingerprints |

Confidence scoring:
- **HIGH (≥ 0.85)** → `MITIGATE` — score = 1.0, short-circuit, no ML needed
- **MEDIUM (0.65–0.85)** → `FORWARD_TO_ML` — score = confidence value

### Layer 2b — CNN Autoencoder (Anomaly Gate)
**File:** `anomly_detector/cnn_detector.py`

Runs **before** LightGBM and acts as a gate for zero-day detection.

- **Reconstruction error (50%)** + **Mahalanobis distance (50%)**
- Score > 0.5 or label ∈ `{anomaly, zeroday}` → triggers LightGBM
- Score clean → LightGBM **skipped** (eliminates false positives on benign traffic)

### Layer 2a — LightGBM Classifier
**File:** `threat_classifier/lgbm_classifier.py`

Only runs when the CNN gate flags an anomaly.

- **70+ features** extracted per request (entropy, length, special char ratio, encoding depth)
- **Classes:** `sqli`, `xss`, `cmdi`, `path-traversal`, `norm`
- Sub-millisecond inference on CPU

### Layer 3 — Weighted Fusion Engine
**File:** `response/engine.py`

All layer scores fused into a single threat score:

```
fused_score = 0.20 × regex_score
            + 0.35 × lgbm_score
            + 0.25 × cnn_score
            + 0.20 × network_score
```

| Fused Score | Action | Effect |
|---|---|---|
| `< 0.25` | **ALLOW** | Request passes through |
| `0.25 – 0.35` | **DELAY** | 1.5s async sleep, then forward |
| `0.35 – 0.50` | **THROTTLE** | 3.0s async sleep, then forward |
| `0.50 – 0.70` | **CAPTCHA** | 429 challenge required |
| `≥ 0.70` | **BLOCK** | 403 Forbidden + IP auto-blacklisted (1h) |

---

## Network Layer IPS

**File:** `pipeline/network_level/network_ips.py`

Runs as a **completely separate process** — completely decoupled from the HTTP pipeline. Requires `sudo` for raw packet capture via Scapy.

### How It Works

```
Raw Packets (eth0)
    └─► FlowAccumulator        builds per-IP flows from raw packets
            └─► LightGBM       classifies known attacks (DDoS, portscan, botnet)
            └─► TCN Ensemble   detects zero-day network anomalies
                    └─► Redis SETEX  threat:ip:{ip}  TTL=60s
                                        └─► Middleware reads on every request (<0.1ms)
```

### What It Detects

| Threat | Model | Signature |
|---|---|---|
| DDoS / SYN Flood | LightGBM | High packet rate, SYN flag spike |
| Port Scan | LightGBM | Many small flows, varied destination ports |
| Botnet C2 | LightGBM | Periodic beaconing, unusual flow timing |
| Zero-Day Network | TCN Ensemble | Anomaly score from AE + VAE + IsolationForest |

Internal fusion: `network_score = 0.55 × lgbm + 0.45 × tcn_ensemble`

### Running the Network Layer

```bash
# Live capture on VPS (requires sudo)
sudo python -m pipeline.network_level.network_ips --interface eth0

# With debug logging
sudo python -m pipeline.network_level.network_ips --interface eth0 --debug

# Simulation mode — no root required, synthetic flows
python -m pipeline.network_level.network_ips --simulate
```

> If not running, `network_score` defaults to `0.0` — the other 4 layers still provide full protection.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | FastAPI 0.95+ |
| ASGI Server | Uvicorn 0.22+ |
| App-Layer ML | LightGBM (classification) + TensorFlow CNN (anomaly) |
| Network-Layer ML | LightGBM + TCN (TFLite) + Ensemble (AE, VAE, IsolationForest) |
| Packet Capture | Scapy |
| Cache / State | Redis 6+ |
| Persistent Storage | PostgreSQL 14+ via asyncpg |
| Frontend Framework | React 19 + Vite 8 |
| UI Styling | TailwindCSS 4 + Lucide React |
| Charts | Chart.js 4 + react-chartjs-2 |
| Visualization | globe.gl + react-simple-maps |
| Routing | React Router 7 |
| Validation | Pydantic |

---

## Project Structure

```
AIM-IPS/
├── main.py                          # FastAPI entry point + all API routes
├── .env                             # Environment variables
├── requirements.txt                 # Python dependencies
│
├── api/
│   └── middleware.py                # IPSMiddleware — orchestrates all layers
│
├── firewall/
│   ├── engine.py                    # Layer 0: Static Firewall
│   ├── regex_filter.py              # Layer 1: Regex Attack Filter
│   ├── rules.py                     # Attack pattern definitions
│   └── decisions.py                 # FirewallDecision enum
│
├── pipeline/
│   ├── application_level/
│   │   └── layer2.py                # Layer 2: CNN + LightGBM Orchestrator
│   └── network_level/
│       ├── network_ips.py           # Network IPS entry point (run with sudo)
│       ├── flow_acuumulator.py      # Packet → Flow feature builder
│       ├── network_classifier.py    # LightGBM + Ensemble fusion
│       ├── tcn_detector.py          # Temporal Convolutional Network
│       ├── ensemble_detector.py     # AE + VAE + IsolationForest
│       └── feature.py               # Network feature definitions
│
├── threat_classifier/
│   └── lgbm_classifier.py           # LightGBM app-layer classifier
│
├── anomly_detector/
│   └── cnn_detector.py              # CNN Autoencoder (zero-day gate)
│
├── response/
│   └── engine.py                    # Layer 3: Weighted fusion + action
│
├── shared/
│   ├── constants.py                 # All thresholds, weights, TTLs, Redis keys
│   └── schemas.py                   # RequestContext, LayerScore dataclasses
│
├── utils/
│   ├── redis_client.py              # RedisClient wrapper
│   ├── redis_threat_store.py        # Network threat score schema
│   └── request_normalizer.py
│
├── db/
│   ├── schema.sql                   # PostgreSQL DDL — run once to set up
│   ├── writer.py                    # Background async DB writer (Redis → PostgreSQL)
│   └── reader.py                    # Async query functions for API endpoints
│
├── models/
│   ├── application_layer/           # LightGBM app classifier weights
│   ├── anomaly_detector/            # CNN Autoencoder weights
│   └── threat_classifier/           # LightGBM threat model
│
└── frontend/
    ├── src/
    │   ├── pages/
    │   │   ├── HomePage.jsx          # Landing page + architecture overview
    │   │   ├── AdminDashboardPage.jsx # Real-time dashboard
    │   │   ├── InspectorPage.jsx     # Layer-by-layer pipeline inspector
    │   │   └── LoginPage.jsx
    │   ├── components/
    │   │   ├── dashboard/            # Charts, globe, event log, threat IPs
    │   │   ├── inspector/            # Per-layer result cards
    │   │   └── common/               # Shared UI components
    │   └── constants/
    │       └── index.js              # Attack presets, layer weights, admin creds
    └── package.json
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL 14+
- Redis 6+

### 1. Install Dependencies

```bash
# Backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 2. Configure Environment

```bash
# Edit .env with your database credentials
nano .env
```

### 3. Setup Database

```bash
psql -U postgres -c "CREATE DATABASE aimips;"
psql -U postgres -d aimips -f db/schema.sql
```

### 4. Start Redis

```bash
redis-server
# or with Docker:
docker run -d -p 6379:6379 redis:latest
```

### 5. Start Everything

```bash
# Terminal 1 — Backend IPS (port 8000)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend dashboard (port 5173)
cd frontend && npm run dev

# Terminal 3 — Network Layer (optional, requires sudo)
sudo python -m pipeline.network_level.network_ips --interface eth0
```

---

## Environment Variables

**Backend** (`.env` in project root):

```env
# PostgreSQL — format: postgresql://USER:PASSWORD@HOST:PORT/DATABASE
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/aimips

# Redis
REDIS_URL=redis://localhost:6379/0

# Trusted Proxy IPs — IPs allowed to set X-Forwarded-For header
# Set to your Nginx/load-balancer IP on VPS to prevent IP spoofing
# Leave as 127.0.0.1 for direct deployment without a reverse proxy
TRUSTED_PROXIES=127.0.0.1
```

**Frontend** (`frontend/.env`):

```env
VITE_APP_SERVER_LAT=6.9271          # Server latitude for globe visualization
VITE_APP_SERVER_LNG=79.8612         # Server longitude
VITE_ADMIN_USERNAME=admin
VITE_ADMIN_PASSWORD=aimips2024
```

---

## Database Setup

AIM-IPS uses a **decoupled write pattern** — the pipeline never writes to the database directly. Instead:

1. `_log_async()` pushes events to Redis `db:queue` (one LPUSH, ~0.1ms, fire-and-forget)
2. A background asyncio task (`db/writer.py`) pops events and batch-inserts into PostgreSQL every 5 seconds or 50 events

This means **zero DB latency** in the request path.

### API Data Priority

```
1. PostgreSQL  →  persistent, survives restarts, full SQL analytics
2. Redis       →  in-memory fallback (24h TTL, 10k events)
3. Empty       →  returns zero stats if no data yet
```

### Pre-built Analytics Views

| View | Description |
|---|---|
| `v_events_24h` | All events from the last 24 hours |
| `v_rpm_30min` | Requests-per-minute buckets for the last 30 minutes |
| `v_top_ips` | Top threat IPs ranked by block count |
| `v_attack_types` | Attack category distribution |
| `v_layer_counts` | Detection layer breakdown |

---

## Running the System

### Backend

```bash
# Development (auto-reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend

```bash
cd frontend
npm run dev        # Development — http://localhost:5173
npm run build      # Production build
npm run preview    # Preview production build
```

### Network Layer IPS

```bash
# Live capture — requires root (Scapy raw socket access)
sudo python -m pipeline.network_level.network_ips --interface eth0

# Debug mode
sudo python -m pipeline.network_level.network_ips --interface eth0 --debug

# Simulation — no root needed, injects synthetic flows for testing
python -m pipeline.network_level.network_ips --simulate
```

### VPS Deployment

On a VPS, configure the trusted proxy so real attacker IPs are captured:

```env
# In .env — set this to your Nginx server's IP
TRUSTED_PROXIES=127.0.0.1,YOUR_NGINX_IP
```

Without this, an attacker can spoof `X-Forwarded-For: 127.0.0.1` to bypass the blacklist.

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | IPS Applied | Description |
|---|---|---|---|
| `GET` | `/health` | No | Health check |
| `GET` | `/status` | No | System status — models, Redis, pipeline layers |
| `POST` | `/api/probe` | **Yes** | Target endpoint — IPS scores every request here |
| `POST` | `/api/inspect` | No | Read-only pipeline inspector — per-layer scores |
| `GET` | `/api/stats` | No | Aggregated 24h statistics |
| `GET` | `/api/events` | No | Event log (`?limit=&action=&attack_type=&ip=`) |
| `POST` | `/api/admin/block-ip` | No | Manually blacklist an IP |
| `GET` | `/api/admin/blocked-ips` | No | List all blacklisted IPs |
| `GET` | `/docs` | No | Swagger UI |

### Example: Inspect a SQL Injection

```bash
curl -X POST http://localhost:8000/api/inspect \
  -H "Content-Type: application/json" \
  -d '{
    "ip": "185.220.101.47",
    "method": "POST",
    "path": "/api/login",
    "body": "username=admin'\''--&password=x",
    "layers_enabled": {
      "layer0": true, "layer1": true, "network": true,
      "layer2_lgbm": true, "layer2_cnn": true
    }
  }' | python -m json.tool
```

### Example: Block an IP

```bash
curl -X POST http://localhost:8000/api/admin/block-ip \
  -H "Content-Type: application/json" \
  -d '{"ip": "203.0.113.45", "reason": "Known scanner", "permanent": false}'
```

---

## Frontend Pages

| Path | Page | Description |
|---|---|---|
| `/` | Home | Architecture overview, pipeline diagram, network layer docs, tech stack |
| `/aim-ips-inspector` | Inspector | Fire custom requests, toggle layers, view per-layer scores and labels |
| `/login` | Login | Admin authentication |
| `/admin-dashboard` | Dashboard | Real-time stats, RPM charts, attack type breakdown, threat globe, event log, top IPs |

### Default Admin Credentials

```
Username: admin
Password: aimips2024
```

> Change before deploying — update `VITE_ADMIN_USERNAME` and `VITE_ADMIN_PASSWORD` in `frontend/.env`.

---

## Attack Presets

The Inspector page includes built-in payloads for testing every layer:

| Preset | Method | Target Layer |
|---|---|---|
| Clean GET request | GET | — (should ALLOW) |
| Clean POST JSON | POST | — (should ALLOW) |
| SQLi — OR bypass | POST | Layer 0 / Layer 1 |
| SQLi — UNION SELECT | GET | Layer 1 |
| SQLi — time based blind | GET | Layer 1 |
| XSS — script tag | POST | Layer 0 / Layer 1 |
| XSS — event handler | POST | Layer 1 |
| Path traversal — /etc/passwd | GET | Layer 0 / Layer 1 |
| CMDi — semicolon cat | POST | Layer 1 |
| Bad UA — sqlmap | GET | Layer 0 |
| Bad UA — nikto | GET | Layer 0 |
| Zero-day anomaly payload | POST | Layer 2b (CNN) |

---

## Redis Key Schema

```
blacklist:ip:{ip}           Blacklisted IP entry
                            JSON: { reason, permanent, timestamp }
                            TTL: 3600s (temp) or -1 (permanent)

ratelimit:ip:{ip}           Request counter (INCR per request)
                            TTL: 60s

threat:ip:{ip}              Network layer threat score
                            JSON: { score, net_lgbm, ensemble, attack_type, confidence, timestamp }
                            TTL: 60s — written by Network IPS, read by middleware

session:ip:{ip}:captcha     CAPTCHA session token
                            TTL: 300s

reqlog:id:{request_id}      Full per-request score log
                            TTL: 120s

admin:events                Recent event list (LPUSH, max 10,000)
                            TTL: 86400s (24h)

db:queue                    Event queue for background DB writer (LPUSH)
                            Max: 50,000 entries — consumed by db/writer.py
```

---

## Scoring & Thresholds

### Layer Fusion Weights

```python
WEIGHT_LAYER1_REGEX  = 0.20   # Regex filter contribution
WEIGHT_LAYER2_LGBM   = 0.35   # LightGBM classifier contribution
WEIGHT_LAYER2_CNN    = 0.25   # CNN autoencoder contribution
WEIGHT_NETWORK       = 0.20   # Network layer contribution
```

### Network Layer Internal Fusion

```python
network_score = 0.55 × lgbm_score + 0.45 × tcn_ensemble_score
```

### CNN Internal Fusion

```python
cnn_score = 0.50 × reconstruction_error + 0.50 × mahalanobis_distance
```

### Action Thresholds

| Range | Action |
|---|---|
| `0.00 – 0.25` | ALLOW |
| `0.25 – 0.35` | DELAY (+1.5s) |
| `0.35 – 0.50` | THROTTLE (+3.0s) |
| `0.50 – 0.70` | CAPTCHA (429) |
| `0.70 – 1.00` | BLOCK (403) |

### Rate Limit

```python
MAX_REQUESTS_PER_MINUTE = 100   # per IP
```

---

## Performance

| Stage | Typical Latency |
|---|---|
| Layer 0 — Static Firewall | < 1ms |
| Layer 1 — Regex Filter | < 2ms |
| Network score lookup (Redis) | < 0.1ms |
| Layer 2b — CNN Autoencoder | ~30–80ms |
| Layer 2a — LightGBM | ~10–30ms |
| Layer 3 — Fusion | < 1ms |
| **Total pipeline** | **~60–150ms** |
| DELAY penalty | +1.5s |
| THROTTLE penalty | +3.0s |
| DB write overhead | 0ms (async, fire-and-forget) |

---

## Debugging

```bash
# Full system status
curl http://localhost:8000/status | python -m json.tool

# Recent events in Redis
redis-cli LRANGE admin:events 0 4

# DB writer queue depth
redis-cli LLEN db:queue

# Recent PostgreSQL events
psql -U postgres -d aimips \
  -c "SELECT ip, action, best_label, final_score, latency_ms FROM attack_events ORDER BY created_at DESC LIMIT 10;"

# Network layer simulation test (no root needed)
python -m pipeline.network_level.network_ips --simulate
```

---

## License

Research project. Not intended for production use without security review.
