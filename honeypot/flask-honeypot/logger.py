import json, os, datetime
from tagger import tag_request
from honeydb import lookup_ip_reputation

# Optional Prometheus
PROM = None
try:
    from prometheus_client import Counter, Gauge, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
    REG = CollectorRegistry()
    REQ_TOTAL = Counter("hp_requests_total", "Total requests", ["method"], registry=REG)
    TAG_TOTAL = Counter("hp_tags_total", "Tagged family hits", ["family"], registry=REG)
    TI_FOUND = Counter("hp_ti_found_total", "Threat intel reputation hits", ["source"], registry=REG)
    UNIQUE_IPS = Gauge("hp_unique_ips", "Unique IPs observed", registry=REG)
    PROM = {"REG": REG, "REQ_TOTAL": REQ_TOTAL, "TAG_TOTAL": TAG_TOTAL, "TI_FOUND": TI_FOUND, "UNIQUE_IPS": UNIQUE_IPS, "CONTENT_TYPE": CONTENT_TYPE_LATEST, "generate_latest": generate_latest}
except Exception:
    PROM = None

LOG_FILE = os.path.join("logs", "honeypot_log.json")
_seen_ips = set()

def _append_json(entry):
    os.makedirs("logs", exist_ok=True)
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    else:
        data = []
    data.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def log_request(req):
    ip = req.headers.get("X-Forwarded-For", req.remote_addr)
    method = req.method
    path = req.path
    query = req.args.to_dict(flat=True)
    form = req.form.to_dict(flat=True)
    headers = {k: v for k, v in req.headers.items()}
    raw_body = req.get_data(as_text=True)

    tags = tag_request(path, query, form, headers, raw_body)
    ti = lookup_ip_reputation(ip)

    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "ip": ip,
        "method": method,
        "path": path,
        "query_params": query,
        "form_data": form,
        "headers": headers,
        "raw_body": raw_body,
        "tags": tags,
        "threat_intel": ti
    }
    _append_json(entry)

    # Prometheus bookkeeping
    if PROM:
        PROM["REQ_TOTAL"].labels(method=method).inc()
        for tag in tags:
            family = tag.split(":", 1)[0]
            PROM["TAG_TOTAL"].labels(family=family).inc()
        if ti and ti.get("ti_found"):
            PROM["TI_FOUND"].labels(source=ti.get("ti_source","unknown")).inc()
        global _seen_ips
        if ip not in _seen_ips:
            _seen_ips.add(ip)
            PROM["UNIQUE_IPS"].set(len(_seen_ips))

def get_metrics_json():
    # Lightweight JSON metrics in case Prometheus not installed
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        total = len(data)
        by_method = {}
        tag_counts = {}
        ips = set()
        ti_hits = 0
        for r in data:
            ips.add(r.get("ip"))
            m = r.get("method","")
            by_method[m] = by_method.get(m,0)+1
            for t in r.get("tags",[]):
                fam = t.split(":",1)[0]
                tag_counts[fam] = tag_counts.get(fam,0)+1
            if r.get("threat_intel",{}).get("ti_found"):
                ti_hits += 1
        return {
            "total_requests": total,
            "by_method": by_method,
            "unique_ips": len(ips),
            "tag_counts": tag_counts,
            "ti_hits": ti_hits
        }
    except Exception as e:
        return {"error": str(e)}
