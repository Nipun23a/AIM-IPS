"""
ai/threat_analysis.py
=====================
Autonomous Adaptive Intelligence for AIM-IPS.

Architecture (all async, zero latency impact on request pipeline):

  IPSMiddleware (hot path, <3ms)
       │  final_score > AI_ANALYSIS_THRESHOLD
       ▼
  ThreatAnalysisQueue.enqueue_threat()   ← RPUSH  ai:threat:queue
       │
       ▼  (background asyncio task)
  run_ai_analysis_worker()
       ├─ ThreatAnalysisQueue.dequeue_threat()   ← LPOP
       ├─ AIReasoningAgent.analyze_threat()
       │     ├─ HoneyDBClient.query_ip_reputation()
       │     ├─ AbuseIPDBClient.check_ip()
       │     ├─ MITREATTACKMapper.map_attack()
       │     └─ Claude API  (claude-sonnet-4-6)
       └─ ThreatAnalysisQueue.store_result()     ← SETEX ai:analysis:{event_id}
              │
              ▼
         /api/ai-analysis/{event_id}  ← dashboard polls
"""

import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Queue configuration ───────────────────────────────────────────────────────
AI_QUEUE_KEY        = "ai:threat:queue"
AI_RESULT_KEY       = "ai:analysis:{event_id}"
AI_RESULT_TTL       = 86400           # 24 hours
AI_ANALYSIS_THRESHOLD = 0.65          # Only analyze captcha/block-level threats


# ── Threat Analysis Queue ─────────────────────────────────────────────────────

class ThreatAnalysisQueue:
    """
    Redis-backed async queue that decouples real-time detection from
    deep AI analysis.  Enqueue is called in the hot path (<1ms), the
    worker runs entirely in the background.
    """

    def __init__(self, redis_raw):
        self.r = redis_raw

    # ── Producer (called from middleware hot-path) ────────────────────────────

    def enqueue_threat(self, threat_event: dict) -> str:
        """
        Push a threat event onto the analysis queue.

        Returns the event_id for tracking.
        """
        try:
            self.r.rpush(AI_QUEUE_KEY, json.dumps(threat_event))
            self.r.ltrim(AI_QUEUE_KEY, 0, 9999)          # cap queue at 10k
        except Exception as e:
            logger.debug("[AIQueue] Enqueue failed (non-fatal): %s", e)
        return threat_event.get("event_id", "")

    # ── Consumer (called from background worker) ──────────────────────────────

    def dequeue_threat(self) -> Optional[dict]:
        """
        Non-blocking pop from the queue.  Returns None if empty.
        Worker calls this in a sleep loop; use run_in_executor for the
        Claude HTTP call to avoid blocking the event loop.
        """
        try:
            raw = self.r.lpop(AI_QUEUE_KEY)
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.debug("[AIQueue] Dequeue failed: %s", e)
        return None

    # ── Result store ──────────────────────────────────────────────────────────

    def store_result(self, event_id: str, analysis: dict) -> None:
        """Cache analysis result in Redis with 24-hour TTL."""
        try:
            key = AI_RESULT_KEY.format(event_id=event_id)
            self.r.setex(key, AI_RESULT_TTL, json.dumps(analysis))
        except Exception as e:
            logger.debug("[AIQueue] Store result failed: %s", e)

    def get_result(self, event_id: str) -> Optional[dict]:
        """Fetch cached analysis result. Returns None if not ready yet."""
        try:
            key = AI_RESULT_KEY.format(event_id=event_id)
            raw = self.r.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def mark_pending(self, event_id: str) -> None:
        """Mark event as 'queued but not yet analyzed' so the frontend shows a spinner."""
        try:
            key = AI_RESULT_KEY.format(event_id=event_id)
            existing = self.r.get(key)
            if not existing:
                self.r.setex(key, 300, json.dumps({"status": "pending"}))
        except Exception:
            pass


# ── HoneyDB Client ────────────────────────────────────────────────────────────

class HoneyDBClient:
    """
    Queries HoneyDB global honeypot data.
    Free tier: 1,000 requests/day  |  https://honeydb.io/api

    Set HONEYDB_API_ID and HONEYDB_API_KEY in .env.
    If keys are absent, all methods return None silently.
    """

    def __init__(self, api_id: str = "", api_key: str = ""):
        self.api_id   = api_id   or os.getenv("HONEYDB_API_ID",  "")
        self.api_key  = api_key  or os.getenv("HONEYDB_API_KEY", "")
        self.base_url = "https://honeydb.io/api"

    def _headers(self) -> dict:
        return {
            "X-HoneyDb-ApiId":  self.api_id,
            "X-HoneyDb-ApiKey": self.api_key,
        }

    def query_ip_reputation(self, ip_address: str) -> Optional[dict]:
        """
        Check if an IP has attacked honeypots globally.

        Returns normalised reputation dict, or None if not in DB / keys missing.
        """
        if not self.api_id or not self.api_key:
            return None
        try:
            import httpx
            resp = httpx.get(
                f"{self.base_url}/bad-hosts/{ip_address}",
                headers=self._headers(),
                timeout=5.0,
            )
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                logger.warning("[HoneyDB] Rate limited — skipping for %s", ip_address)
                return None
            resp.raise_for_status()
            data  = resp.json()
            count = int(data.get("count", 0) or 0)
            return {
                "ip":               ip_address,
                "count":            count,
                "first_seen":       data.get("first_seen"),
                "last_seen":        data.get("last_seen"),
                "services_targeted": list(data.get("remote_hosts", []))[:10],
                "threat_level":     "high" if count > 10 else ("medium" if count > 2 else "low"),
            }
        except Exception as e:
            logger.debug("[HoneyDB] Query failed for %s: %s", ip_address, e)
            return None

    def query_recent_attacks(self, limit: int = 10) -> list:
        """Return recent global attack patterns (useful for trend context)."""
        if not self.api_id or not self.api_key:
            return []
        try:
            import httpx
            resp = httpx.get(
                f"{self.base_url}/attacks/recent",
                headers=self._headers(),
                params={"limit": limit},
                timeout=5.0,
            )
            resp.raise_for_status()
            return resp.json() if isinstance(resp.json(), list) else []
        except Exception as e:
            logger.debug("[HoneyDB] Recent attacks query failed: %s", e)
            return []


# ── AbuseIPDB Client ──────────────────────────────────────────────────────────

class AbuseIPDBClient:
    """
    Crowd-sourced IP abuse reputation.
    Free tier: 1,000 requests/day  |  https://api.abuseipdb.com/api/v2

    Set ABUSEIPDB_API_KEY in .env.
    """

    def __init__(self, api_key: str = ""):
        self.api_key  = api_key or os.getenv("ABUSEIPDB_API_KEY", "")
        self.base_url = "https://api.abuseipdb.com/api/v2"

    def check_ip(self, ip_address: str, max_age_days: int = 90) -> Optional[dict]:
        """
        Returns abuse confidence score (0-100) + metadata.
        Score interpretation:
            > 75  → high confidence malicious
            25-75 → suspicious
            < 25  → low risk
        """
        if not self.api_key:
            return None
        try:
            import httpx
            resp = httpx.get(
                f"{self.base_url}/check",
                headers={"Key": self.api_key, "Accept": "application/json"},
                params={"ipAddress": ip_address, "maxAgeInDays": max_age_days},
                timeout=5.0,
            )
            resp.raise_for_status()
            d = resp.json().get("data", {})
            return {
                "ip":                     ip_address,
                "abuse_confidence_score": int(d.get("abuseConfidenceScore", 0)),
                "total_reports":          int(d.get("totalReports", 0)),
                "last_reported_at":       d.get("lastReportedAt"),
                "is_whitelisted":         bool(d.get("isWhitelisted", False)),
                "country_code":           d.get("countryCode"),
                "usage_type":             d.get("usageType"),
                "isp":                    d.get("isp"),
            }
        except Exception as e:
            logger.debug("[AbuseIPDB] Query failed for %s: %s", ip_address, e)
            return None


# ── MITRE ATT&CK Mapper ───────────────────────────────────────────────────────

class MITREATTACKMapper:
    """
    Local mapping from AIM-IPS attack labels → MITRE ATT&CK techniques.
    No API needed.
    """

    TECHNIQUE_MAP: dict = {
        # Application-layer attacks
        "sqli":              {"id": "T1190",     "name": "Exploit Public-Facing Application",             "tactic": "Initial Access",        "url": "https://attack.mitre.org/techniques/T1190/"},
        "sql_injection":     {"id": "T1190",     "name": "Exploit Public-Facing Application",             "tactic": "Initial Access",        "url": "https://attack.mitre.org/techniques/T1190/"},
        "xss":               {"id": "T1059.007", "name": "Command and Scripting: JavaScript",             "tactic": "Execution",             "url": "https://attack.mitre.org/techniques/T1059/007/"},
        "cmdi":              {"id": "T1059",     "name": "Command and Scripting Interpreter",             "tactic": "Execution",             "url": "https://attack.mitre.org/techniques/T1059/"},
        "command_injection": {"id": "T1059",     "name": "Command and Scripting Interpreter",             "tactic": "Execution",             "url": "https://attack.mitre.org/techniques/T1059/"},
        "path_traversal":    {"id": "T1083",     "name": "File and Directory Discovery",                  "tactic": "Discovery",             "url": "https://attack.mitre.org/techniques/T1083/"},
        "path-traversal":    {"id": "T1083",     "name": "File and Directory Discovery",                  "tactic": "Discovery",             "url": "https://attack.mitre.org/techniques/T1083/"},
        "rce":               {"id": "T1059",     "name": "Command and Scripting Interpreter",             "tactic": "Execution",             "url": "https://attack.mitre.org/techniques/T1059/"},
        # Network-layer attacks
        "portscan":          {"id": "T1046",     "name": "Network Service Scanning",                      "tactic": "Discovery",             "url": "https://attack.mitre.org/techniques/T1046/"},
        "port_scan":         {"id": "T1046",     "name": "Network Service Scanning",                      "tactic": "Discovery",             "url": "https://attack.mitre.org/techniques/T1046/"},
        "ddos":              {"id": "T1498",     "name": "Network Denial of Service",                     "tactic": "Impact",                "url": "https://attack.mitre.org/techniques/T1498/"},
        "botnet":            {"id": "T1583.005", "name": "Acquire Infrastructure: Botnet",                "tactic": "Resource Development",  "url": "https://attack.mitre.org/techniques/T1583/005/"},
        # Anomaly / unknown
        "anomaly":           {"id": "T1190",     "name": "Exploit Public-Facing Application",             "tactic": "Initial Access",        "url": "https://attack.mitre.org/techniques/T1190/"},
        "zeroday":           {"id": "T1190",     "name": "Exploit Public-Facing Application (Zero-Day)", "tactic": "Initial Access",        "url": "https://attack.mitre.org/techniques/T1190/"},
        "zero-day":          {"id": "T1190",     "name": "Exploit Public-Facing Application (Zero-Day)", "tactic": "Initial Access",        "url": "https://attack.mitre.org/techniques/T1190/"},
        "unknown":           {"id": "T1190",     "name": "Exploit Public-Facing Application (Unknown)",  "tactic": "Initial Access",        "url": "https://attack.mitre.org/techniques/T1190/"},
    }

    KILL_CHAIN: dict = {
        "Initial Access":         "Exploitation",
        "Execution":              "Exploitation",
        "Persistence":            "Installation",
        "Privilege Escalation":   "Exploitation",
        "Defense Evasion":        "Command & Control",
        "Credential Access":      "Exploitation",
        "Discovery":              "Reconnaissance",
        "Lateral Movement":       "Lateral Movement",
        "Collection":             "Actions on Objectives",
        "Command and Control":    "Command & Control",
        "Exfiltration":           "Actions on Objectives",
        "Impact":                 "Actions on Objectives",
        "Resource Development":   "Weaponization",
    }

    def map_attack(self, attack_type: str) -> Optional[dict]:
        """Map AIM-IPS label → MITRE technique dict."""
        if not attack_type:
            return None
        key = attack_type.lower().replace("-", "_").replace(" ", "_")
        return (
            self.TECHNIQUE_MAP.get(key)
            or self.TECHNIQUE_MAP.get(attack_type.lower())
        )

    def get_kill_chain_phase(self, tactic: str) -> str:
        """Map MITRE tactic → Lockheed Martin Cyber Kill Chain phase."""
        return self.KILL_CHAIN.get(tactic, "Unknown")


# ── AI Reasoning Agent ────────────────────────────────────────────────────────

class AIReasoningAgent:
    """
    Orchestrates deep threat analysis using Claude + external threat intel.

    Flow:
        threat_event
            → _gather_threat_intelligence()   (HoneyDB + AbuseIPDB + MITRE)
            → _build_analysis_prompt()
            → _query_claude()                  (claude-sonnet-4-6, temp=0)
            → _parse_claude_response()
            → augmented analysis dict
    """

    def __init__(
        self,
        anthropic_api_key: str = "",
        honeydb_api_id:    str = "",
        honeydb_api_key:   str = "",
        abuseipdb_api_key: str = "",
    ):
        _key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not _key:
            raise ValueError("ANTHROPIC_API_KEY is required for AIReasoningAgent")

        import anthropic
        self.client    = anthropic.Anthropic(api_key=_key)
        self.honeydb   = HoneyDBClient(honeydb_api_id, honeydb_api_key)
        self.abuseipdb = AbuseIPDBClient(abuseipdb_api_key)
        self.mitre     = MITREATTACKMapper()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_threat(self, threat_event: dict) -> dict:
        """
        Main analysis function.  Runs synchronously — call via
        loop.run_in_executor() from the async worker to avoid blocking.

        Returns a structured analysis dict ready for Redis / PostgreSQL storage.
        """
        intel    = self._gather_threat_intelligence(threat_event)
        prompt   = self._build_analysis_prompt(threat_event, intel)
        raw_text = self._query_claude(prompt)
        analysis = self._parse_claude_response(raw_text)

        # Enrich with MITRE + intel + metadata
        tc = analysis.setdefault("threat_classification", {})
        attack_type_raw = tc.get("attack_type") or threat_event.get("attack_type", "unknown")
        mitre = intel.get("mitre") or self.mitre.map_attack(attack_type_raw)
        if mitre:
            tc["mitre_technique"]  = mitre
            tc["kill_chain_phase"] = self.mitre.get_kill_chain_phase(mitre.get("tactic", ""))

        analysis["threat_intelligence"]  = intel
        analysis["event_id"]             = threat_event.get("event_id", "")
        analysis["analysis_timestamp"]   = time.time()
        analysis["status"]               = "complete"
        return analysis

    # ── Step 1: Gather threat intelligence ───────────────────────────────────

    def _gather_threat_intelligence(self, threat_event: dict) -> dict:
        ip          = threat_event.get("ip_address", "")
        attack_type = threat_event.get("attack_type", "unknown")

        intel: dict = {}

        intel["honeydb"]  = self.honeydb.query_ip_reputation(ip)
        intel["abuseipdb"] = self.abuseipdb.check_ip(ip)
        intel["mitre"]    = self.mitre.map_attack(attack_type)

        # Synthesise global_prevalence from combined signals
        count = int((intel["honeydb"]  or {}).get("count",                   0) or 0)
        abuse = int((intel["abuseipdb"] or {}).get("abuse_confidence_score", 0) or 0)
        if count > 20 or abuse > 80:
            intel["global_prevalence"] = "widespread"
        elif count > 5 or abuse > 40:
            intel["global_prevalence"] = "emerging"
        else:
            intel["global_prevalence"] = "isolated"

        return intel

    # ── Step 2: Build Claude prompt ───────────────────────────────────────────

    def _build_analysis_prompt(self, threat_event: dict, intel: dict) -> str:
        scores   = threat_event.get("detection_scores", {})
        payload  = str(threat_event.get("payload") or "")[:500]
        shap_str = json.dumps(threat_event.get("shap_explanation") or {}, indent=2)[:600]
        corr_str = json.dumps(threat_event.get("correlation_context") or {}, indent=2)[:400]

        honey_str = (
            json.dumps(intel.get("honeydb"),   indent=2)
            if intel.get("honeydb") else "No data available"
        )
        abuse_str = (
            json.dumps(intel.get("abuseipdb"), indent=2)
            if intel.get("abuseipdb") else "No data available"
        )
        mitre_str = (
            json.dumps(intel.get("mitre"),     indent=2)
            if intel.get("mitre") else "Not mapped"
        )

        return f"""You are a senior cybersecurity threat analyst at a SOC. \
Analyse the AIM-IPS alert below and return ONLY valid JSON — no markdown, no prose outside the JSON object.

THREAT EVENT:
  Timestamp     : {threat_event.get('timestamp', '')}
  Source IP     : {threat_event.get('ip_address', '')}
  Method / URL  : {threat_event.get('method', 'GET')} {threat_event.get('url', '/')}
  Payload       : {payload}

AIM-IPS DETECTION SCORES (multi-layer pipeline):
  LightGBM classifier score  : {scores.get('lgbm_score', 0):.4f}
  CNN Mahalanobis distance   : {scores.get('cnn_mahalanobis', 0):.4f}
  Fusion score               : {scores.get('fusion_score', 0):.4f}
  Cross-pipeline correlation : {scores.get('correlation_amplification', 1.0):.2f}x
  Final threat score         : {scores.get('final_score', 0):.4f}
  Action taken               : {threat_event.get('action_taken', 'BLOCK')}

LAYER-BY-LAYER FEATURE WEIGHTS (SHAP proxy):
{shap_str}

CROSS-PIPELINE CORRELATION CONTEXT:
{corr_str}

EXTERNAL THREAT INTELLIGENCE:
  HoneyDB (global honeypot hits) :
{honey_str}

  AbuseIPDB (crowd-sourced abuse score) :
{abuse_str}

  MITRE ATT&CK initial mapping :
{mitre_str}

YOUR ANALYSIS TASKS — be specific, technical, and actionable:

1. CLASSIFY: primary attack type, OWASP Top 10 category, severity (CRITICAL/HIGH/MEDIUM/LOW),
   confidence (0.0-1.0), is_novel_variant (bool).
2. ROOT CAUSE: what vulnerability is targeted, why ML models flagged it, evasion techniques used,
   automated vs manual judgment.
3. THREAT ASSESSMENT: sophistication, potential impact, campaign indicators.
4. MITIGATIONS (concrete):
   - WAF rules in ModSecurity syntax
   - Regex patterns for the AIM-IPS pre-filter
   - Model threshold adjustment suggestions
   - Immediate response actions (next 1 hour)
   - Long-term code/architecture fixes
5. ANALYST SUMMARY: 2-3 sentence executive brief.
6. IOCs: IPs, payloads, patterns, domains.

OUTPUT — strict JSON only (no markdown code fences, no text before/after):

{{
  "threat_classification": {{
    "attack_type": "string",
    "owasp_category": "string",
    "severity": "CRITICAL|HIGH|MEDIUM|LOW",
    "confidence": 0.0,
    "is_novel_variant": false
  }},
  "root_cause_analysis": "detailed string",
  "attack_sophistication": "automated|manual|advanced",
  "evasion_techniques": ["string"],
  "mitigation_recommendations": {{
    "waf_rules": ["ModSecurity rule string"],
    "regex_patterns": ["pattern string"],
    "threshold_adjustments": {{"layer": "suggested change"}},
    "immediate_actions": ["string"],
    "long_term_fixes": ["string"]
  }},
  "analyst_summary": "2-3 sentence string",
  "iocs": ["string"]
}}"""

    # ── Step 3: Query Claude ──────────────────────────────────────────────────

    def _query_claude(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    # ── Step 4: Parse response ────────────────────────────────────────────────

    def _parse_claude_response(self, text: str) -> dict:
        """
        Extract JSON from Claude's response.  Handles markdown fences and
        minor formatting artifacts.
        """
        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract the first {...} block as a fallback
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        logger.warning("[AIAgent] Failed to parse Claude JSON response — raw: %.300s", text)
        return {
            "status":       "parse_error",
            "error":        "Failed to parse AI response",
            "raw_response": text[:800],
        }


# ── Helper: build threat event from RequestContext ────────────────────────────

def build_threat_event(ctx) -> dict:
    """
    Convert a RequestContext (from middleware) into the threat_event dict
    expected by ThreatAnalysisQueue.enqueue_threat().

    Called after fusion + correlation, so final_score is set.
    """
    scores_dict = ctx.scores_dict() if hasattr(ctx, "scores_dict") else {}

    # Extract best non-benign label from layer scores
    attack_type = "unknown"
    shap_proxy  = {}
    for ls in reversed(getattr(ctx, "layer_scores", [])):
        lbl = getattr(ls, "label", "") or ""
        if lbl and lbl not in ("clean", "norm", "normal", ""):
            attack_type = lbl
        sc = getattr(ls, "score", 0)
        if sc > 0:
            shap_proxy[getattr(ls, "layer", "?")] = round(sc, 4)

    # Correlation context if present
    corr_ctx = {}
    if hasattr(ctx, "network_threat") and ctx.network_threat:
        corr_ctx["network_threat"] = ctx.network_threat
    if hasattr(ctx, "network_score"):
        corr_ctx["network_score"] = ctx.network_score

    return {
        "event_id":    getattr(ctx, "request_id", ""),
        "timestamp":   getattr(ctx, "timestamp",  time.time()),
        "ip_address":  getattr(ctx, "ip",         ""),
        "url":         getattr(ctx, "path",        "/"),
        "method":      getattr(ctx, "method",      "GET"),
        "payload":     getattr(ctx, "body",        "")[:500],
        "attack_type": attack_type,
        "action_taken": getattr(ctx, "action", "BLOCK"),
        "detection_scores": {
            "lgbm_score":                scores_dict.get("app_lgbm_score",  0.0),
            "cnn_mahalanobis":           scores_dict.get("deep_anomaly",    0.0),
            "fusion_score":              scores_dict.get("final_risk",      getattr(ctx, "final_score", 0.0)),
            "correlation_amplification": 1.0,
            "final_score":               getattr(ctx, "final_score",        0.0),
        },
        "shap_explanation":  shap_proxy,
        "correlation_context": corr_ctx,
    }
