from collections import defaultdict
import time
import re

from .rules import *
from .decisions import FirewallDecision


class StaticFirewall:
    def __init__(self):
        self.request_log = defaultdict(list)

    def _rate_limit(self, ip):
        now = time.time()
        window = 60  # seconds

        self.request_log[ip] = [
            t for t in self.request_log[ip] if now - t < window
        ]
        self.request_log[ip].append(now)
        return len(self.request_log[ip]) > MAX_REQUESTS_PER_MINUTE

    def _match(self, text, patterns):
        text = text.lower()
        for p in patterns:
            if re.search(p, text):
                return True
        return False

    def inspect(self, request):
        ip = request.get("ip", "")
        body = request.get("body", "") or ""
        path = request.get("path", "") or ""
        headers = request.get("headers", {})
        ua = headers.get("User-Agent", "").lower()

        # ─────────────────────────────────────────
        # HARD BLOCK RULES (HIGH CONFIDENCE)
        # ─────────────────────────────────────────

        if self._match(body, SQLI_PATTERNS):
            return FirewallDecision.MITIGATE, "SQL Injection"

        if self._match(body, XSS_PATTERNS):
            return FirewallDecision.MITIGATE, "XSS"

        if self._match(path, PATH_TRAVERSAL):
            return FirewallDecision.MITIGATE, "Path Traversal"

        if any(bad_ua in ua for bad_ua in BAD_USER_AGENTS):
            return FirewallDecision.MITIGATE, "Malicious User-Agent"

        if self._rate_limit(ip):
            return FirewallDecision.MITIGATE, "Rate Limiting"

        # ─────────────────────────────────────────
        # SUSPICION HEURISTICS (NEW 🔥)
        # ─────────────────────────────────────────

        suspicion_score = 0

        # URL / encoding (common in obfuscated attacks)
        if "%" in body or "%25" in body:
            suspicion_score += 1

        # High special character ratio
        if body:
            special_ratio = sum(not c.isalnum() for c in body) / max(len(body), 1)
            if special_ratio > 0.30:
                suspicion_score += 1

        # Suspicious automation tools
        if any(tool in ua for tool in ["curl", "python", "wget", "httpclient"]):
            suspicion_score += 1

        # Very long payloads / queries
        if len(body) > 200 or len(path) > 150:
            suspicion_score += 1

        # Encoded traversal attempts
        if "%2e%2e" in body.lower() or "%2f" in body.lower():
            suspicion_score += 1

        if suspicion_score >= 2:
            return FirewallDecision.FORWARD_TO_ML, {
                "reason": "Suspicious but unclear",
                "suspicion_score": suspicion_score
            }

        # ─────────────────────────────────────────
        # UNCOMMON METHOD → ML
        # ─────────────────────────────────────────

        if request.get("method") not in ["GET", "POST"]:
            return FirewallDecision.FORWARD_TO_ML, "Uncommon HTTP Method"

        # ─────────────────────────────────────────
        # CLEAN TRAFFIC
        # ─────────────────────────────────────────

        return FirewallDecision.ALLOW, "Normal Traffic"
