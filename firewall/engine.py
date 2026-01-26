

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
        ip = request.get("ip"," ")
        body = request.get("body"," ")
        path = request.get("path"," ")
        headers = request.get("headers", {})
        ua = headers.get("User-Agent", "").lower()

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
        
        if request.get("method") not in ["GET", "POST"]:
            return FirewallDecision.FORWARD_TO_ML, "Uncommon HTTP Method"
        
        return FirewallDecision.ALLOW, "Normal Traffic"