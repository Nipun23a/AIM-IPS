import re
import logging
from urllib.parse import unquote,unquote_plus
from html import unescape
from typing import Tuple,Optional

from firewall.decisions import FirewallDecision

logger = logging.getLogger(__name__)



SQLI_HIGH = [
    (re.compile(r"union\s+(?:all\s+)?select\s+", re.I), 1.0, "UNION SELECT injection"),
    (re.compile(r"'\s*(?:or|and)\s+['\d].*?--", re.I), 1.0, "Tautology with comment"),
    (re.compile(r"'[\s]*--", re.I), 1.0, "Quote-comment bypass"),
    (re.compile(r"'[\s]*#", re.I), 1.0, "Quote-hash bypass"),
    (re.compile(r"(?:'\s*(?:or|and)\s+'\d+'='['\d]|'\s*or\s+\d+=\d+)", re.I), 1.0, "Classic OR/AND tautology"),
    (re.compile(r";\s*(?:drop|truncate|delete|insert|update|create|alter)\s+", re.I), 1.0, "Stacked query"),
    (re.compile(r"(?:sleep\s*\(\s*\d+|benchmark\s*\(\s*\d+|pg_sleep\s*\(\s*\d+|waitfor\s+delay)", re.I), 1.0, "Time-based blind injection"),
    (re.compile(r"from\s+information_schema\s*\.", re.I), 1.0, "Information schema probe"),
    (re.compile(r"(?:sysobjects|syscolumns|sys\.tables|sys\.columns|pg_catalog)", re.I), 1.0, "System table access"),
    (re.compile(r"(?:exec|execute)\s+(?:xp_cmdshell|sp_executesql|sp_makewebtask)", re.I), 1.0, "SQL exec command"),
    (re.compile(r"(?:load_file\s*\(|into\s+(?:out|dump)file\s*['\"])", re.I), 1.0, "SQL file read/write"),
]

SQLI_MEDIUM = [
    (re.compile(r"['\"].*?(?:select|insert|update|delete|drop|union|exec)", re.I), 0.7, "Quote with SQL keyword"),
    (re.compile(r"(?:--|#|/\*[\s\S]*?\*/)\s*$", re.I), 0.65, "SQL comment at end"),
    (re.compile(r"(?:select|union|insert|update|delete|drop).{0,30}(?:from|where|table|database)", re.I), 0.7, "Multiple SQL keywords"),
    (re.compile(r"(?:0x[0-9a-f]{4,}|char\s*\(\s*\d+)", re.I), 0.65, "SQL hex/char encoding"),
    (re.compile(r"having\s+\d+=\d+|group\s+by\s+.+having", re.I), 0.75, "HAVING clause injection"),
]

XSS_HIGH = [
    (re.compile(r"<script[\s>]", re.I), 1.0, "Script tag"),
    (re.compile(r"<script[^>]+src\s*=", re.I), 1.0, "Script with src"),
    (re.compile(r"javascript\s*:", re.I), 1.0, "JavaScript protocol"),
    (re.compile(r"on(?:error|load|click|mouseover|focus|blur|change|submit|keypress|keydown|keyup)\s*=\s*['\"]?(?:alert|eval|document|window|location|fetch)", re.I), 1.0, "Event handler with JS"),
    (re.compile(r"(?:eval|setTimeout|setInterval)\s*\(", re.I), 1.0, "JS code execution"),
    (re.compile(r"document\s*\.\s*(?:cookie|write|location|domain|referrer)", re.I), 1.0, "DOM access"),
    (re.compile(r"vbscript\s*:", re.I), 1.0, "VBScript protocol"),
    (re.compile(r"<svg[^>]+on\w+\s*=", re.I), 1.0, "SVG event handler"),
    (re.compile(r"<iframe[^>]+src\s*=\s*['\"]?(?:javascript|data|vbscript)", re.I), 1.0, "Iframe injection"),
    (re.compile(r"data\s*:\s*text/html", re.I), 1.0, "Data URI HTML"),
]

XSS_MEDIUM = [
    (re.compile(r"on\w{2,20}\s*=\s*['\"]", re.I), 0.75, "Generic event handler"),
    (re.compile(r"<(?:img|body|input|form|a|link|meta|object|embed)[^>]+(?:src|href|action|data)\s*=\s*['\"]?(?:javascript|vbscript|data)", re.I), 0.8, "HTML tag with JS URI"),
    (re.compile(r"(?:alert|prompt|confirm)\s*\(", re.I), 0.7, "XSS alert/prompt/confirm"),
    (re.compile(r"&#x[0-9a-f]+;|&#\d+;", re.I), 0.65, "HTML entity encoding"),
    (re.compile(r"expression\s*\(", re.I), 0.8, "CSS expression"),
]

# ── Path Traversal Patterns ───────────────────────────────────

PATH_HIGH = [
    (re.compile(r"(?:\.\./){3,}|(?:\.\.\\){3,}"), 1.0, "Multiple directory traversal"),
    (re.compile(r"/etc/(?:passwd|shadow|hosts|group|sudoers|crontab)", re.I), 1.0, "/etc sensitive file"),
    (re.compile(r"(?:c:\\windows\\|\\system32\\|\\syswow64\\)", re.I), 1.0, "Windows system path"),
    (re.compile(r"/proc/(?:self|version|cmdline|environ|net/)", re.I), 1.0, "/proc access"),
    (re.compile(r"(?:\.htaccess|\.htpasswd|web\.config|php\.ini|nginx\.conf)", re.I), 1.0, "Server config file"),
    (re.compile(r"(?:\.ssh/|id_rsa|authorized_keys|\.aws/credentials)", re.I), 1.0, "Credentials file"),
    (re.compile(r"%00|\\x00|\x00"), 1.0, "Null byte injection"),
]

PATH_MEDIUM = [
    (re.compile(r"(?:\.\./){2}|(?:\.\.\\){2}"), 0.8, "Directory traversal"),
    (re.compile(r"(?:%2e%2e%2f|%2e%2e/|\.\.%2f|%252e%252e)", re.I), 0.9, "Encoded path traversal"),
    (re.compile(r"(?:%2e%2e%5c|%252e%252e%255c)", re.I), 0.9, "Encoded Windows traversal"),
    (re.compile(r"\.\./.*?\.(?:conf|cfg|ini|log|bak|sql|db|env)", re.I), 0.75, "Traversal to config file"),
]

# ── Command Injection Patterns ───────────────────────────────

CMDI_HIGH = [
    (re.compile(r"(?:;|\||&&|`|\$\()\s*(?:cat|ls|id|whoami|uname|pwd|wget|curl|nc|netcat|bash|sh|python|perl|ruby|php)\b", re.I), 1.0, "Shell command after separator"),
    (re.compile(r"\$\([^)]+\)|`[^`]+`"), 1.0, "Command substitution"),
    (re.compile(r"(?:bash|nc|netcat|python|perl|ruby)\s+.*?(?:\d{1,3}\.){3}\d{1,3}\s+\d{2,5}", re.I), 1.0, "Reverse shell"),
    (re.compile(r"/(?:bin|usr/bin|sbin|usr/sbin)/(?:bash|sh|nc|wget|curl|python|perl|ruby|php)", re.I), 1.0, "Binary path execution"),
    (re.compile(r"\$(?:PATH|IFS|HOME|SHELL|USER|LD_PRELOAD)\b"), 1.0, "Environment variable injection"),
    (re.compile(r">\s*/etc/|>\s*/var/www/|>\s*/tmp/", re.I), 1.0, "Output redirection to system path"),
]

CMDI_MEDIUM = [
    (re.compile(r"\|\s*(?:grep|awk|sed|cut|head|tail|sort|uniq|wc|xargs)\b", re.I), 0.7, "Pipe with text command"),
    (re.compile(r";\s*(?:echo|printf|sleep|ping|nslookup|dig)\b", re.I), 0.75, "Semicolon with command"),
    (re.compile(r"`[^`]{3,}`"), 0.7, "Backtick expression"),
    (re.compile(r"&&\s*\w+|;\s*\w+\s*;"), 0.65, "Chained commands"),
]

class RegexFilter:
    SCAN_HEADERS = {"referer", "x-forwarded-for", "user-agent", "x-custom-header"}

    def __init__(self):
        # List of (compiled_pattern, rule_id, attack_type) tuples.
        # Replaced atomically by load_adaptive_rules() — GIL makes list
        # assignment thread-safe without an explicit lock.
        self._adaptive_patterns: list = []

    def load_adaptive_rules(self, rules: list) -> None:
        """
        Hot-reload adaptive patterns from AdaptiveRuleStore.get_active_rules().
        Called by the background reload task in main.py every 30 seconds.
        """
        new_patterns = []
        for rule in rules:
            try:
                compiled = re.compile(rule["pattern"], re.IGNORECASE | re.DOTALL)
                new_patterns.append((compiled, rule["rule_id"], rule.get("attack_type", "adaptive")))
            except re.error:
                pass
        self._adaptive_patterns = new_patterns   # atomic replacement
        if new_patterns:
            logger.info("[RegexFilter] %d adaptive rule(s) loaded", len(new_patterns))

    def inspect(self,request:dict) -> Tuple[FirewallDecision,dict]:
        body         = str(request.get("body", "") or "")
        path         = str(request.get("path", "") or "")
        query_params = request.get("query_params", {}) or {}
        headers      = request.get("headers", {}) or {}

        targets = []

        if body:
            targets.append(("body", body))
        if path:
            targets.append(("path", path))
        for k, v in query_params.items():
            targets.append((f"query:{k}", str(v)))
        for k, v in headers.items():
            if k.lower() in self.SCAN_HEADERS:
                targets.append((f"header:{k}", str(v)))
        
        best_match = None

        for field_name,raw_value in targets:
            decoded_variants = self._decode_all(raw_value)
            for decoded in decoded_variants:
                result = self._scan_field(decoded, field_name, raw_value)
                if result:
                    if best_match is None or result["confidence"] > best_match["confidence"]:
                        best_match = result
                    if result["confidence"] >= 1.0:
                        return FirewallDecision.MITIGATE, best_match
        
        if best_match is None:
            return FirewallDecision.ALLOW, {"pattern_group": None, "confidence": 0.0}
        
        if best_match["confidence"] >= 1.0:
            return FirewallDecision.MITIGATE, best_match
        
        return FirewallDecision.FORWARD_TO_ML, best_match
    
    def _scan_field(
        self, decoded: str, field_name: str, raw_value: str
    ) -> Optional[dict]:

        best = None

        pattern_groups = [
            ("sqli",           SQLI_HIGH,  SQLI_MEDIUM),
            ("xss",            XSS_HIGH,   XSS_MEDIUM),
            ("path_traversal", PATH_HIGH,  PATH_MEDIUM),
            ("cmdi",           CMDI_HIGH,  CMDI_MEDIUM),
        ]

        for group_name, high_patterns, medium_patterns in pattern_groups:

            for pattern, confidence, description in high_patterns:
                match = pattern.search(decoded)
                if match:
                    result = {
                        "pattern_group": group_name,
                        "pattern":       description,
                        "confidence":    confidence,
                        "matched_in":    field_name,
                        "matched_text":  match.group(0)[:100],  
                        "decoded_input": decoded[:200],
                    }
                    if best is None or confidence > best["confidence"]:
                        best = result
                    if confidence >= 1.0:
                        return best 

            for pattern, confidence, description in medium_patterns:
                match = pattern.search(decoded)
                if match:
                    result = {
                        "pattern_group": group_name,
                        "pattern":       description,
                        "confidence":    confidence,
                        "matched_in":    field_name,
                        "matched_text":  match.group(0)[:100],
                        "decoded_input": decoded[:200],
                    }
                    if best is None or confidence > best["confidence"]:
                        best = result

        # ── Adaptive patterns (Claude-generated, hot-reloaded) ────────────────
        for compiled, rule_id, attack_type in self._adaptive_patterns:
            match = compiled.search(decoded)
            if match:
                result = {
                    "pattern_group": attack_type,
                    "pattern":       f"adaptive:{rule_id}",
                    "confidence":    1.0,
                    "matched_in":    field_name,
                    "matched_text":  match.group(0)[:100],
                    "decoded_input": decoded[:200],
                    "adaptive":      True,
                    "rule_id":       rule_id,
                }
                # Fire-and-forget match counter (best effort)
                try:
                    from utils.redis_client import RedisClient
                    r = RedisClient.get_redis()
                    if r:
                        from ai.adaptive_rules import AdaptiveRuleStore
                        AdaptiveRuleStore(r.raw).increment_match(rule_id)
                except Exception:
                    pass
                return result   # always high-confidence → MITIGATE

        return best

    def _decode_all(self, value: str) -> list:
        variants = set()
        variants.add(value)
        variants.add(value.lower())
        try:
            d1 = unquote(value)
            variants.add(d1)
            variants.add(d1.lower())
        except Exception:
            pass
        try:
            d2 = unquote(unquote(value))
            variants.add(d2)
            variants.add(d2.lower())
        except Exception:
            pass
        try:
            d3 = unquote_plus(value)
            variants.add(d3)
            variants.add(d3.lower())
        except Exception:
            pass
        try:
            d4 = unescape(value)
            variants.add(d4)
            variants.add(d4.lower())
        except Exception:
            pass
        try:
            d5 = unescape(unquote(value))
            variants.add(d5)
            variants.add(d5.lower())
        except Exception:
            pass
        try:
            d6 = re.sub(r"[\t\n\r\x0b\x0c]+", " ", value)
            variants.add(d6)
            variants.add(d6.lower())
        except Exception:
            pass
        try:
            d7 = re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL)
            variants.add(d7)
            variants.add(d7.lower())
        except Exception:
            pass

        return list(variants)
    

    def confidence_to_score(self, confidence: float) -> float:
        if confidence >= 1.0:
            return 1.0
        if confidence <= 0.6:
            return 0.2
        return round(0.2 + (confidence - 0.6) * (0.8 / 0.4), 3)











    

        
            
