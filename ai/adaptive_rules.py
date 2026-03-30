"""
ai/adaptive_rules.py
====================
Manages the adaptive rule feedback loop:
  Claude analysis  →  validated regex patterns  →  Redis store
  Background task  →  RegexFilter.load_adaptive_rules()  →  Layer 1

Redis layout:
  adaptive:rules          — Hash: rule_id → JSON rule object
  adaptive:rules:count    — total rules ever pushed (for dedup / cap)

Each rule object:
  {
    "rule_id":         "a1b2c3d4",
    "pattern":         "(?i)(union[\\s\\+]+select)",
    "attack_type":     "sqli",
    "source_event_id": "3df0644c-...",
    "severity":        "CRITICAL",
    "confidence":      0.97,
    "status":          "active",          # active | rejected
    "match_count":     0,
    "created_at":      1711234567.0,
    "last_matched_at": null
  }
"""

import json
import logging
import re
import time
import uuid

logger = logging.getLogger(__name__)

ADAPTIVE_RULES_KEY = "adaptive:rules"     # Redis hash
MAX_ADAPTIVE_RULES = 500                  # hard cap to prevent unbounded growth

# Benign inputs used to reject overly-broad patterns
_BENIGN_SAMPLES = [
    "/api/status", "/health", "GET", "POST",
    "application/json", "text/html",
    "Mozilla/5.0 (Windows NT 10.0)",
    "localhost", "127.0.0.1",
    "username=alice", "search=hello world",
    "page=1&limit=20", "id=42",
]


# ── Pattern safety validator ───────────────────────────────────────────────────

def validate_pattern(pattern: str) -> bool:
    """
    Safety checks before auto-applying a Claude-generated regex.
    Returns True if the pattern is safe to add to Layer 1.
    """
    if not pattern or not isinstance(pattern, str):
        return False

    stripped = pattern.strip()

    # Length bounds: too short = useless, too long = ReDoS risk
    if len(stripped) < 5 or len(stripped) > 500:
        return False

    # Must compile without error
    try:
        compiled = re.compile(stripped, re.IGNORECASE | re.DOTALL)
    except re.error as e:
        logger.debug("[AdaptiveRules] Rejected pattern (compile error): %s — %s", stripped[:80], e)
        return False

    # Must NOT match the empty string (would fire on every request)
    if compiled.search(""):
        logger.debug("[AdaptiveRules] Rejected pattern (matches empty): %s", stripped[:80])
        return False

    # Must NOT be a bare wildcard
    if stripped in (".*", ".+", ".{0,}", ".{1,}", "(?s).*"):
        return False

    # Must not match 3 or more benign samples (too broad)
    benign_hits = sum(1 for s in _BENIGN_SAMPLES if compiled.search(s))
    if benign_hits >= 3:
        logger.debug(
            "[AdaptiveRules] Rejected pattern (too broad, %d benign hits): %s",
            benign_hits, stripped[:80],
        )
        return False

    return True


# ── Adaptive Rule Store ────────────────────────────────────────────────────────

class AdaptiveRuleStore:
    """
    Thin wrapper around a Redis hash that stores adaptive regex rules.
    Thread-safe for concurrent reads; writes use simple HSET (last-write-wins).
    """

    def __init__(self, redis_raw):
        self._r = redis_raw

    # ── Write ──────────────────────────────────────────────────────────────────

    def push_patterns(
        self,
        patterns: list,
        attack_type: str,
        event_id: str,
        severity: str,
        confidence: float,
    ) -> list:
        """
        Validate and store Claude-generated regex patterns.
        Returns list of rule_ids that were accepted and stored.
        Silently skips duplicates and invalid patterns.
        """
        if not patterns:
            return []

        # Fetch existing patterns to prevent exact duplicates
        try:
            existing = {
                json.loads(v).get("pattern")
                for v in (self._r.hvals(ADAPTIVE_RULES_KEY) or [])
            }
            total = self._r.hlen(ADAPTIVE_RULES_KEY) or 0
        except Exception:
            existing = set()
            total = 0

        if total >= MAX_ADAPTIVE_RULES:
            logger.warning("[AdaptiveRules] Rule cap (%d) reached — skipping push", MAX_ADAPTIVE_RULES)
            return []

        accepted = []
        for raw_pattern in patterns:
            if not isinstance(raw_pattern, str):
                continue
            pattern = raw_pattern.strip()

            if pattern in existing:
                continue  # exact duplicate

            if not validate_pattern(pattern):
                continue

            rule_id = uuid.uuid4().hex[:8]
            rule = {
                "rule_id":         rule_id,
                "pattern":         pattern,
                "attack_type":     str(attack_type)[:50],
                "source_event_id": str(event_id)[:36],
                "severity":        str(severity)[:20],
                "confidence":      round(float(confidence), 4),
                "status":          "active",
                "match_count":     0,
                "created_at":      round(time.time(), 3),
                "last_matched_at": None,
            }
            try:
                self._r.hset(ADAPTIVE_RULES_KEY, rule_id, json.dumps(rule))
                existing.add(pattern)
                accepted.append(rule_id)
                total += 1
                if total >= MAX_ADAPTIVE_RULES:
                    break
            except Exception as e:
                logger.error("[AdaptiveRules] Redis write failed: %s", e)

        if accepted:
            logger.info(
                "[AdaptiveRules] Pushed %d new rule(s) for %s (severity=%s)",
                len(accepted), attack_type, severity,
            )
        return accepted

    def increment_match(self, rule_id: str) -> None:
        """Called by RegexFilter when an adaptive pattern fires."""
        try:
            raw = self._r.hget(ADAPTIVE_RULES_KEY, rule_id)
            if raw:
                rule = json.loads(raw)
                rule["match_count"]     = rule.get("match_count", 0) + 1
                rule["last_matched_at"] = round(time.time(), 3)
                self._r.hset(ADAPTIVE_RULES_KEY, rule_id, json.dumps(rule))
        except Exception:
            pass

    def remove_rule(self, rule_id: str) -> bool:
        """Delete a rule (reject from dashboard)."""
        try:
            return bool(self._r.hdel(ADAPTIVE_RULES_KEY, rule_id))
        except Exception:
            return False

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_active_rules(self) -> list:
        """Return all active rules, sorted newest-first."""
        return self._get_rules(status_filter="active")

    def get_all_rules(self) -> list:
        return self._get_rules(status_filter=None)

    def _get_rules(self, status_filter=None) -> list:
        try:
            raw_all = self._r.hvals(ADAPTIVE_RULES_KEY) or []
        except Exception:
            return []
        rules = []
        for raw in raw_all:
            try:
                rule = json.loads(raw)
                if status_filter is None or rule.get("status") == status_filter:
                    rules.append(rule)
            except Exception:
                pass
        return sorted(rules, key=lambda r: r.get("created_at", 0), reverse=True)
