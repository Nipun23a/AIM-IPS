"""
test_ai_analysis.py
===================
Tests for the AI adaptive intelligence system.

Run:
    pytest test_ai_analysis.py -v
"""

import json
import sys
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# ── Stub `anthropic` so tests run without the SDK installed ──────────────────
if "anthropic" not in sys.modules:
    _anthropic_stub = MagicMock()
    _anthropic_stub.Anthropic = MagicMock
    sys.modules["anthropic"] = _anthropic_stub


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_redis_mock():
    """Return a mock redis client with RPUSH / LPOP / SETEX / GET support."""
    store: dict = {}
    queue: list = []

    mock = MagicMock()
    mock.rpush.side_effect = lambda key, val: queue.append((key, val))
    mock.ltrim.return_value = True
    mock.lpop.side_effect  = lambda key: queue.pop(0)[1] if queue else None
    mock.setex.side_effect = lambda key, ttl, val: store.update({key: val})
    mock.get.side_effect   = lambda key: store.get(key)
    mock._store = store
    mock._queue = queue
    return mock


def _sample_threat_event(event_id: str = "test-001") -> dict:
    return {
        "event_id":    event_id,
        "timestamp":   time.time(),
        "ip_address":  "203.0.113.45",
        "url":         "/api/users",
        "method":      "POST",
        "payload":     "username=admin' UNION SELECT NULL,password FROM users--",
        "attack_type": "sqli",
        "action_taken": "BLOCK",
        "detection_scores": {
            "lgbm_score":                0.91,
            "cnn_mahalanobis":           0.78,
            "fusion_score":              0.87,
            "correlation_amplification": 1.3,
            "final_score":               0.94,
        },
        "shap_explanation":  {"layer2_lgbm": 0.91, "layer1_regex": 0.85, "layer2_cnn": 0.78},
        "correlation_context": {"network_score": 0.45, "portscan_seen": True},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. ThreatAnalysisQueue
# ─────────────────────────────────────────────────────────────────────────────

class TestThreatAnalysisQueue(unittest.TestCase):

    def setUp(self):
        from ai.threat_analysis import ThreatAnalysisQueue
        self.redis = _make_redis_mock()
        self.queue = ThreatAnalysisQueue(self.redis)

    def test_enqueue_returns_event_id(self):
        evt = _sample_threat_event("evt-123")
        result = self.queue.enqueue_threat(evt)
        self.assertEqual(result, "evt-123")

    def test_enqueue_pushes_to_redis(self):
        self.queue.enqueue_threat(_sample_threat_event())
        self.assertTrue(len(self.redis._queue) > 0)
        _, raw = self.redis._queue[0]
        parsed = json.loads(raw)
        self.assertEqual(parsed["ip_address"], "203.0.113.45")

    def test_dequeue_returns_none_when_empty(self):
        result = self.queue.dequeue_threat()
        self.assertIsNone(result)

    def test_enqueue_dequeue_roundtrip(self):
        evt = _sample_threat_event("roundtrip-42")
        self.queue.enqueue_threat(evt)
        dequeued = self.queue.dequeue_threat()
        self.assertIsNotNone(dequeued)
        self.assertEqual(dequeued["event_id"], "roundtrip-42")
        self.assertEqual(dequeued["attack_type"], "sqli")

    def test_store_and_get_result(self):
        analysis = {"status": "complete", "event_id": "evt-1", "severity": "HIGH"}
        self.queue.store_result("evt-1", analysis)
        result = self.queue.get_result("evt-1")
        self.assertIsNotNone(result)
        self.assertEqual(result["severity"], "HIGH")

    def test_get_result_missing(self):
        result = self.queue.get_result("non-existent")
        self.assertIsNone(result)

    def test_mark_pending(self):
        self.queue.mark_pending("pending-evt")
        result = self.queue.get_result("pending-evt")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "pending")

    def test_redis_error_on_enqueue_does_not_raise(self):
        """Queue must never crash the request pipeline."""
        self.redis.rpush.side_effect = Exception("Redis down")
        # Should not raise
        self.queue.enqueue_threat(_sample_threat_event())


# ─────────────────────────────────────────────────────────────────────────────
# 2. MITREATTACKMapper
# ─────────────────────────────────────────────────────────────────────────────

class TestMITREATTACKMapper(unittest.TestCase):

    def setUp(self):
        from ai.threat_analysis import MITREATTACKMapper
        self.mapper = MITREATTACKMapper()

    def test_sqli_maps_to_T1190(self):
        result = self.mapper.map_attack("sqli")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "T1190")
        self.assertEqual(result["tactic"], "Initial Access")

    def test_xss_maps_correctly(self):
        result = self.mapper.map_attack("xss")
        self.assertIsNotNone(result)
        self.assertIn("T1059", result["id"])

    def test_portscan_maps_to_T1046(self):
        result = self.mapper.map_attack("portscan")
        self.assertEqual(result["id"], "T1046")

    def test_ddos_maps_to_T1498(self):
        result = self.mapper.map_attack("ddos")
        self.assertEqual(result["id"], "T1498")

    def test_case_insensitive(self):
        self.assertEqual(
            self.mapper.map_attack("SQLI"),
            self.mapper.map_attack("sqli"),
        )

    def test_hyphenated_path_traversal(self):
        result = self.mapper.map_attack("path-traversal")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "T1083")

    def test_unknown_returns_none_or_default(self):
        result = self.mapper.map_attack("completely_unknown_xyz")
        # Should return None for unmapped types
        self.assertIsNone(result)

    def test_kill_chain_initial_access(self):
        phase = self.mapper.get_kill_chain_phase("Initial Access")
        self.assertEqual(phase, "Exploitation")

    def test_kill_chain_discovery(self):
        phase = self.mapper.get_kill_chain_phase("Discovery")
        self.assertEqual(phase, "Reconnaissance")


# ─────────────────────────────────────────────────────────────────────────────
# 3. HoneyDBClient
# ─────────────────────────────────────────────────────────────────────────────

class TestHoneyDBClient(unittest.TestCase):

    def setUp(self):
        from ai.threat_analysis import HoneyDBClient
        self.client = HoneyDBClient(api_id="test_id", api_key="test_key")

    def test_returns_none_without_keys(self):
        from ai.threat_analysis import HoneyDBClient
        client = HoneyDBClient(api_id="", api_key="")
        result = client.query_ip_reputation("1.2.3.4")
        self.assertIsNone(result)

    def test_404_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("httpx.get", return_value=mock_resp):
            result = self.client.query_ip_reputation("1.2.3.4")
        self.assertIsNone(result)

    def test_429_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        with patch("httpx.get", return_value=mock_resp):
            result = self.client.query_ip_reputation("1.2.3.4")
        self.assertIsNone(result)

    def test_successful_response_normalised(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "count":        47,
            "first_seen":   "2024-01-01T00:00:00Z",
            "last_seen":    "2026-03-20T12:00:00Z",
            "remote_hosts": ["http", "ssh", "ftp"],
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            result = self.client.query_ip_reputation("203.0.113.45")
        self.assertIsNotNone(result)
        self.assertEqual(result["count"], 47)
        self.assertEqual(result["threat_level"], "high")
        self.assertIn("http", result["services_targeted"])

    def test_threat_level_low(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"count": 1}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            result = self.client.query_ip_reputation("1.2.3.4")
        self.assertEqual(result["threat_level"], "low")

    def test_network_error_returns_none(self):
        with patch("httpx.get", side_effect=Exception("connection refused")):
            result = self.client.query_ip_reputation("1.2.3.4")
        self.assertIsNone(result)


# ─────────────────────────────────────────────────────────────────────────────
# 4. AbuseIPDBClient
# ─────────────────────────────────────────────────────────────────────────────

class TestAbuseIPDBClient(unittest.TestCase):

    def setUp(self):
        from ai.threat_analysis import AbuseIPDBClient
        self.client = AbuseIPDBClient(api_key="test_key")

    def test_returns_none_without_key(self):
        from ai.threat_analysis import AbuseIPDBClient
        client = AbuseIPDBClient(api_key="")
        result = client.check_ip("1.2.3.4")
        self.assertIsNone(result)

    def test_successful_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "abuseConfidenceScore": 98,
                "totalReports":         156,
                "lastReportedAt":       "2026-03-20T10:00:00+00:00",
                "isWhitelisted":        False,
                "countryCode":          "CN",
                "usageType":            "Data Center/Web Hosting/Transit",
                "isp":                  "SomeHosting Inc.",
            }
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            result = self.client.check_ip("203.0.113.45")
        self.assertEqual(result["abuse_confidence_score"], 98)
        self.assertEqual(result["total_reports"], 156)
        self.assertEqual(result["country_code"], "CN")

    def test_network_error_returns_none(self):
        with patch("httpx.get", side_effect=Exception("timeout")):
            result = self.client.check_ip("1.2.3.4")
        self.assertIsNone(result)


# ─────────────────────────────────────────────────────────────────────────────
# 5. AIReasoningAgent — prompt generation + response parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestAIReasoningAgentParsing(unittest.TestCase):
    """
    Tests the prompt builder and response parser without calling the real
    Claude API (mock the client).
    """

    def _make_agent(self):
        from ai.threat_analysis import AIReasoningAgent
        with patch("anthropic.Anthropic"):
            agent = AIReasoningAgent.__new__(AIReasoningAgent)
            agent.client    = MagicMock()
            agent.honeydb   = MagicMock()
            agent.abuseipdb = MagicMock()
            from ai.threat_analysis import MITREATTACKMapper
            agent.mitre     = MITREATTACKMapper()
        return agent

    def test_prompt_contains_ip(self):
        agent = self._make_agent()
        evt   = _sample_threat_event()
        intel = {"honeydb": None, "abuseipdb": None, "mitre": None, "global_prevalence": "isolated"}
        prompt = agent._build_analysis_prompt(evt, intel)
        self.assertIn("203.0.113.45", prompt)

    def test_prompt_contains_scores(self):
        agent = self._make_agent()
        evt   = _sample_threat_event()
        intel = {"honeydb": None, "abuseipdb": None, "mitre": None, "global_prevalence": "isolated"}
        prompt = agent._build_analysis_prompt(evt, intel)
        self.assertIn("0.9400", prompt)   # final_score

    def test_prompt_contains_payload(self):
        agent = self._make_agent()
        evt   = _sample_threat_event()
        intel = {"honeydb": None, "abuseipdb": None, "mitre": None, "global_prevalence": "isolated"}
        prompt = agent._build_analysis_prompt(evt, intel)
        self.assertIn("UNION SELECT", prompt)

    def test_parse_clean_json(self):
        agent = self._make_agent()
        raw = json.dumps({
            "threat_classification": {
                "attack_type": "SQL Injection",
                "owasp_category": "A03:2021",
                "severity": "CRITICAL",
                "confidence": 0.94,
                "is_novel_variant": False,
            },
            "root_cause_analysis": "Insufficient input validation.",
            "attack_sophistication": "automated",
            "evasion_techniques": ["basic_obfuscation"],
            "mitigation_recommendations": {
                "waf_rules":             ["SecRule ARGS ..."],
                "regex_patterns":        ["\\bUNION\\b"],
                "threshold_adjustments": {},
                "immediate_actions":     ["Block IP"],
                "long_term_fixes":       ["Parameterised queries"],
            },
            "analyst_summary": "SQL injection attempt from known bad IP.",
            "iocs": ["203.0.113.45"],
        })
        result = agent._parse_claude_response(raw)
        self.assertEqual(result["threat_classification"]["severity"], "CRITICAL")
        self.assertEqual(result["threat_classification"]["confidence"], 0.94)

    def test_parse_markdown_wrapped_json(self):
        agent = self._make_agent()
        raw = '```json\n{"threat_classification": {"attack_type": "XSS", "severity": "HIGH", "confidence": 0.85, "is_novel_variant": false, "owasp_category": "A03"}}\n```'
        result = agent._parse_claude_response(raw)
        self.assertEqual(result["threat_classification"]["attack_type"], "XSS")

    def test_parse_invalid_returns_error_dict(self):
        agent = self._make_agent()
        result = agent._parse_claude_response("this is not json at all")
        self.assertIn("error", result)

    def test_parse_extracts_embedded_json(self):
        """Should extract JSON block even if surrounded by extra text."""
        agent = self._make_agent()
        raw = 'Here is the analysis:\n{"threat_classification": {"attack_type": "CMDi", "severity": "CRITICAL", "confidence": 0.88, "is_novel_variant": false, "owasp_category": "A03"}, "root_cause_analysis": "cmd injection", "attack_sophistication": "manual", "evasion_techniques": [], "mitigation_recommendations": {}, "analyst_summary": "cmd", "iocs": []}\nEnd of analysis.'
        result = agent._parse_claude_response(raw)
        self.assertEqual(result["threat_classification"]["attack_type"], "CMDi")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integration: full analyze_threat flow (mocked Claude)
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_ANALYSIS_JSON = json.dumps({
    "threat_classification": {
        "attack_type":     "SQL Injection - UNION-based",
        "owasp_category":  "A03:2021 - Injection",
        "severity":        "CRITICAL",
        "confidence":      0.94,
        "is_novel_variant": False,
    },
    "root_cause_analysis":   "UNION-based SQL injection targeting user credentials.",
    "attack_sophistication": "automated",
    "evasion_techniques":    ["comment_obfuscation"],
    "mitigation_recommendations": {
        "waf_rules":             ['SecRule ARGS "@rx (?i)union\\s+select" "id:900201,phase:2,block"'],
        "regex_patterns":        ["\\b(UNION|SELECT)\\s+(SELECT|FROM|NULL)"],
        "threshold_adjustments": {"layer1_regex": "lower confidence threshold to 0.55"},
        "immediate_actions":     ["Block 203.0.113.45 for 24h", "Audit /api/users endpoint"],
        "long_term_fixes":       ["Use parameterised queries", "Add integer validation"],
    },
    "analyst_summary": "High-confidence SQL injection from known malicious IP following port scan. Immediate blocking recommended.",
    "iocs":            ["203.0.113.45", "UNION SELECT NULL,password FROM users"],
})


class TestAIReasoningAgentIntegration(unittest.TestCase):

    def _make_agent_with_mock_claude(self):
        from ai.threat_analysis import AIReasoningAgent
        with patch("anthropic.Anthropic"):
            agent = AIReasoningAgent.__new__(AIReasoningAgent)
            # Mock Claude to return deterministic JSON
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text=_MOCK_ANALYSIS_JSON)]
            agent.client = MagicMock()
            agent.client.messages.create.return_value = mock_msg
            # Mock threat intel to avoid HTTP calls
            agent.honeydb            = MagicMock()
            agent.honeydb.query_ip_reputation.return_value = {
                "ip": "203.0.113.45", "count": 47, "threat_level": "high",
                "first_seen": "2024-01-01", "last_seen": "2026-03-20",
                "services_targeted": ["http"],
            }
            agent.abuseipdb          = MagicMock()
            agent.abuseipdb.check_ip.return_value = {
                "ip": "203.0.113.45", "abuse_confidence_score": 98,
                "total_reports": 156, "country_code": "CN", "isp": "BadISP",
            }
            from ai.threat_analysis import MITREATTACKMapper
            agent.mitre = MITREATTACKMapper()
        return agent

    def test_analyze_threat_returns_complete_result(self):
        agent  = self._make_agent_with_mock_claude()
        result = agent.analyze_threat(_sample_threat_event())

        self.assertEqual(result["status"], "complete")
        self.assertIn("threat_classification", result)
        self.assertIn("threat_intelligence",   result)
        self.assertIn("mitigation_recommendations", result)
        self.assertIn("analyst_summary",       result)

    def test_analyze_threat_severity_critical(self):
        agent  = self._make_agent_with_mock_claude()
        result = agent.analyze_threat(_sample_threat_event())
        self.assertEqual(result["threat_classification"]["severity"], "CRITICAL")

    def test_analyze_threat_adds_mitre(self):
        agent  = self._make_agent_with_mock_claude()
        result = agent.analyze_threat(_sample_threat_event())
        self.assertIn("mitre_technique", result["threat_classification"])
        self.assertEqual(result["threat_classification"]["mitre_technique"]["id"], "T1190")

    def test_analyze_threat_adds_kill_chain(self):
        agent  = self._make_agent_with_mock_claude()
        result = agent.analyze_threat(_sample_threat_event())
        self.assertIn("kill_chain_phase", result["threat_classification"])
        self.assertEqual(result["threat_classification"]["kill_chain_phase"], "Exploitation")

    def test_analyze_threat_includes_honeydb(self):
        agent  = self._make_agent_with_mock_claude()
        result = agent.analyze_threat(_sample_threat_event())
        self.assertEqual(result["threat_intelligence"]["honeydb"]["count"], 47)

    def test_analyze_threat_global_prevalence_widespread(self):
        agent  = self._make_agent_with_mock_claude()
        result = agent.analyze_threat(_sample_threat_event())
        self.assertEqual(result["threat_intelligence"]["global_prevalence"], "widespread")

    def test_event_id_preserved(self):
        agent  = self._make_agent_with_mock_claude()
        result = agent.analyze_threat(_sample_threat_event("my-event-id"))
        self.assertEqual(result["event_id"], "my-event-id")

    def test_waf_rules_present(self):
        agent  = self._make_agent_with_mock_claude()
        result = agent.analyze_threat(_sample_threat_event())
        rules  = result["mitigation_recommendations"]["waf_rules"]
        self.assertTrue(len(rules) > 0)
        self.assertIn("SecRule", rules[0])

    def test_iocs_list(self):
        agent  = self._make_agent_with_mock_claude()
        result = agent.analyze_threat(_sample_threat_event())
        self.assertIn("203.0.113.45", result["iocs"])


# ─────────────────────────────────────────────────────────────────────────────
# 7. AI_ANALYSIS_THRESHOLD constant
# ─────────────────────────────────────────────────────────────────────────────

class TestThreshold(unittest.TestCase):
    def test_threshold_value(self):
        from ai.threat_analysis import AI_ANALYSIS_THRESHOLD
        # Must be in captcha/block territory (0.50-0.85)
        self.assertGreaterEqual(AI_ANALYSIS_THRESHOLD, 0.50)
        self.assertLessEqual(AI_ANALYSIS_THRESHOLD, 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
