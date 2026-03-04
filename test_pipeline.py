"""
test_pipeline.py
─────────────────
Manual test script for AIM-IPS pipeline.

Tests every layer independently AND end-to-end with real attack payloads.
Run from project root:
    python test_pipeline.py

No pytest needed — plain Python, prints clear pass/fail results.
"""

import sys
import json
import time
sys.path.insert(0, ".")

# ── Colours for terminal output ──────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}\n" + "─"*60)


# ─────────────────────────────────────────────────────────────
# TEST PAYLOADS
# Expected: what layer/action should trigger
# ─────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── Clean traffic ────────────────────────────────────────
    {
        "name":     "Clean GET request",
        "request":  {"ip": "192.168.1.1", "method": "GET",  "path": "/api/users", "body": "", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {"page": "1"}},
        "expect_action": "ALLOW",
        "expect_layer":  None,
    },
    {
        "name":     "Clean POST with JSON",
        "request":  {"ip": "192.168.1.2", "method": "POST", "path": "/api/login",  "body": '{"username":"admin","password":"secret123"}', "headers": {"User-Agent": "Mozilla/5.0","Content-Type":"application/json"}, "query_params": {}},
        "expect_action": "ALLOW",
        "expect_layer":  None,
    },

    # ── SQLi ─────────────────────────────────────────────────
    {
        "name":     "SQLi — classic OR 1=1",
        "request":  {"ip": "10.0.0.1", "method": "POST", "path": "/login", "body": "username=admin'--&password=x", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },
    {
        "name":     "SQLi — UNION SELECT",
        "request":  {"ip": "10.0.0.2", "method": "GET",  "path": "/search", "body": "", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {"q": "' UNION SELECT NULL, username, password FROM users--"}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },
    {
        "name":     "SQLi — time based blind",
        "request":  {"ip": "10.0.0.3", "method": "GET",  "path": "/product", "body": "", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {"id": "1; SELECT SLEEP(5)--"}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },
    {
        "name":     "SQLi — encoded evasion (%27)",
        "request":  {"ip": "10.0.0.4", "method": "GET",  "path": "/item", "body": "", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {"id": "1%27%20OR%20%271%27%3D%271"}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },

    # ── XSS ──────────────────────────────────────────────────
    {
        "name":     "XSS — script tag",
        "request":  {"ip": "10.0.1.1", "method": "POST", "path": "/comment", "body": "<script>alert('xss')</script>", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },
    {
        "name":     "XSS — event handler",
        "request":  {"ip": "10.0.1.2", "method": "POST", "path": "/profile", "body": '<img src=x onerror=alert(document.cookie)>', "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },
    {
        "name":     "XSS — javascript protocol",
        "request":  {"ip": "10.0.1.3", "method": "GET",  "path": "/redirect", "body": "", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {"url": "javascript:alert(1)"}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },

    # ── Path Traversal ───────────────────────────────────────
    {
        "name":     "Path Traversal — /etc/passwd",
        "request":  {"ip": "10.0.2.1", "method": "GET",  "path": "/download", "body": "", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {"file": "../../../etc/passwd"}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },
    {
        "name":     "Path Traversal — encoded",
        "request":  {"ip": "10.0.2.2", "method": "GET",  "path": "/file", "body": "", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {"path": "%2e%2e%2f%2e%2e%2fetc%2fpasswd"}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },

    # ── Command Injection ────────────────────────────────────
    {
        "name":     "CMDi — semicolon + cat",
        "request":  {"ip": "10.0.3.1", "method": "POST", "path": "/ping", "body": "host=127.0.0.1; cat /etc/passwd", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },
    {
        "name":     "CMDi — pipe + whoami",
        "request":  {"ip": "10.0.3.2", "method": "POST", "path": "/exec", "body": "cmd=ls | whoami", "headers": {"User-Agent": "Mozilla/5.0"}, "query_params": {}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer1_regex",
    },

    # ── Layer 0 triggers ─────────────────────────────────────
    {
        "name":     "Bad User Agent — sqlmap",
        "request":  {"ip": "10.0.4.1", "method": "GET",  "path": "/", "body": "", "headers": {"User-Agent": "sqlmap/1.7"}, "query_params": {}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer0_firewall",
    },
    {
        "name":     "Bad User Agent — nikto",
        "request":  {"ip": "10.0.4.2", "method": "GET",  "path": "/", "body": "", "headers": {"User-Agent": "Nikto/2.1.6"}, "query_params": {}},
        "expect_action": "BLOCK",
        "expect_layer":  "layer0_firewall",
    },
]


# ─────────────────────────────────────────────────────────────
# TEST 1 — REGEX FILTER IN ISOLATION
# ─────────────────────────────────────────────────────────────

def test_regex_filter():
    header("TEST 1 — Layer 1: RegexFilter (isolation)")
    from firewall.regex_filter import RegexFilter
    from firewall.decisions import FirewallDecision

    rf = RegexFilter()
    passed = 0
    failed = 0

    for tc in TEST_CASES:
        req    = tc["request"]
        name   = tc["name"]
        expect = tc["expect_action"]

        decision, detail = rf.inspect(req)

        if expect == "BLOCK":
            if decision == FirewallDecision.MITIGATE:
                ok(f"{name}")
                info(f"   group={detail.get('pattern_group')} | pattern={detail.get('pattern')} | conf={detail.get('confidence')}")
                passed += 1
            elif decision == FirewallDecision.FORWARD_TO_ML:
                warn(f"{name} → FORWARD_TO_ML (medium confidence, not hard block)")
                info(f"   group={detail.get('pattern_group')} | conf={detail.get('confidence')}")
                passed += 1  # still detected
            else:
                fail(f"{name} → expected BLOCK, got {decision}")
                failed += 1

        elif expect == "ALLOW":
            if decision == FirewallDecision.ALLOW:
                ok(f"{name}")
                passed += 1
            else:
                fail(f"{name} → expected ALLOW, got {decision} | {detail.get('pattern_group')} | {detail.get('pattern')}")
                failed += 1

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 2 — LGBM CLASSIFIER IN ISOLATION
# ─────────────────────────────────────────────────────────────

def test_lgbm():
    header("TEST 2 — Layer 2a: LightGBM Classifier (isolation)")
    from threat_classifier.lgbm_classifier import LGBMAppClassifier
    from shared.schemas import RequestContext

    try:
        lgbm = LGBMAppClassifier().load()
    except Exception as e:
        fail(f"Model load failed: {e}")
        return False

    # Payloads with expected label (approximate — model decides)
    payloads = [
        ("Clean",          "GET /api/users?page=1",                         "norm"),
        ("SQLi UNION",     "' UNION SELECT NULL, password FROM users--",    "sqli"),
        ("SQLi OR",        "admin' OR '1'='1",                              "sqli"),
        ("XSS script",     "<script>alert(document.cookie)</script>",       "xss"),
        ("XSS event",      "<img src=x onerror=alert(1)>",                  "xss"),
        ("CMDi",           "; cat /etc/passwd",                             "cmdi"),
        ("Path Traversal", "../../../../etc/passwd",                        "path-traversal"),
    ]

    passed = 0
    failed = 0

    for name, payload, expected_label in payloads:
        ctx = RequestContext(ip="1.2.3.4", method="POST", path="/test", body=payload)
        score = lgbm.predict(ctx)

        all_probs = score.metadata.get("all_probs", {})
        probs_str = " | ".join(f"{k}={v:.2f}" for k, v in sorted(all_probs.items(), key=lambda x: -x[1]))

        if score.label == expected_label:
            ok(f"{name} → {score.label} (score={score.score:.3f})")
            info(f"   {probs_str}")
            passed += 1
        elif score.label == "norm" and expected_label != "norm":
            fail(f"{name} → predicted '{score.label}' expected '{expected_label}' (score={score.score:.3f})")
            info(f"   {probs_str}")
            failed += 1
        else:
            warn(f"{name} → predicted '{score.label}' expected '{expected_label}' (score={score.score:.3f})")
            info(f"   {probs_str}")
            passed += 1  # detected as attack, just different class

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 3 — CNN AUTOENCODER IN ISOLATION
# ─────────────────────────────────────────────────────────────

def test_cnn():
    header("TEST 3 — Layer 2b: CNN Autoencoder (isolation)")
    from anomly_detector.cnn_detector import CNNAnomalyDetector
    from shared.schemas import RequestContext

    try:
        cnn = CNNAnomalyDetector().load()
    except Exception as e:
        fail(f"Model load failed: {e}")
        return False

    payloads = [
        ("Clean request",     "GET /api/users?page=1",                      "clean"),
        ("SQLi payload",      "' UNION SELECT NULL, password FROM users--", "anom"),
        ("XSS payload",       "<script>alert(document.cookie)</script>",    "anom"),
        ("CMDi payload",      "; cat /etc/passwd",                          "anom"),
        ("Path traversal",    "../../../../etc/passwd",                     "anom"),
        ("Normal JSON",       '{"name":"John","age":30}',                   "clean"),
    ]

    passed = 0
    failed = 0

    for name, payload, expected in payloads:
        ctx   = RequestContext(ip="1.2.3.4", method="POST", path="/test", body=payload)
        score = cnn.predict(ctx)

        recon = score.metadata.get("recon_score", 0)
        maha  = score.metadata.get("maha_score", 0)
        fused = score.metadata.get("fusion_used", False)

        is_anom = score.score > 0.5

        if expected == "anom" and is_anom:
            ok(f"{name} → anomaly detected (score={score.score:.3f})")
            info(f"   recon={recon:.3f} | maha={maha:.3f} | fusion={fused}")
            passed += 1
        elif expected == "clean" and not is_anom:
            ok(f"{name} → clean (score={score.score:.3f})")
            info(f"   recon={recon:.3f} | maha={maha:.3f} | fusion={fused}")
            passed += 1
        else:
            fail(f"{name} → score={score.score:.3f} expected={'anomaly' if expected=='anom' else 'clean'}")
            info(f"   recon={recon:.3f} | maha={maha:.3f} | fusion={fused}")
            failed += 1

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 4 — FULL PIPELINE END-TO-END (no HTTP server needed)
# ─────────────────────────────────────────────────────────────

def test_full_pipeline():
    header("TEST 4 — Full Pipeline End-to-End (no server)")
    from shared.schemas import RequestContext, LayerScore
    from shared.constants import LAYER_0, LAYER_1
    from firewall.decisions import FirewallDecision
    from firewall.regex_filter import RegexFilter
    from pipeline.layer2 import Layer2MLOrchestrator
    from response.engine import ResponseEngine

    # Load models once
    try:
        layer2 = Layer2MLOrchestrator().load()
    except Exception as e:
        fail(f"Layer2 load failed: {e}")
        return False

    rf     = RegexFilter()
    engine = ResponseEngine(auto_blacklist=False)  # don't actually blacklist in tests

    passed = 0
    failed = 0

    for tc in TEST_CASES:
        req    = tc["request"]
        name   = tc["name"]
        expect = tc["expect_action"]

        # Build context
        ctx = RequestContext(
            ip           = req["ip"],
            method       = req["method"],
            path         = req["path"],
            body         = req.get("body", ""),
            headers      = req.get("headers", {}),
            query_params = req.get("query_params", {}),
        )

        # Layer 0 — Static Firewall
        try:
            from firewall.engine import StaticFirewall
            fw_decision, fw_reason = StaticFirewall().inspect(req)
            if fw_decision == FirewallDecision.MITIGATE:
                ctx.add_score(LayerScore.hard_block(LAYER_0, str(fw_reason), str(fw_reason)))
                ctx.short_circuited = True
                ctx.action = "BLOCK"
                ctx.block_reason = str(fw_reason)
        except Exception as e:
            ctx.add_score(LayerScore.clean(LAYER_0))

        # Layer 1 — Regex Filter
        if not ctx.short_circuited:
            l1_decision, l1_detail = rf.inspect(req)
            if l1_decision == FirewallDecision.MITIGATE:
                ctx.add_score(LayerScore.hard_block(
                    LAYER_1,
                    l1_detail.get("pattern_group", "regex"),
                    l1_detail.get("pattern", "")
                ))
                ctx.short_circuited = True
                ctx.action = "BLOCK"
                ctx.block_reason = l1_detail.get("pattern", "regex match")
            elif l1_decision == FirewallDecision.FORWARD_TO_ML:
                conf = l1_detail.get("confidence", 0.65)
                ctx.add_score(LayerScore(
                    score=rf.confidence_to_score(conf),
                    label=l1_detail.get("pattern_group", "suspicious"),
                    confidence=conf, layer=LAYER_1, triggered=False,
                    metadata=l1_detail,
                ))
            else:
                ctx.add_score(LayerScore.clean(LAYER_1))

        # Layer 2 — ML
        if not ctx.short_circuited:
            ctx = layer2.run(ctx)

        # Layer 3 — Response Engine
        decision, detail = engine.decide(ctx)

        # ── Print result ────────────────────────────────────
        scores = ctx.scores_dict()
        score_str = (
            f"regex={scores['regex_conf']:.2f} | "
            f"lgbm={scores['app_lgbm_score']:.2f} | "
            f"cnn={scores['deep_anomaly']:.2f} | "
            f"net={scores['net_lgbm_score']:.2f} | "
            f"final={ctx.final_score:.3f}"
        )

        # Check result
        action_match = ctx.action == expect or decision.value == expect

        if action_match:
            ok(f"{name} → {BOLD}{ctx.action}{RESET}")
            info(f"   {score_str}")
            if ctx.block_reason:
                info(f"   reason: {ctx.block_reason}")
            passed += 1
        else:
            fail(f"{name}")
            info(f"   expected={expect} got={ctx.action}")
            info(f"   {score_str}")
            if ctx.block_reason:
                info(f"   reason: {ctx.block_reason}")
            failed += 1

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 5 — SCORE FUSION SANITY CHECK
# ─────────────────────────────────────────────────────────────

def test_score_fusion():
    header("TEST 5 — Score Fusion Sanity Check")
    from response.engine import ResponseEngine
    from shared.schemas import RequestContext, LayerScore
    from shared.constants import LAYER_1, LAYER_2_LGBM, LAYER_2_CNN

    engine = ResponseEngine(auto_blacklist=False)

    fusion_cases = [
        {
            "name":   "All zeros → ALLOW",
            "scores": {"regex": 0.0, "lgbm": 0.0, "cnn": 0.0, "net": 0.0},
            "expect": "ALLOW",
        },
        {
            "name":   "High SQLi from LGBM → BLOCK",
            "scores": {"regex": 0.9, "lgbm": 0.95, "cnn": 0.8, "net": 0.0},
            "expect": "BLOCK",
        },
        {
            "name":   "Medium suspicion → THROTTLE or CAPTCHA",
            "scores": {"regex": 0.5, "lgbm": 0.6, "cnn": 0.4, "net": 0.0},
            "expect": ("THROTTLE", "CAPTCHA"),
        },
        {
            "name":   "Low suspicion → DELAY",
            "scores": {"regex": 0.35, "lgbm": 0.35, "cnn": 0.25, "net": 0.0},
            "expect": "DELAY",
        },
        {
            "name":   "Network score tips balance → CAPTCHA",
            "scores": {"regex": 0.2, "lgbm": 0.3, "cnn": 0.2, "net": 0.9},
            "expect": ("CAPTCHA", "THROTTLE"),
        },
    ]

    passed = 0
    failed = 0

    for fc in fusion_cases:
        ctx = RequestContext(ip="1.2.3.4", method="GET", path="/test")
        ctx.network_score = fc["scores"]["net"]

        ctx.add_score(LayerScore(
            score=fc["scores"]["regex"], label="test",
            confidence=fc["scores"]["regex"], layer=LAYER_1, triggered=False,
        ))
        ctx.add_score(LayerScore(
            score=fc["scores"]["lgbm"], label="test",
            confidence=fc["scores"]["lgbm"], layer=LAYER_2_LGBM, triggered=False,
        ))
        ctx.add_score(LayerScore(
            score=fc["scores"]["cnn"], label="test",
            confidence=fc["scores"]["cnn"], layer=LAYER_2_CNN, triggered=False,
        ))

        decision, detail = engine.decide(ctx)
        action = ctx.action

        expected = fc["expect"]
        if isinstance(expected, tuple):
            match = action in expected
        else:
            match = action == expected

        # Compute expected fused score manually for verification
        from shared.constants import (
            WEIGHT_LAYER1_REGEX, WEIGHT_LAYER2_LGBM,
            WEIGHT_LAYER2_CNN, WEIGHT_NETWORK
        )
        expected_fused = (
            WEIGHT_LAYER1_REGEX * fc["scores"]["regex"] +
            WEIGHT_LAYER2_LGBM  * fc["scores"]["lgbm"] +
            WEIGHT_LAYER2_CNN   * fc["scores"]["cnn"] +
            WEIGHT_NETWORK      * fc["scores"]["net"]
        )

        if match:
            ok(f"{fc['name']} → {BOLD}{action}{RESET} (fused={ctx.final_score:.3f}, expected≈{expected_fused:.3f})")
            passed += 1
        else:
            fail(f"{fc['name']} → got={action} expected={expected} (fused={ctx.final_score:.3f})")
            failed += 1

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 6 — REDIS CONNECTION
# ─────────────────────────────────────────────────────────────

def test_redis():
    header("TEST 6 — Redis Operations")
    from utils.redis_client import RedisClient

    try:
        r = RedisClient.get_redis()
    except Exception as e:
        fail(f"Redis connection failed: {e}")
        return False

    passed = 0
    failed = 0

    # Ping
    if r.ping():
        ok("Redis ping")
        passed += 1
    else:
        fail("Redis ping")
        failed += 1

    # Write + read network threat score
    test_ip = "192.168.99.99"
    r.set_network_threat_score(test_ip, score=0.82, net_lgbm=0.8, tcn=0.85, attack_type="DDoS")
    data = r.get_network_threat_score(test_ip)
    if data and abs(data["score"] - 0.82) < 0.001:
        ok(f"Network score write/read (score={data['score']})")
        passed += 1
    else:
        fail(f"Network score write/read — got {data}")
        failed += 1

    # get_network_score_value convenience method
    val = r.get_network_score_value(test_ip)
    if abs(val - 0.82) < 0.001:
        ok(f"get_network_score_value={val}")
        passed += 1
    else:
        fail(f"get_network_score_value={val}")
        failed += 1

    # Blacklist
    r.blacklist_ip(test_ip, reason="test")
    if r.is_blacklisted(test_ip):
        ok("Blacklist write/check")
        passed += 1
    else:
        fail("Blacklist write/check")
        failed += 1

    r.remove_from_blacklist(test_ip)
    if not r.is_blacklisted(test_ip):
        ok("Blacklist remove")
        passed += 1
    else:
        fail("Blacklist remove")
        failed += 1

    # Rate limit
    r.raw.delete(f"ratelimit:ip:{test_ip}")
    count = r.increment_request_count(test_ip)
    if count == 1:
        ok(f"Rate limit increment (count={count})")
        passed += 1
    else:
        fail(f"Rate limit increment (count={count})")
        failed += 1

    # Captcha
    r.set_captcha_session(test_ip)
    if r.is_captcha_pending(test_ip):
        ok("Captcha challenge set")
        passed += 1
    else:
        fail("Captcha challenge set")
        failed += 1

    r.resolve_captcha(test_ip)
    if r.is_captcha_solved(test_ip):
        ok("Captcha resolved")
        passed += 1
    else:
        fail("Captcha resolved")
        failed += 1

    # Cleanup
    r.raw.delete(f"threat:ip:{test_ip}")
    r.raw.delete(f"ratelimit:ip:{test_ip}")
    r.raw.delete(f"session:ip:{test_ip}:captcha")

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*60}")
    print("AIM-IPS PIPELINE TEST SUITE")
    print(f"{'='*60}{RESET}")

    results = {}

    # Run all tests
    tests = [
        ("Regex Filter",       test_regex_filter),
        ("LightGBM",           test_lgbm),
        ("CNN Autoencoder",    test_cnn),
        ("Full Pipeline",      test_full_pipeline),
        ("Score Fusion",       test_score_fusion),
        ("Redis",              test_redis),
    ]

    for test_name, test_fn in tests:
        try:
            results[test_name] = test_fn()
        except Exception as e:
            print(f"\n  {RED}ERROR in {test_name}: {e}{RESET}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print(f"\n{BOLD}{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}{RESET}\n")

    total_passed = 0
    total_failed = 0
    for name, passed in results.items():
        if passed:
            print(f"  {GREEN}✓{RESET} {name}")
            total_passed += 1
        else:
            print(f"  {RED}✗{RESET} {name}")
            total_failed += 1

    print(f"\n  {BOLD}Total: {GREEN}{total_passed} passed{RESET} | {RED}{total_failed} failed{RESET}{RESET}\n")

    if total_failed == 0:
        print(f"  {GREEN}{BOLD}All tests passed — pipeline is working correctly ✓{RESET}\n")
    else:
        print(f"  {RED}{BOLD}{total_failed} test suite(s) failed — check output above{RESET}\n")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)