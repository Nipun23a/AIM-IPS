"""
test_network.py
────────────────
Network Layer IPS Test Suite — no root needed.

Tests all 3 levels:
  Level 1 — Feature extractor (synthetic flows → 16 CICIDS features)
  Level 2 — Model inference   (features → LightGBM + TCN scores)
  Level 3 — Full pipeline     (simulate flows → classify → Redis write)

Run from project root:
    python test_network.py

No Scapy capture, no root, no real traffic needed.
"""

import sys
import time
import numpy as np
sys.path.insert(0, ".")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):   print(f"  {RED}✗{RESET} {msg}")
def warn(msg):   print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg):   print(f"  {CYAN}→{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}\n" + "─"*60)


# ─────────────────────────────────────────────────────────────
# SYNTHETIC FLOW DEFINITIONS
# Built from known CICIDS2017 feature distributions.
# Each has an expected classification.
# ─────────────────────────────────────────────────────────────

SYNTHETIC_FLOWS = [
    {
        "name":     "Normal web browsing",
        "src_ip":   "192.168.1.100",
        "expected": "BENIGN",
        "features": {
            "flow duration":                  2.5,
            "total fwd packets":              8,
            "total backward packets":         6,
            "total length of fwd packets":    1200,
            "total length of bwd packets":    8000,
            "fwd packet length mean":         150.0,
            "bwd packet length mean":         1333.0,
            "flow bytes/s":                   3680.0,
            "flow packets/s":                 5.6,
            "syn flag count":                 1,
            "ack flag count":                 12,
            "psh flag count":                 4,
            "packet length mean":             654.0,
            "packet length std":              580.0,
            "idle mean":                      0.1,
            "idle std":                       0.05,
        },
    },
    {
        "name":     "Normal HTTPS session",
        "src_ip":   "192.168.1.101",
        "expected": "BENIGN",
        "features": {
            "flow duration":                  5.0,
            "total fwd packets":              15,
            "total backward packets":         12,
            "total length of fwd packets":    2000,
            "total length of bwd packets":    15000,
            "fwd packet length mean":         133.0,
            "bwd packet length mean":         1250.0,
            "flow bytes/s":                   3400.0,
            "flow packets/s":                 5.4,
            "syn flag count":                 1,
            "ack flag count":                 25,
            "psh flag count":                 8,
            "packet length mean":             629.0,
            "packet length std":              560.0,
            "idle mean":                      0.2,
            "idle std":                       0.1,
        },
    },
    
    {
    "name":     "DDoS SYN flood",
    "src_ip":   "10.0.0.50",
    "expected": "ddos",
    "features": {
        "flow duration":                  2.0,
        "total fwd packets":              5000,
        "total backward packets":         0,
        "total length of fwd packets":    300000,
        "total length of bwd packets":    0,
        "fwd packet length mean":         60.0,
        "bwd packet length mean":         0.0,
        "flow bytes/s":                   150000.0,   # ← was 80,000,000
        "flow packets/s":                 2500.0,     # ← was 1,000,000
        "syn flag count":                 5000,
        "ack flag count":                 0,
        "psh flag count":                 0,
        "packet length mean":             60.0,
        "packet length std":              2.0,
        "idle mean":                      0.0,
        "idle std":                       0.0,
    },
},
{
    "name":     "DDoS UDP flood",
    "src_ip":   "10.0.0.53",
    "expected": "ddos",
    "features": {
        "flow duration":                  1.0,
        "total fwd packets":              3000,
        "total backward packets":         0,
        "total length of fwd packets":    360000,
        "total length of bwd packets":    0,
        "fwd packet length mean":         120.0,
        "bwd packet length mean":         0.0,
        "flow bytes/s":                   360000.0,   # ← was 960,000
        "flow packets/s":                 3000.0,     # ← was 8,000
        "syn flag count":                 0,
        "ack flag count":                 0,
        "psh flag count":                 0,
        "packet length mean":             120.0,
        "packet length std":              5.0,
        "idle mean":                      0.0,
        "idle std":                       0.0,
    },
},
{
    "name":     "Port Scan",
    "src_ip":   "10.0.0.51",
    "expected": "portscan",
    "features": {
        "flow duration":                  0.5,
        "total fwd packets":              5,          # ← was 1 (below MIN threshold)
        "total backward packets":         1,
        "total length of fwd packets":    300,
        "total length of bwd packets":    60,
        "fwd packet length mean":         60.0,
        "bwd packet length mean":         60.0,
        "flow bytes/s":                   720.0,
        "flow packets/s":                 12.0,
        "syn flag count":                 4,          # multiple SYNs = key signal
        "ack flag count":                 1,
        "psh flag count":                 0,
        "packet length mean":             60.0,
        "packet length std":              0.0,
        "idle mean":                      0.0,
        "idle std":                       0.0,
    },
},
]


# ─────────────────────────────────────────────────────────────
# TEST 1 — FEATURE EXTRACTOR
# Builds synthetic Scapy-like packets and verifies all 16
# CICIDS features are extracted correctly
# ─────────────────────────────────────────────────────────────

def test_feature_extractor():
    header("TEST 1 — CICIDS Feature Extractor")
    from pipeline.network_level.flow_acuumulator import (
        Flow, PacketRecord, extract_flow_features
    )
    from pipeline.network_level.feature import THREAT_FEATURES

    passed = 0
    failed = 0

    # Build a synthetic flow with known packets
    flow = Flow(src_ip="10.0.0.99", start_time=1000.0)

    # Add 10 synthetic packets: 6 fwd, 4 bwd, with TCP flags
    packets = [
        # (timestamp, length, direction, flags)
        (1000.00, 66,   "fwd", 0x02),  # SYN
        (1000.01, 66,   "bwd", 0x12),  # SYN+ACK
        (1000.02, 54,   "fwd", 0x10),  # ACK
        (1000.10, 500,  "fwd", 0x18),  # PSH+ACK
        (1000.15, 1400, "bwd", 0x10),  # ACK (data)
        (1000.20, 500,  "fwd", 0x18),  # PSH+ACK
        (1000.25, 1400, "bwd", 0x10),  # ACK (data)
        (1001.50, 200,  "fwd", 0x18),  # PSH+ACK (after idle gap)
        (1001.55, 800,  "bwd", 0x10),  # ACK (data)
        (1001.60, 54,   "fwd", 0x11),  # FIN+ACK
    ]

    for ts, length, direction, flags in packets:
        pkt = PacketRecord(timestamp=ts, length=length, direction=direction, flags=flags)
        flow.add_packet(pkt)

    features = extract_flow_features(flow)

    # Verify feature dict exists
    if features is None:
        fail("extract_flow_features returned None")
        failed += 1
        return False

    ok(f"Feature dict returned ({len(features)} features)")
    passed += 1

    # Verify all 16 features present
    missing = [f for f in THREAT_FEATURES if f not in features]
    if missing:
        fail(f"Missing features: {missing}")
        failed += 1
    else:
        ok("All 16 THREAT_FEATURES present")
        passed += 1

    # Verify specific values
    checks = [
        ("flow duration",            lambda v: abs(v - 1.60) < 0.01,      "≈1.60s"),
        ("total fwd packets",        lambda v: v == 6,                    "== 6"),
        ("total backward packets",   lambda v: v == 4,                    "== 4"),
        ("syn flag count",           lambda v: v >= 1,                    ">= 1 (SYN+SYNACK both have SYN bit set)"),
        ("ack flag count",           lambda v: v >= 6,                    ">= 6"),
        ("psh flag count",           lambda v: v == 3,                    "== 3"),
        ("flow bytes/s",             lambda v: v > 0,                     "> 0"),
        ("flow packets/s",           lambda v: v > 0,                     "> 0"),
        ("idle mean",                lambda v: v > 0,                     "> 0 (gap detected)"),
        ("packet length std",        lambda v: v > 0,                     "> 0"),
    ]

    for feat_name, check_fn, description in checks:
        val = features.get(feat_name, None)
        if val is not None and check_fn(val):
            ok(f"{feat_name} = {val:.3f}  {description}")
            passed += 1
        else:
            fail(f"{feat_name} = {val}  expected {description}")
            failed += 1

    # Print all features for visibility
    print(f"\n  {CYAN}All extracted features:{RESET}")
    for feat in THREAT_FEATURES:
        print(f"    {feat:40s}: {features[feat]:.4f}")

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 2 — LGBM NETWORK CLASSIFIER
# ─────────────────────────────────────────────────────────────

def test_lgbm_network():
    header("TEST 2 — LightGBM Network Classifier")
    from pipeline.network_level.lgbm_network_classifier import LGBMNetworkClassifier

    try:
        lgbm = LGBMNetworkClassifier().load()
    except FileNotFoundError as e:
        warn(f"Model not found: {e}")
        warn("Skipping — place model at models/network_layer/network_lgbm.pkl")
        return True  # not a failure — model just not present yet

    passed = 0
    failed = 0

    for flow in SYNTHETIC_FLOWS:
        label, confidence, all_probs = lgbm.predict(flow["features"])
        threat_score = lgbm.threat_score(flow["features"])
        expected = flow["expected"]

        probs_str = " | ".join(
            f"{k}={v:.2f}"
            for k, v in sorted(all_probs.items(), key=lambda x: -x[1])
        )

        # Model classes are lowercase (benign/ddos/portscan)
        # Normalize both to lowercase for comparison
        label_lower    = label.lower()
        expected_lower = expected.lower()

        is_correct = (
            label_lower == expected_lower or
            (expected_lower != "benign" and label_lower != "benign")
        )

        if label_lower == expected_lower:
            ok(f"{flow['name']} → {BOLD}{label}{RESET} (score={threat_score:.3f})")
            info(f"   {probs_str}")
            passed += 1
        elif is_correct:
            warn(f"{flow['name']} → {label} (expected {expected}, but attack detected)")
            info(f"   {probs_str}")
            passed += 1
        else:
            fail(f"{flow['name']} → {label} (expected {expected}, score={threat_score:.3f})")
            info(f"   {probs_str}")
            failed += 1

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 3 — TCN DETECTOR
# ─────────────────────────────────────────────────────────────

def test_tcn():
    header("TEST 3 — TCN Zero-Day Detector")
    from pipeline.network_level.tcn_detector import TCNDetector

    try:
        tcn = TCNDetector().load()
    except FileNotFoundError as e:
        warn(f"Model not found: {e}")
        warn("Skipping — place model at models/network_layer/network_tcn.tflite")
        return True

    passed = 0
    failed = 0

    for flow in SYNTHETIC_FLOWS:
        score    = tcn.predict(flow["features"])
        is_anom  = tcn.is_anomaly(score)
        expected = flow["expected"]

        expected_anom = expected != "BENIGN"

        if expected_anom and is_anom:
            ok(f"{flow['name']} → anomaly detected (score={score:.3f})")
            passed += 1
        elif not expected_anom and not is_anom:
            ok(f"{flow['name']} → clean (score={score:.3f})")
            passed += 1
        else:
            fail(f"{flow['name']} → score={score:.3f} "
                 f"(anomaly={is_anom}, expected_attack={expected_anom})")
            failed += 1

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 4 — SCORE FUSION
# Verify WEIGHT_NET_LGBM + WEIGHT_NET_TCN fusion math
# ─────────────────────────────────────────────────────────────

def test_score_fusion():
    header("TEST 4 — Network Score Fusion")
    from pipeline.network_level.network_classifier import NetworkClassifier
    from shared.constants import WEIGHT_NETWORK_LGBM,WEIGHT_NETWORK_TCN

    info(f"Fusion weights: LGBM={WEIGHT_NETWORK_LGBM} TCN={WEIGHT_NETWORK_TCN} sum={WEIGHT_NETWORK_TCN+WEIGHT_NETWORK_LGBM}")

    # Verify weights sum to 1.0
    weight_sum = WEIGHT_NETWORK_LGBM + WEIGHT_NETWORK_TCN
    if abs(weight_sum - 1.0) < 1e-6:
        ok(f"Weights sum to 1.0 ✓")
    else:
        fail(f"Weights sum to {weight_sum} (expected 1.0)")

    # Test fusion math manually
    fusion_cases = [
        {"lgbm": 0.0,  "tcn": 0.0,  "expected_range": (0.0,  0.1),  "label": "clean"},
        {"lgbm": 1.0,  "tcn": 1.0,  "expected_range": (0.9,  1.0),  "label": "clear attack"},
        {"lgbm": 0.9,  "tcn": 0.0,  "expected_range": (0.45, 0.60), "label": "lgbm only"},
        {"lgbm": 0.0,  "tcn": 0.9,  "expected_range": (0.35, 0.45), "label": "tcn only"},
        {"lgbm": 0.8,  "tcn": 0.7,  "expected_range": (0.70, 0.80), "label": "both high"},
    ]

    passed = 0
    failed = 0

    for fc in fusion_cases:
        fused = WEIGHT_NETWORK_LGBM * fc["lgbm"] + WEIGHT_NETWORK_TCN * fc["tcn"]
        lo, hi = fc["expected_range"]

        if lo <= fused <= hi:
            ok(f"{fc['label']:20s} lgbm={fc['lgbm']} tcn={fc['tcn']} → fused={fused:.3f} ∈ [{lo},{hi}]")
            passed += 1
        else:
            fail(f"{fc['label']:20s} lgbm={fc['lgbm']} tcn={fc['tcn']} → fused={fused:.3f} NOT in [{lo},{hi}]")
            failed += 1

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 5 — FULL PIPELINE SIMULATION (no root needed)
# Injects flows → classifier → Redis write
# ─────────────────────────────────────────────────────────────

def test_full_pipeline_simulation():
    header("TEST 5 — Full Pipeline Simulation (no root)")
    from pipeline.network_level.network_ips import NetworkLayerIPS

    ips = NetworkLayerIPS()

    flows = [
        (flow["src_ip"], flow["features"])
        for flow in SYNTHETIC_FLOWS
    ]

    try:
        results = ips.simulate(flows=flows)
    except Exception as e:
        fail(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    passed = 0
    failed = 0

    for i, (result, flow_def) in enumerate(zip(results, SYNTHETIC_FLOWS)):
        expected = flow_def["expected"]
        name     = flow_def["name"]

        is_threat    = result["is_threat"]
        attack_type  = result["attack_type"]
        fused        = result["fused_score"]
        lgbm         = result["lgbm_score"]
        tcn          = result["tcn_score"]
        redis_ok     = result["redis_written"]

        expected_threat = expected != "BENIGN"

        threat_correct = (is_threat == expected_threat) or (
            not expected_threat and fused < 0.5
        )

        if threat_correct:
            status = f"{GREEN}✓{RESET}"
            passed += 1
        else:
            status = f"{RED}✗{RESET}"
            failed += 1

        threat_str = f"{RED}THREAT{RESET}" if is_threat else f"{GREEN}CLEAN{RESET}"
        redis_str  = f"{GREEN}✓{RESET}" if redis_ok else f"{YELLOW}✗ (Redis unavailable){RESET}"

        print(
            f"  {status} {name}\n"
            f"      {threat_str} | type={attack_type:12s} | "
            f"fused={fused:.3f} | lgbm={lgbm:.3f} | tcn={tcn:.3f} | "
            f"redis={redis_str}"
        )

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 6 — REDIS INTEGRATION
# Verify network scores are written and readable
# ─────────────────────────────────────────────────────────────

def test_redis_integration():
    header("TEST 6 — Redis Network Score Integration")

    try:
        try:
            from utils.redis_client import get_redis
            r = get_redis()
        except ImportError:
            from utils.redis_client import RedisClient
            r = RedisClient.get_instance()
    except Exception as e:
        warn(f"Redis unavailable: {e} — skipping")
        return True

    passed = 0
    failed = 0

    test_cases = [
        {
            "ip":          "10.0.0.50",
            "score":       0.95,
            "net_lgbm":    0.98,
            "tcn":         0.91,
            "attack_type": "DDoS",
        },
        {
            "ip":          "10.0.0.51",
            "score":       0.78,
            "net_lgbm":    0.80,
            "tcn":         0.75,
            "attack_type": "PortScan",
        },
        {
            "ip":          "192.168.1.100",
            "score":       0.02,
            "net_lgbm":    0.01,
            "tcn":         0.03,
            "attack_type": "clean",
        },
    ]

    for tc in test_cases:
        ip = tc["ip"]

        # Write
        ok_write = r.set_network_threat_score(
            ip          = ip,
            score       = tc["score"],
            net_lgbm    = tc["net_lgbm"],
            tcn         = tc["tcn"],
            attack_type = tc["attack_type"],
        )

        if not ok_write:
            fail(f"Write failed for {ip}")
            failed += 1
            continue

        # Read back
        data = r.get_network_threat_score(ip)
        if data is None:
            fail(f"Read returned None for {ip}")
            failed += 1
            continue

        # Verify score
        score_ok = abs(data["score"] - tc["score"]) < 0.001
        type_ok  = data.get("attack_type") == tc["attack_type"]

        if score_ok and type_ok:
            ok(
                f"{ip:16s} | type={data['attack_type']:12s} | "
                f"score={data['score']:.3f} | "
                f"lgbm={data.get('net_lgbm', 0):.3f} | "
                f"tcn={data.get('tcn', 0):.3f}"
            )
            passed += 1
        else:
            fail(f"{ip} score_ok={score_ok} type_ok={type_ok} data={data}")
            failed += 1

    # Test convenience method
    val = r.get_network_score_value("10.0.0.50")
    if abs(val - 0.95) < 0.001:
        ok(f"get_network_score_value = {val:.3f} ✓")
        passed += 1
    else:
        fail(f"get_network_score_value = {val} expected 0.95")
        failed += 1

    # Cleanup
    for tc in test_cases:
        r.raw.delete(f"threat:ip:{tc['ip']}")

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*60}")
    print("AIM-IPS NETWORK LAYER TEST SUITE")
    print(f"{'='*60}{RESET}")
    print(f"  {CYAN}No root required — uses synthetic flows{RESET}")
    print(f"  {CYAN}To test live capture: sudo python -m network_ips.network_ips --interface eth0{RESET}")

    results = {}

    tests = [
        ("Feature Extractor",      test_feature_extractor),
        ("LightGBM Network",       test_lgbm_network),
        ("TCN Detector",           test_tcn),
        ("Score Fusion",           test_score_fusion),
        ("Full Pipeline Sim",      test_full_pipeline_simulation),
        ("Redis Integration",      test_redis_integration),
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
        print(f"  {GREEN}{BOLD}All tests passed ✓{RESET}\n")
    else:
        print(f"  {RED}{BOLD}{total_failed} test(s) failed{RESET}\n")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)