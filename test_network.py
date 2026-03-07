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
Real flows are loaded from: data_collector/data_sets/cicids/
"""

import sys
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path

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
# CICIDS COLUMN → THREAT_FEATURES mapping
#
# The CSV uses mixed-case / spaced column names.
# THREAT_FEATURES uses lowercase with spaces.
# We normalise CSV headers to lowercase+stripped to match.
# ─────────────────────────────────────────────────────────────

THREAT_FEATURES = [
    "flow duration",
    "total fwd packets",
    "total backward packets",
    "total length of fwd packets",
    "total length of bwd packets",
    "fwd packet length mean",
    "bwd packet length mean",
    "flow bytes/s",
    "flow packets/s",
    "syn flag count",
    "ack flag count",
    "psh flag count",
    "packet length mean",
    "packet length std",
    "idle mean",
    "idle std",
]

# How many real flows to sample per label class for the pipeline tests
FLOWS_PER_CLASS = 3

# Dataset directory (relative to project root)
CICIDS_DIR = Path("data_collector/data_sets/cicids")

# ─────────────────────────────────────────────────────────────
# NETWORK-LEVEL LABEL WHITELIST
#
# CICIDS contains both network-layer and application-layer attacks.
# We keep only labels that correspond to what the network classifier
# actually detects (LightGBM classes: benign / ddos / portscan,
# plus botnet which is also network-layer).
#
# Excluded (application-layer, handled by app-layer IPS):
#   web attack – brute force / sql injection / xss
#   ftp-patator, ssh-patator  (credential brute-force)
#   infiltration, heartbleed  (application exploits)
# ─────────────────────────────────────────────────────────────

NETWORK_LEVEL_LABELS = {
    "benign",
    "ddos",
    "portscan",
    "bot",       # Botnet C&C beaconing — network layer
}

def _is_network_level(expected: str) -> bool:
    """Return True only for labels the network classifier handles."""
    return expected.lower() in NETWORK_LEVEL_LABELS


# ─────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip all column names so they match THREAT_FEATURES."""
    df.columns = df.columns.str.strip().str.lower()
    return df


def _label_to_expected(raw_label: str) -> str:
    """
    Map raw CICIDS label string to the classifier's output labels.
    Returns one of: 'benign' | 'ddos' | 'portscan' | <other-attack>
    """
    lbl = raw_label.strip().lower()
    if lbl == "benign":
        return "BENIGN"
    if "ddos" in lbl or "dos" in lbl:
        return "ddos"
    if "portscan" in lbl or "port scan" in lbl:
        return "portscan"
    return lbl  # bot, infiltration, webattack, etc.


def load_cicids_flows(
    cicids_dir: Path = CICIDS_DIR,
    flows_per_class: int = FLOWS_PER_CLASS,
    seed: int = 42,
) -> list:
    """
    Load real flows from CICIDS CSV files.

    Scans all *.csv files in cicids_dir, normalises column names,
    maps to THREAT_FEATURES, and returns a list of dicts matching
    the SYNTHETIC_FLOWS schema used by the test suite:

        {
            "name":     str,
            "src_ip":   str,      # synthetic — "cicids:<label>:<row_idx>"
            "expected": str,      # BENIGN | ddos | portscan | …
            "features": dict,     # {feature_name: float, …}
        }
    """
    random.seed(seed)
    np.random.seed(seed)

    csv_files = sorted(cicids_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {cicids_dir.resolve()}\n"
            "Expected layout: data_collector/data_sets/cicids/*.csv"
        )

    print(f"  {CYAN}→{RESET} Found {len(csv_files)} CSV file(s) in {cicids_dir}")

    # ── Read all files, collect rows per label ────────────────
    label_buckets: dict = {}   # label_str → list of feature-dicts

    LABEL_COL_CANDIDATES = ["label", "attack_cat", "attack_category",
                             "attack", "class"]

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            warn(f"  Could not read {csv_path.name}: {e}")
            continue

        df = _normalise_cols(df)

        # Find label column
        label_col = None
        for c in LABEL_COL_CANDIDATES:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            warn(f"  No label column in {csv_path.name} — skipping")
            continue

        # Check all THREAT_FEATURES are present
        missing_feats = [f for f in THREAT_FEATURES if f not in df.columns]
        if missing_feats:
            warn(f"  {csv_path.name} missing features: {missing_feats[:5]}{'…' if len(missing_feats)>5 else ''} — skipping")
            continue

        # Drop rows with NaN/Inf in feature columns
        feat_df = df[THREAT_FEATURES].copy()
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        mask = ~feat_df.isnull().any(axis=1)
        feat_df = feat_df[mask]
        labels  = df[label_col][mask].astype(str)

        kept_labels    = []
        skipped_labels = []
        for raw_lbl in labels.unique():
            expected = _label_to_expected(raw_lbl)
            if not _is_network_level(expected):
                skipped_labels.append(raw_lbl.strip())
                continue
            rows = feat_df[labels == raw_lbl]
            if rows.empty:
                continue
            if expected not in label_buckets:
                label_buckets[expected] = []
            for _, row in rows.iterrows():
                label_buckets[expected].append(row.to_dict())
            kept_labels.append(raw_lbl.strip())

        info(f"  {csv_path.name} — kept: {kept_labels or ['(none)']} | "
             f"skipped (app-layer): {skipped_labels or ['(none)']}")

    if not label_buckets:
        raise RuntimeError(
            "Could not extract any flows from the CICIDS dataset.\n"
            "Check that the CSV files contain the required feature columns."
        )

    # ── Sample flows_per_class rows per label ─────────────────
    flows = []
    for expected, rows in sorted(label_buckets.items()):
        sample_n = min(flows_per_class, len(rows))
        sampled  = random.sample(rows, sample_n)
        for i, feat_dict in enumerate(sampled):
            # Coerce to float, replace any remaining NaN/Inf with 0
            clean = {}
            for k, v in feat_dict.items():
                try:
                    fv = float(v)
                    clean[k] = 0.0 if (np.isnan(fv) or np.isinf(fv)) else fv
                except (TypeError, ValueError):
                    clean[k] = 0.0

            flows.append({
                "name":     f"{expected} (row {i})",
                "src_ip":   f"cicids:{expected}:{i}",
                "expected": expected,
                "features": clean,
            })

    # Shuffle so benign and attacks are interleaved
    random.shuffle(flows)
    return flows


# ─────────────────────────────────────────────────────────────
# Lazy-load FLOWS — only parsed once
# ─────────────────────────────────────────────────────────────

_FLOWS_CACHE: list = []

def get_flows() -> list:
    global _FLOWS_CACHE
    if not _FLOWS_CACHE:
        header("Loading Real CICIDS Flows")
        try:
            _FLOWS_CACHE = load_cicids_flows()
        except FileNotFoundError as e:
            warn(str(e))
            warn("Falling back to SYNTHETIC_FLOWS")
            _FLOWS_CACHE = SYNTHETIC_FLOWS_FALLBACK
    return _FLOWS_CACHE


# ─────────────────────────────────────────────────────────────
# FALLBACK synthetic flows (used only if dataset is absent)
# ─────────────────────────────────────────────────────────────

SYNTHETIC_FLOWS_FALLBACK = [
    {
        "name":     "Normal web browsing (synthetic)",
        "src_ip":   "192.168.1.100",
        "expected": "BENIGN",
        "features": {
            "flow duration": 2.5, "total fwd packets": 8,
            "total backward packets": 6, "total length of fwd packets": 1200,
            "total length of bwd packets": 8000, "fwd packet length mean": 150.0,
            "bwd packet length mean": 1333.0, "flow bytes/s": 3680.0,
            "flow packets/s": 5.6, "syn flag count": 1, "ack flag count": 12,
            "psh flag count": 4, "packet length mean": 654.0,
            "packet length std": 580.0, "idle mean": 0.1, "idle std": 0.05,
        },
    },
    {
        "name":     "DDoS SYN flood (synthetic)",
        "src_ip":   "10.0.0.50",
        "expected": "ddos",
        "features": {
            "flow duration": 2.0, "total fwd packets": 5000,
            "total backward packets": 0, "total length of fwd packets": 300000,
            "total length of bwd packets": 0, "fwd packet length mean": 60.0,
            "bwd packet length mean": 0.0, "flow bytes/s": 150000.0,
            "flow packets/s": 2500.0, "syn flag count": 5000, "ack flag count": 0,
            "psh flag count": 0, "packet length mean": 60.0,
            "packet length std": 2.0, "idle mean": 0.0, "idle std": 0.0,
        },
    },
    {
        "name":     "Port Scan (synthetic)",
        "src_ip":   "10.0.0.51",
        "expected": "portscan",
        "features": {
            "flow duration": 0.5, "total fwd packets": 5,
            "total backward packets": 1, "total length of fwd packets": 300,
            "total length of bwd packets": 60, "fwd packet length mean": 60.0,
            "bwd packet length mean": 60.0, "flow bytes/s": 720.0,
            "flow packets/s": 12.0, "syn flag count": 4, "ack flag count": 1,
            "psh flag count": 0, "packet length mean": 60.0,
            "packet length std": 0.0, "idle mean": 0.0, "idle std": 0.0,
        },
    },
]


# ─────────────────────────────────────────────────────────────
# TEST 1 — FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────

def test_feature_extractor():
    header("TEST 1 — CICIDS Feature Extractor")
    from pipeline.network_level.flow_acuumulator import (
        Flow, PacketRecord, extract_flow_features
    )
    from pipeline.network_level.feature import THREAT_FEATURES as MODEL_FEATURES

    passed = 0
    failed = 0

    flow = Flow(src_ip="10.0.0.99", start_time=1000.0)

    packets = [
        (1000.00, 66,   "fwd", 0x02),
        (1000.01, 66,   "bwd", 0x12),
        (1000.02, 54,   "fwd", 0x10),
        (1000.10, 500,  "fwd", 0x18),
        (1000.15, 1400, "bwd", 0x10),
        (1000.20, 500,  "fwd", 0x18),
        (1000.25, 1400, "bwd", 0x10),
        (1001.50, 200,  "fwd", 0x18),
        (1001.55, 800,  "bwd", 0x10),
        (1001.60, 54,   "fwd", 0x11),
    ]

    for ts, length, direction, flags in packets:
        pkt = PacketRecord(timestamp=ts, length=length, direction=direction, flags=flags)
        flow.add_packet(pkt)

    features = extract_flow_features(flow)

    if features is None:
        fail("extract_flow_features returned None")
        return False

    ok(f"Feature dict returned ({len(features)} features)")
    passed += 1

    missing = [f for f in MODEL_FEATURES if f not in features]
    if missing:
        fail(f"Missing features: {missing}")
        failed += 1
    else:
        ok("All 16 THREAT_FEATURES present")
        passed += 1

    checks = [
        ("flow duration",          lambda v: abs(v - 1.60 * 1_000_000) < 10_000, "≈1.60s (in µs)"),
        ("total fwd packets",      lambda v: v == 6,                              "== 6"),
        ("total backward packets", lambda v: v == 4,                              "== 4"),
        ("syn flag count",         lambda v: v >= 1,                              ">= 1"),
        ("ack flag count",         lambda v: v >= 6,                              ">= 6"),
        ("psh flag count",         lambda v: v == 3,                              "== 3"),
        ("flow bytes/s",           lambda v: v > 0,                               "> 0"),
        ("flow packets/s",         lambda v: v > 0,                               "> 0"),
        ("idle mean",              lambda v: v > 0,                               "> 0 (gap detected)"),
        ("packet length std",      lambda v: v > 0,                               "> 0"),
    ]

    for feat_name, check_fn, description in checks:
        val = features.get(feat_name)
        if val is not None and check_fn(val):
            ok(f"{feat_name} = {val:.3f}  {description}")
            passed += 1
        else:
            fail(f"{feat_name} = {val}  expected {description}")
            failed += 1

    print(f"\n  {CYAN}All extracted features:{RESET}")
    for feat in MODEL_FEATURES:
        print(f"    {feat:40s}: {features[feat]:.4f}")

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 2 — LGBM NETWORK CLASSIFIER
# ─────────────────────────────────────────────────────────────

def test_lgbm_network():
    header("TEST 2 — LightGBM Network Classifier (real CICIDS flows)")
    from pipeline.network_level.lgbm_network_classifier import LGBMNetworkClassifier

    try:
        lgbm = LGBMNetworkClassifier().load()
    except FileNotFoundError as e:
        warn(f"Model not found: {e}")
        warn("Skipping — place model at models/network_layer/network_lgbm.pkl")
        return True

    flows = get_flows()
    passed = 0
    failed = 0

    for flow in flows:
        label, confidence, all_probs = lgbm.predict(flow["features"])
        threat_score = lgbm.threat_score(flow["features"])
        expected = flow["expected"]

        probs_str = " | ".join(
            f"{k}={v:.2f}"
            for k, v in sorted(all_probs.items(), key=lambda x: -x[1])
        )

        label_lower    = label.lower()
        expected_lower = expected.lower()

        if label_lower == expected_lower:
            ok(f"{flow['name']} → {BOLD}{label}{RESET} (score={threat_score:.3f})")
            info(f"   {probs_str}")
            passed += 1
        elif expected_lower != "benign" and label_lower != "benign":
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
    header("TEST 3 — TCN Zero-Day Detector (real CICIDS flows)")
    from pipeline.network_level.tcn_detector import TCNDetector

    try:
        tcn = TCNDetector().load()
    except FileNotFoundError as e:
        warn(f"Model not found: {e}")
        warn("Skipping — place model at models/network_layer/network_tcn.tflite")
        return True

    flows = get_flows()
    passed = 0
    failed = 0

    for flow in flows:
        score    = tcn.predict(flow["features"])
        is_anom  = tcn.is_anomaly(score)
        expected = flow["expected"]
        expected_anom = expected.upper() != "BENIGN"

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
# ─────────────────────────────────────────────────────────────

def test_score_fusion():
    header("TEST 4 — Network Score Fusion")
    from pipeline.network_level.network_classifier import NetworkClassifier
    from shared.constants import WEIGHT_NETWORK_LGBM, WEIGHT_NETWORK_TCN

    info(f"Fusion weights: LGBM={WEIGHT_NETWORK_LGBM} TCN={WEIGHT_NETWORK_TCN} "
         f"sum={WEIGHT_NETWORK_TCN + WEIGHT_NETWORK_LGBM}")

    weight_sum = WEIGHT_NETWORK_LGBM + WEIGHT_NETWORK_TCN
    if abs(weight_sum - 1.0) < 1e-6:
        ok(f"Weights sum to 1.0 ✓")
    else:
        fail(f"Weights sum to {weight_sum} (expected 1.0)")

    fusion_cases = [
        {"lgbm": 0.0, "tcn": 0.0, "expected_range": (0.0,  0.1),  "label": "clean"},
        {"lgbm": 1.0, "tcn": 1.0, "expected_range": (0.9,  1.0),  "label": "clear attack"},
        {"lgbm": 0.9, "tcn": 0.0, "expected_range": (0.45, 0.60), "label": "lgbm only"},
        {"lgbm": 0.0, "tcn": 0.9, "expected_range": (0.35, 0.45), "label": "tcn only"},
        {"lgbm": 0.8, "tcn": 0.7, "expected_range": (0.70, 0.80), "label": "both high"},
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
# ─────────────────────────────────────────────────────────────

def test_full_pipeline_simulation():
    header("TEST 5 — Full Pipeline Simulation (real CICIDS flows, no root)")
    from pipeline.network_level.network_ips import NetworkLayerIPS

    ips = NetworkLayerIPS()

    real_flows = get_flows()
    flows_input = [(f["src_ip"], f["features"]) for f in real_flows]

    try:
        results = ips.simulate(flows=flows_input)
    except Exception as e:
        fail(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    passed = 0
    failed = 0

    # Print summary header
    print(f"\n  {'Flow':<35} {'Expected':<12} {'Got':<12} {'Fused':>7} {'LGBM':>7} {'TCN':>7}  Redis")
    print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*7} {'─'*7} {'─'*7}  ─────")

    for result, flow_def in zip(results, real_flows):
        expected      = flow_def["expected"]
        name          = flow_def["name"]
        is_threat     = result["is_threat"]
        attack_type   = result["attack_type"]
        fused         = result["fused_score"]
        lgbm          = result["lgbm_score"]
        tcn           = result["tcn_score"]
        redis_ok      = result["redis_written"]

        expected_threat = expected.upper() != "BENIGN"
        threat_correct  = is_threat == expected_threat or (not expected_threat and fused < 0.5)

        status_sym   = f"{GREEN}✓{RESET}" if threat_correct else f"{RED}✗{RESET}"
        redis_str    = f"{GREEN}✓{RESET}" if redis_ok else f"{YELLOW}✗{RESET}"
        threat_label = f"{RED}THREAT{RESET}" if is_threat else f"{GREEN}CLEAN {RESET}"

        # Truncate long names for display
        short_name = name[:33] + ".." if len(name) > 35 else name

        print(
            f"  {status_sym} {short_name:<35} "
            f"{expected:<12} {attack_type:<12} "
            f"{fused:>7.3f} {lgbm:>7.3f} {tcn:>7.3f}  {redis_str}"
        )

        if threat_correct:
            passed += 1
        else:
            failed += 1

    # Accuracy summary
    total = passed + failed
    acc = passed / total * 100 if total > 0 else 0
    print(f"\n  Accuracy: {acc:.1f}% ({passed}/{total})")
    print(f"  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 6 — REDIS INTEGRATION
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
        {"ip": "10.0.0.50", "score": 0.95, "net_lgbm": 0.98, "tcn": 0.91, "attack_type": "DDoS"},
        {"ip": "10.0.0.51", "score": 0.78, "net_lgbm": 0.80, "tcn": 0.75, "attack_type": "PortScan"},
        {"ip": "192.168.1.100", "score": 0.02, "net_lgbm": 0.01, "tcn": 0.03, "attack_type": "clean"},
    ]

    for tc in test_cases:
        ip = tc["ip"]
        ok_write = r.set_network_threat_score(
            ip=ip, score=tc["score"], net_lgbm=tc["net_lgbm"],
            tcn=tc["tcn"], attack_type=tc["attack_type"],
        )
        if not ok_write:
            fail(f"Write failed for {ip}")
            failed += 1
            continue

        data = r.get_network_threat_score(ip)
        if data is None:
            fail(f"Read returned None for {ip}")
            failed += 1
            continue

        score_ok = abs(data["score"] - tc["score"]) < 0.001
        type_ok  = data.get("attack_type") == tc["attack_type"]

        if score_ok and type_ok:
            ok(f"{ip:16s} | type={data['attack_type']:12s} | score={data['score']:.3f} | "
               f"lgbm={data.get('net_lgbm', 0):.3f} | tcn={data.get('tcn', 0):.3f}")
            passed += 1
        else:
            fail(f"{ip} score_ok={score_ok} type_ok={type_ok} data={data}")
            failed += 1

    val = r.get_network_score_value("10.0.0.50")
    if abs(val - 0.95) < 0.001:
        ok(f"get_network_score_value = {val:.3f} ✓")
        passed += 1
    else:
        fail(f"get_network_score_value = {val} expected 0.95")
        failed += 1

    for tc in test_cases:
        r.raw.delete(f"threat:ip:{tc['ip']}")

    print(f"\n  Results: {GREEN}{passed} passed{RESET} | {RED}{failed} failed{RESET}")
    return failed == 0


# ─────────────────────────────────────────────────────────────
# TEST 7 — DATASET LOADER SELF-TEST
# Verifies the CICIDS loader works and reports class distribution
# ─────────────────────────────────────────────────────────────

def test_dataset_loader():
    header("TEST 7 — CICIDS Dataset Loader")

    try:
        flows = load_cicids_flows(flows_per_class=5)
    except FileNotFoundError as e:
        warn(str(e))
        warn("Dataset not present — skipping loader test")
        return True
    except Exception as e:
        fail(f"Loader raised unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    if not flows:
        fail("No flows returned from loader")
        return False

    ok(f"Loaded {len(flows)} real flows from CICIDS dataset")

    # Verify schema
    for i, flow in enumerate(flows[:3]):
        schema_ok = all(k in flow for k in ("name", "src_ip", "expected", "features"))
        feat_ok   = all(f in flow["features"] for f in THREAT_FEATURES)
        if schema_ok and feat_ok:
            ok(f"Flow [{i}] '{flow['name']}' — schema ✓ | expected={flow['expected']}")
        else:
            fail(f"Flow [{i}] bad schema: schema_ok={schema_ok} feat_ok={feat_ok}")
            return False

    # Class distribution (network-level only — app-layer already filtered)
    from collections import Counter
    dist = Counter(f["expected"] for f in flows)
    print(f"\n  {CYAN}Network-level class distribution:{RESET}")
    for lbl, cnt in sorted(dist.items()):
        bar = "█" * cnt
        print(f"    {lbl:<12} {bar} ({cnt})")

    # Spot-check feature values are finite floats
    bad_values = 0
    for flow in flows:
        for feat, val in flow["features"].items():
            if not isinstance(val, float) or np.isnan(val) or np.isinf(val):
                warn(f"  Non-finite value: {flow['name']} → {feat}={val}")
                bad_values += 1

    if bad_values == 0:
        ok("All feature values are finite floats ✓")
    else:
        fail(f"{bad_values} non-finite feature values found")
        return False

    return True


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*60}")
    print("AIM-IPS NETWORK LAYER TEST SUITE")
    print(f"{'='*60}{RESET}")
    print(f"  {CYAN}No root required — uses real CICIDS flows{RESET}")
    print(f"  {CYAN}Dataset: {CICIDS_DIR.resolve()}{RESET}")
    print(f"  {CYAN}Flows per class: {FLOWS_PER_CLASS}{RESET}")

    results = {}

    tests = [
        ("Dataset Loader",         test_dataset_loader),
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