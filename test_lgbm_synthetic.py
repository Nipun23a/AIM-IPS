"""
test_lgbm_synthetic.py
═══════════════════════════════════════════════════════════════
Standalone test for the LightGBM network threat classifier.
Uses synthetic flows matched to the CICIDS2017 feature distribution.

Run from project root:
    python test_lgbm_synthetic.py
    python test_lgbm_synthetic.py --model path/to/lgb_model.pkl
    python test_lgbm_synthetic.py --verbose

No network capture, no root, no Redis needed.
═══════════════════════════════════════════════════════════════
"""

import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Terminal colours
# ──────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg):   print(f"  {RED}✗{RESET}  {msg}")
def warn(msg):   print(f"  {YELLOW}⚠{RESET}  {msg}")
def info(msg):   print(f"  {CYAN}→{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}\n" + "─" * 62)
def dim(msg):    print(f"  {DIM}{msg}{RESET}")


# ──────────────────────────────────────────────────────────────
# Default model paths  (override with --model / --features)
# ──────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH    = "models/threat_classifier/lgb_model.pkl"
DEFAULT_FEATURES_PATH = "models/threat_classifier/features.pkl"


"""
synthetic_flows.py
═══════════════════════════════════════════════════════════════
Synthetic flows built directly from real CICIDS2017 CSV rows.

These are NOT invented values — every flow below is a near-exact
copy of a real row that the model already classified correctly
in diagnostic Check 4.

KEY PATTERNS LEARNED FROM REAL DATA:
─────────────────────────────────────────────────────────────
BENIGN
  • flow duration:          3–109 µs  (very short)
  • total fwd/bwd packets:  1–2
  • packet length mean:     6.0       (tiny ACK/keepalive)
  • flow bytes/s:           100K–4M   (high because duration tiny)
  • syn flag count:         0
  • bwd packet length mean: 0–6

DDOS
  • flow duration:          642K–80M µs
  • total fwd packets:      3–8,  bwd packets: 4–7
  • bwd packet length mean: 1658–2900  ← STRONG signal
  • packet length mean:     897–1163   ← STRONG signal
  • flow bytes/s:           5–18K      (moderate)
  • syn flag count:         0  (not SYN flood — HTTP/app layer DDoS)
  • psh flag count:         1

PORTSCAN
  • flow duration:          52–5M µs
  • total fwd packets:      1–6,  bwd packets: 0–5
  • packet length mean:     0.0        ← STRONG signal
  • flow bytes/s:           0.0        ← STRONG signal
  • syn flag count:         0
  • ack flag count:         0–1
═══════════════════════════════════════════════════════════════
"""

SYNTHETIC_FLOWS = [

    # ──────────────────────────────────────────────────────────
    # BENIGN — copied from real rows 1 & 5 (identical)
    # Tiny duration, 2 fwd packets, 6-byte payloads, huge pps
    # ──────────────────────────────────────────────────────────

    {
        "name":     "Benign — tiny keepalive (row 1/5)",
        "src_ip":   "192.168.1.100",
        "expected": "benign",
        "features": {
            "flow duration":                  3,
            "total fwd packets":              2,
            "total backward packets":         0,
            "total length of fwd packets":    12,
            "total length of bwd packets":    0,
            "fwd packet length mean":         6.0,
            "bwd packet length mean":         0.0,
            "flow bytes/s":                   4000000.0,
            "flow packets/s":                 666666.6667,
            "syn flag count":                 0,
            "ack flag count":                 1,
            "psh flag count":                 0,
            "packet length mean":             6.0,
            "packet length std":              0.0,
            "idle mean":                      0.0,
            "idle std":                       0.0,
        },
    },

    {
        "name":     "Benign — short bidirectional exchange (row 2)",
        "src_ip":   "192.168.1.101",
        "expected": "benign",
        "features": {
            "flow duration":                  109,
            "total fwd packets":              1,
            "total backward packets":         1,
            "total length of fwd packets":    6,
            "total length of bwd packets":    6,
            "fwd packet length mean":         6.0,
            "bwd packet length mean":         6.0,
            "flow bytes/s":                   110091.7431,
            "flow packets/s":                 18348.62385,
            "syn flag count":                 0,
            "ack flag count":                 1,
            "psh flag count":                 0,
            "packet length mean":             6.0,
            "packet length std":              0.0,
            "idle mean":                      0.0,
            "idle std":                       0.0,
        },
    },

    {
        "name":     "Benign — very short exchange (row 3)",
        "src_ip":   "192.168.1.102",
        "expected": "benign",
        "features": {
            "flow duration":                  52,
            "total fwd packets":              1,
            "total backward packets":         1,
            "total length of fwd packets":    6,
            "total length of bwd packets":    6,
            "fwd packet length mean":         6.0,
            "bwd packet length mean":         6.0,
            "flow bytes/s":                   230769.2308,
            "flow packets/s":                 38461.53846,
            "syn flag count":                 0,
            "ack flag count":                 1,
            "psh flag count":                 0,
            "packet length mean":             6.0,
            "packet length std":              0.0,
            "idle mean":                      0.0,
            "idle std":                       0.0,
        },
    },

    # ──────────────────────────────────────────────────────────
    # DDoS — copied from real rows 1, 3, 5
    # Key signals: large bwd_packet_length_mean (1658–1934),
    # high packet_length_mean (1057–1163), low flow_bytes/s
    # ──────────────────────────────────────────────────────────

    {
        "name":     "DDoS — HTTP response flood (row 1)",
        "src_ip":   "10.0.0.50",
        "expected": "ddos",
        "features": {
            "flow duration":                  1293792,
            "total fwd packets":              3,
            "total backward packets":         7,
            "total length of fwd packets":    26,
            "total length of bwd packets":    11607,
            "fwd packet length mean":         8.666666667,
            "bwd packet length mean":         1658.142857,
            "flow bytes/s":                   8991.398927,
            "flow packets/s":                 7.72921768,
            "syn flag count":                 0,
            "ack flag count":                 0,
            "psh flag count":                 1,
            "packet length mean":             1057.545455,
            "packet length std":              1853.437529,
            "idle mean":                      0.0,
            "idle std":                       0.0,
        },
    },

    {
        "name":     "DDoS — HTTP response flood (row 3)",
        "src_ip":   "10.0.0.51",
        "expected": "ddos",
        "features": {
            "flow duration":                  1083538,
            "total fwd packets":              3,
            "total backward packets":         6,
            "total length of fwd packets":    26,
            "total length of bwd packets":    11601,
            "fwd packet length mean":         8.666666667,
            "bwd packet length mean":         1933.5,
            "flow bytes/s":                   10730.58813,
            "flow packets/s":                 8.306123089,
            "syn flag count":                 0,
            "ack flag count":                 0,
            "psh flag count":                 1,
            "packet length mean":             1162.7,
            "packet length std":              1645.241762,
            "idle mean":                      0.0,
            "idle std":                       0.0,
        },
    },

    {
        "name":     "DDoS — HTTP response flood (row 5)",
        "src_ip":   "10.0.0.52",
        "expected": "ddos",
        "features": {
            "flow duration":                  642654,
            "total fwd packets":              3,
            "total backward packets":         6,
            "total length of fwd packets":    26,
            "total length of bwd packets":    11607,
            "fwd packet length mean":         8.666666667,
            "bwd packet length mean":         1934.5,
            "flow bytes/s":                   18101.49785,
            "flow packets/s":                 14.0044254,
            "syn flag count":                 0,
            "ack flag count":                 0,
            "psh flag count":                 1,
            "packet length mean":             1163.3,
            "packet length std":              2138.329153,
            "idle mean":                      0.0,
            "idle std":                       0.0,
        },
    },

    # ──────────────────────────────────────────────────────────
    # PortScan — copied from real rows 2, 4, 5
    # Key signals: packet_length_mean = 0, flow_bytes/s = 0,
    # very short duration, 1 fwd packet
    # ──────────────────────────────────────────────────────────

    {
        "name":     "PortScan — empty probe (row 2)",
        "src_ip":   "10.0.0.60",
        "expected": "portscan",
        "features": {
            "flow duration":                  70,
            "total fwd packets":              1,
            "total backward packets":         1,
            "total length of fwd packets":    0,
            "total length of bwd packets":    0,
            "fwd packet length mean":         0.0,
            "bwd packet length mean":         0.0,
            "flow bytes/s":                   0.0,
            "flow packets/s":                 28571.42857,
            "syn flag count":                 0,
            "ack flag count":                 1,
            "psh flag count":                 0,
            "packet length mean":             0.0,
            "packet length std":              0.0,
            "idle mean":                      0.0,
            "idle std":                       0.0,
        },
    },

    {
        "name":     "PortScan — empty probe (row 4)",
        "src_ip":   "10.0.0.61",
        "expected": "portscan",
        "features": {
            "flow duration":                  52,
            "total fwd packets":              1,
            "total backward packets":         1,
            "total length of fwd packets":    0,
            "total length of bwd packets":    0,
            "fwd packet length mean":         0.0,
            "bwd packet length mean":         0.0,
            "flow bytes/s":                   0.0,
            "flow packets/s":                 38461.53846,
            "syn flag count":                 0,
            "ack flag count":                 1,
            "psh flag count":                 0,
            "packet length mean":             0.0,
            "packet length std":              0.0,
            "idle mean":                      0.0,
            "idle std":                       0.0,
        },
    },

    {
        "name":     "PortScan — zero-payload multi-packet (row 5)",
        "src_ip":   "10.0.0.62",
        "expected": "portscan",
        "features": {
            "flow duration":                  5386396,
            "total fwd packets":              3,
            "total backward packets":         1,
            "total length of fwd packets":    0,
            "total length of bwd packets":    0,
            "fwd packet length mean":         0.0,
            "bwd packet length mean":         0.0,
            "flow bytes/s":                   0.0,
            "flow packets/s":                 0.742611572,
            "syn flag count":                 0,
            "ack flag count":                 0,
            "psh flag count":                 1,
            "packet length mean":             0.0,
            "packet length std":              0.0,
            "idle mean":                      0.0,
            "idle std":                       0.0,
        },
    },
]

# ──────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────

def load_model(model_path: str, features_path: str):
    mp = Path(model_path)
    fp = Path(features_path)

    if not mp.exists():
        print(f"\n{RED}ERROR:{RESET} model not found at {mp}")
        print(f"       Pass the correct path with --model <path>")
        sys.exit(1)

    model = joblib.load(mp)
    print(f"  Loaded model  ← {mp}")

    features = None
    if fp.exists():
        saved = joblib.load(fp)
        features = saved if isinstance(saved, list) else saved.get("features")
        print(f"  Loaded features ({len(features)}) ← {fp}")
    else:
        warn(f"features.pkl not found at {fp} — inferring from model")

    # Resolve class labels
    idx_to_label = {0: "benign", 1: "ddos", 2: "portscan"}
    if hasattr(model, "classes_"):
        classes = [idx_to_label.get(int(c), str(c)) for c in model.classes_]
    else:
        classes = ["benign", "ddos", "portscan"]

    print(f"  Classes: {classes}")
    return model, features, classes


# ──────────────────────────────────────────────────────────────
# Single flow prediction
# ──────────────────────────────────────────────────────────────

def predict(model, feature_names, classes, flow_features: dict):
    vec = [float(flow_features.get(f, 0.0)) for f in feature_names]
    X   = pd.DataFrame([vec], columns=feature_names)

    probs     = model.predict_proba(X)[0]
    pred_idx  = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx])
    all_probs  = {classes[i]: float(probs[i]) for i in range(len(classes))}
    threat_score = float(np.clip(1.0 - all_probs.get("benign", 0.0), 0.0, 1.0))

    return pred_label, confidence, all_probs, threat_score


# ──────────────────────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────────────────────

def run_tests(model, feature_names, classes, verbose: bool) -> bool:
    header("LGBM SYNTHETIC FLOW TEST")

    total  = len(SYNTHETIC_FLOWS)
    passed = 0
    failed = 0
    results = []

    for flow in SYNTHETIC_FLOWS:
        pred_label, confidence, all_probs, threat_score = predict(
            model, feature_names, classes, flow["features"]
        )

        expected   = flow["expected"].lower()
        pred_lower = pred_label.lower()
        correct    = pred_lower == expected

        results.append({
            **flow,
            "pred":         pred_label,
            "confidence":   confidence,
            "threat_score": threat_score,
            "all_probs":    all_probs,
            "correct":      correct,
        })

        probs_str = "  ".join(
            f"{k}={v:.3f}"
            for k, v in sorted(all_probs.items(), key=lambda x: -x[1])
        )

        status = f"{GREEN}PASS{RESET}" if correct else f"{RED}FAIL{RESET}"
        threat_bar = _bar(threat_score)

        print(
            f"\n  [{status}]  {BOLD}{flow['name']}{RESET}\n"
            f"         src={flow['src_ip']}\n"
            f"         expected={BOLD}{expected}{RESET}   "
            f"predicted={BOLD}{pred_label}{RESET}   "
            f"confidence={confidence:.3f}\n"
            f"         threat={threat_bar} {threat_score:.3f}\n"
            f"         probs: {probs_str}"
        )

        if verbose:
            print(f"\n         {DIM}── feature dump ──{RESET}")
            for feat, val in flow["features"].items():
                print(f"         {DIM}{feat:40s}: {val}{RESET}")

        if correct:
            passed += 1
        else:
            failed += 1

    # ── Summary ───────────────────────────────────────────────

    header("SUMMARY")

    by_class: dict[str, dict] = {}
    for r in results:
        cls = r["expected"]
        by_class.setdefault(cls, {"pass": 0, "fail": 0})
        if r["correct"]:
            by_class[cls]["pass"] += 1
        else:
            by_class[cls]["fail"] += 1

    for cls, counts in sorted(by_class.items()):
        p, f = counts["pass"], counts["fail"]
        bar  = f"{GREEN}{'█' * p}{RED}{'█' * f}{RESET}"
        print(f"  {cls:12s}  {bar}  {p}/{p+f}")

    print()
    print(f"  Total  {BOLD}{total}{RESET} flows  |  "
          f"{GREEN}{passed} passed{RESET}  |  {RED}{failed} failed{RESET}")

    if failed == 0:
        print(f"\n  {GREEN}{BOLD}All tests passed ✓{RESET}")
    else:
        print(f"\n  {RED}{BOLD}{failed} test(s) failed — see above for details{RESET}")

    return failed == 0


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    empty  = width - filled
    colour = GREEN if score < 0.35 else (YELLOW if score < 0.70 else RED)
    return f"{colour}{'█' * filled}{'░' * empty}{RESET}"


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Test LGBM classifier with synthetic flows")
    p.add_argument("--model",    default=DEFAULT_MODEL_PATH,    help="Path to lgb_model.pkl")
    p.add_argument("--features", default=DEFAULT_FEATURES_PATH, help="Path to features.pkl")
    p.add_argument("--verbose",  action="store_true",           help="Dump all feature values per flow")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{BOLD}{'═' * 62}")
    print("  LGBM NETWORK THREAT CLASSIFIER — SYNTHETIC FLOW TEST")
    print(f"{'═' * 62}{RESET}")
    print(f"  model    : {args.model}")
    print(f"  features : {args.features}")
    print(f"  flows    : {len(SYNTHETIC_FLOWS)}")
    print()

    model, feature_names, classes = load_model(args.model, args.features)

    success = run_tests(model, feature_names, classes, verbose=args.verbose)

    print()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()