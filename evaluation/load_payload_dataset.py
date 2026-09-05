#!/usr/bin/env python3
"""
Payload Dataset Loader for Evaluation Framework

Loads payload_full.csv and computes s_app scores using the frozen LightGBM classifier.

Usage:
    python -m evaluation.load_payload_dataset --precompute
    python -m evaluation.load_payload_dataset --evaluate
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.harness import LabelledRequest
from evaluation.strategies.base import derive_context_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
PAYLOAD_CSV_PATH = Path("data_collector/payload_full.csv")
SCORED_OUTPUT_PATH = Path("evaluation/data/payload_scored.csv")


def precompute_scores(
    input_path: Path = PAYLOAD_CSV_PATH,
    output_path: Path = SCORED_OUTPUT_PATH,
) -> Path:
    """
    Pre-compute s_app scores using the frozen LightGBM classifier.
    """
    import pandas as pd
    from shared.schemas import RequestContext
    from threat_classifier.lgbm_classifier import LGBMAppClassifier

    # Load classifier
    classifier = LGBMAppClassifier()
    classifier.load()
    logger.info("LightGBM classifier loaded")

    # Load dataset
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    logger.info(f"Columns: {list(df.columns)}")

    # Check columns
    payload_col = "payload" if "payload" in df.columns else "Query"
    label_col = "label" if "label" in df.columns else "Label"
    attack_type_col = "attack_type" if "attack_type" in df.columns else None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, row in df.iterrows():
        payload = str(row[payload_col]) if pd.notna(row[payload_col]) else ""

        # Determine label
        label_val = row[label_col]
        if isinstance(label_val, str):
            is_attack = label_val.lower() not in ["norm", "normal", "benign", "0", "false"]
            attack_type = label_val.lower() if is_attack else "benign"
        else:
            is_attack = bool(int(label_val))
            attack_type = "attack" if is_attack else "benign"

        # Get attack type if available
        if attack_type_col and attack_type_col in df.columns:
            attack_type = str(row[attack_type_col]).lower()

        # Simulate different endpoints based on attack type
        endpoint_map = {
            "sqli": "/api/users/{id}",
            "xss": "/api/search",
            "cmdi": "/api/admin/exec",
            "path-traversal": "/api/files/{id}",
            "norm": "/api/products",
            "benign": "/api/products",
        }
        path = endpoint_map.get(attack_type, "/api/default")
        method = "POST" if len(payload) > 50 else "GET"

        # Build context and get score
        ctx = RequestContext(
            ip="0.0.0.0",
            method=method,
            path=path,
            body=payload,
        )

        try:
            layer_score = classifier.predict(ctx)
            s_app = layer_score.score
            pred_label = layer_score.label
            pred_conf = layer_score.confidence
        except Exception as e:
            if idx < 5:
                logger.warning(f"Error scoring row {idx}: {e}")
            s_app = 0.5
            pred_label = "error"
            pred_conf = 0.0

        rows.append({
            "request_idx": idx,
            "timestamp_s": float(idx) * 0.1,  # 10 requests per second
            "path": path,
            "method": method,
            "context_key": derive_context_key(path, method),
            "payload_len": len(payload),
            "s_app": round(s_app, 6),
            "pred_label": pred_label,
            "pred_conf": round(pred_conf, 4),
            "is_attack": int(is_attack),
            "attack_type": attack_type,
        })

        if (idx + 1) % 5000 == 0:
            logger.info(f"Scored {idx + 1}/{len(df)} entries...")

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} scored entries to {output_path}")

    # Summary
    n_attacks = sum(r["is_attack"] for r in rows)
    n_benign = len(rows) - n_attacks

    attack_rows = [r for r in rows if r["is_attack"]]
    benign_rows = [r for r in rows if not r["is_attack"]]

    mean_score_attack = np.mean([r["s_app"] for r in attack_rows]) if attack_rows else 0
    mean_score_benign = np.mean([r["s_app"] for r in benign_rows]) if benign_rows else 0

    logger.info(f"\nDataset summary:")
    logger.info(f"  Total: {len(rows)}")
    logger.info(f"  Attacks: {n_attacks} ({100*n_attacks/len(rows):.1f}%)")
    logger.info(f"  Benign: {n_benign} ({100*n_benign/len(rows):.1f}%)")
    logger.info(f"  Mean s_app (attack): {mean_score_attack:.4f}")
    logger.info(f"  Mean s_app (benign): {mean_score_benign:.4f}")

    # Attack type breakdown
    attack_types = {}
    for r in rows:
        t = r["attack_type"]
        attack_types[t] = attack_types.get(t, 0) + 1
    logger.info(f"  Attack types: {attack_types}")

    return output_path


def load_scored_payload(
    path: Path = SCORED_OUTPUT_PATH,
    requests_per_second: float = 10.0,
) -> Iterator[LabelledRequest]:
    """Load pre-scored payload data for evaluation."""
    import pandas as pd

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} scored entries from {path}")

    # Assign sequential time
    interval = 1.0 / requests_per_second

    for idx, row in df.iterrows():
        yield LabelledRequest(
            request_idx=int(idx),
            stream_time_s=float(idx) * interval,
            s_app=float(row["s_app"]),
            is_attack=bool(row["is_attack"]),
            path=str(row["path"]),
            method=str(row["method"]),
        )


def split_for_evaluation(
    path: Path = SCORED_OUTPUT_PATH,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[LabelledRequest], List[LabelledRequest]]:
    """Split into validation (for fitting theta_global) and test sets."""
    all_requests = list(load_scored_payload(path))
    n = len(all_requests)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    val_end = int(n * val_ratio)
    val_indices = sorted(indices[:val_end])
    test_indices = sorted(indices[val_end:])

    val_list = [all_requests[i] for i in val_indices]
    test_list = [all_requests[i] for i in test_indices]

    logger.info(f"Split: validation={len(val_list)}, test={len(test_list)}")
    return val_list, test_list


def run_evaluation(
    scored_path: Path = SCORED_OUTPUT_PATH,
    output_dir: Path = Path("evaluation/results/payload"),
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full evaluation on payload dataset."""
    from evaluation.strategies import (
        FixedThresholdStrategy,
        ValidationOptimisedStrategy,
        ContextAdaptiveStrategy,
    )
    from evaluation.harness import ReplayHarness
    from evaluation.data_loader import fit_validation_threshold
    from evaluation.summarize import generate_full_report

    output_dir.mkdir(parents=True, exist_ok=True)

    # Split data
    logger.info("Splitting data...")
    val_list, test_list = split_for_evaluation(scored_path, val_ratio, seed)

    # Fit theta_global on validation
    logger.info("Fitting validation threshold...")
    theta_global, fit_info = fit_validation_threshold(val_list, metric="f1")
    logger.info(f"Fitted theta_global = {theta_global:.4f} (F1 = {fit_info['best_score']:.4f})")

    # Save fit info
    with open(output_dir / "validation_fit.json", "w") as f:
        json.dump({
            "theta_global": theta_global,
            "fit_info": {k: v for k, v in fit_info.items() if k != "all_results"},
        }, f, indent=2)

    # Create strategies
    strategies = [
        FixedThresholdStrategy(theta=0.5),
        ValidationOptimisedStrategy(theta_global=theta_global),
        ContextAdaptiveStrategy(
            theta_global=theta_global,
            alpha=0.02,
            margin=0.02,
            delta=0.15,
            n_min=50,
            window_size=500,
            enable_anti_poisoning=True,
        ),
        ContextAdaptiveStrategy(
            theta_global=theta_global,
            alpha=0.02,
            margin=0.02,
            delta=0.15,
            n_min=50,
            window_size=500,
            enable_anti_poisoning=False,
        ),
    ]

    logger.info(f"Running evaluation with {len(strategies)} strategies on {len(test_list)} test samples")

    # Run harness
    harness = ReplayHarness(
        strategies=strategies,
        confirmation_delay_s=60.0,
        window_size_s=300.0,
        recall_floor=0.95,
        output_dir=output_dir,
    )

    summary = harness.run(iter(test_list))
    harness.save_results()

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)

    for name, stats in summary["strategies"].items():
        status = "PASS" if stats["passes_recall_floor"] else "FAIL"
        logger.info(
            f"\n{name}:\n"
            f"  F1={stats['f1']:.4f}, Precision={stats['precision']:.4f}, "
            f"Recall={stats['recall']:.4f} [{status}]\n"
            f"  FPR={stats['fpr']:.4f}, FNR={stats['fnr']:.4f}\n"
            f"  Latency: mean={stats['mean_latency_us']:.0f}μs, "
            f"p99={stats['p99_latency_us']:.0f}μs"
        )

    # Generate summary report
    generate_full_report(output_dir, output_dir / "summary", recall_floor=0.95)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Payload dataset evaluation")
    parser.add_argument("--precompute", action="store_true", help="Pre-compute s_app scores")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--input", type=Path, default=PAYLOAD_CSV_PATH)
    parser.add_argument("--scored", type=Path, default=SCORED_OUTPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/payload"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.precompute:
        precompute_scores(args.input, args.scored)

    if args.evaluate:
        if not args.scored.exists():
            logger.error(f"Scored data not found: {args.scored}")
            logger.error("Run with --precompute first")
            sys.exit(1)
        run_evaluation(args.scored, args.output_dir, seed=args.seed)

    if not args.precompute and not args.evaluate:
        parser.print_help()
