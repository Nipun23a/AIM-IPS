#!/usr/bin/env python3
"""
Honeypot Data Loader for Evaluation Framework

Loads honeypot_log.json and computes s_app scores using the frozen LightGBM classifier.
Outputs a scored dataset ready for the replay harness.

Usage:
    # Step 1: Pre-compute scores (run once)
    python -m evaluation.load_honeypot_data --precompute

    # Step 2: Run evaluation with scored data
    python -m evaluation.load_honeypot_data --evaluate
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Tuple
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.harness import LabelledRequest
from evaluation.strategies.base import derive_context_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default paths
HONEYPOT_LOG_PATH = Path("data_collector/honeypot_log.json")
SCORED_OUTPUT_PATH = Path("evaluation/data/honeypot_scored.csv")


def parse_tags_to_label(tags: List[str]) -> Tuple[bool, str, float]:
    """
    Parse honeypot tags to determine ground-truth label.

    Tags format: ["SQLi:0.25", "CmdInjection:0.5", ...]

    Returns:
        (is_attack, attack_type, max_confidence)
    """
    if not tags:
        return False, "benign", 0.0

    max_conf = 0.0
    attack_type = "unknown"

    for tag in tags:
        if ":" in tag:
            name, conf_str = tag.rsplit(":", 1)
            try:
                conf = float(conf_str)
                if conf > max_conf:
                    max_conf = conf
                    attack_type = name.lower()
            except ValueError:
                continue

    # Consider it an attack if any tag has confidence > threshold
    # Using 0.3 as threshold since your data shows SQLi:0.25 which might be FP
    ATTACK_THRESHOLD = 0.4
    is_attack = max_conf >= ATTACK_THRESHOLD

    return is_attack, attack_type, max_conf


def load_honeypot_raw(
    path: Path = HONEYPOT_LOG_PATH,
    attack_threshold: float = 0.4,
) -> Iterator[Dict[str, Any]]:
    """
    Load raw honeypot log entries.

    Args:
        path: Path to honeypot_log.json
        attack_threshold: Tag confidence threshold for attack classification

    Yields:
        Dict with parsed request data
    """
    with open(path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} entries from {path}")

    for idx, entry in enumerate(data):
        tags = entry.get("tags", [])
        is_attack, attack_type, confidence = parse_tags_to_label(tags)

        # Override threshold if needed
        if confidence < attack_threshold:
            is_attack = False

        # Parse timestamp
        ts_str = entry.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            timestamp_s = ts.timestamp()
        except:
            timestamp_s = float(idx)

        # Build payload from various sources
        payload_parts = []
        if entry.get("raw_body"):
            payload_parts.append(entry["raw_body"])
        if entry.get("path"):
            payload_parts.append(entry["path"])
        if entry.get("query_params"):
            qs = "&".join(f"{k}={v}" for k, v in entry["query_params"].items())
            if qs:
                payload_parts.append(qs)
        if entry.get("form_data"):
            fd = "&".join(f"{k}={v}" for k, v in entry["form_data"].items())
            if fd:
                payload_parts.append(fd)

        payload = " ".join(payload_parts)

        yield {
            "idx": idx,
            "timestamp_s": timestamp_s,
            "ip": entry.get("ip", "0.0.0.0"),
            "method": entry.get("method", "GET"),
            "path": entry.get("path", "/"),
            "payload": payload,
            "is_attack": is_attack,
            "attack_type": attack_type,
            "tag_confidence": confidence,
            "tags": tags,
            "headers": entry.get("headers", {}),
        }


def precompute_scores(
    input_path: Path = HONEYPOT_LOG_PATH,
    output_path: Path = SCORED_OUTPUT_PATH,
    attack_threshold: float = 0.4,
) -> Path:
    """
    Pre-compute s_app scores using the frozen LightGBM classifier.

    This runs the classifier ONCE on all data. The classifier is never
    modified during evaluation.

    Args:
        input_path: Path to honeypot_log.json
        output_path: Path for scored CSV output
        attack_threshold: Tag confidence threshold for attack classification

    Returns:
        Path to scored output file
    """
    from shared.schemas import RequestContext
    from threat_classifier.lgbm_classifier import LGBMAppClassifier

    # Load classifier
    classifier = LGBMAppClassifier()
    classifier.load()
    logger.info("LightGBM classifier loaded")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process all entries
    rows = []
    for entry in load_honeypot_raw(input_path, attack_threshold):
        # Build RequestContext for classifier
        ctx = RequestContext(
            ip=entry["ip"],
            method=entry["method"],
            path=entry["path"],
            body=entry["payload"],
            headers=entry["headers"],
        )

        # Get classifier prediction
        try:
            layer_score = classifier.predict(ctx)
            s_app = layer_score.score
            pred_label = layer_score.label
            pred_conf = layer_score.confidence
        except Exception as e:
            logger.warning(f"Error scoring entry {entry['idx']}: {e}")
            s_app = 0.5
            pred_label = "error"
            pred_conf = 0.0

        rows.append({
            "request_idx": entry["idx"],
            "timestamp_s": entry["timestamp_s"],
            "ip": entry["ip"],
            "method": entry["method"],
            "path": entry["path"],
            "context_key": derive_context_key(entry["path"], entry["method"]),
            "payload_len": len(entry["payload"]),
            "s_app": round(s_app, 6),
            "pred_label": pred_label,
            "pred_conf": round(pred_conf, 4),
            "is_attack": int(entry["is_attack"]),
            "attack_type": entry["attack_type"],
            "tag_confidence": entry["tag_confidence"],
            "tags": "|".join(entry["tags"]),
        })

        if (entry["idx"] + 1) % 500 == 0:
            logger.info(f"Scored {entry['idx'] + 1} entries...")

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} scored entries to {output_path}")

    # Print summary statistics
    n_attacks = sum(r["is_attack"] for r in rows)
    n_benign = len(rows) - n_attacks
    mean_score_attack = sum(r["s_app"] for r in rows if r["is_attack"]) / max(n_attacks, 1)
    mean_score_benign = sum(r["s_app"] for r in rows if not r["is_attack"]) / max(n_benign, 1)

    logger.info(f"\nDataset summary:")
    logger.info(f"  Total: {len(rows)}")
    logger.info(f"  Attacks: {n_attacks} ({100*n_attacks/len(rows):.1f}%)")
    logger.info(f"  Benign: {n_benign} ({100*n_benign/len(rows):.1f}%)")
    logger.info(f"  Mean s_app (attack): {mean_score_attack:.4f}")
    logger.info(f"  Mean s_app (benign): {mean_score_benign:.4f}")

    return output_path


def load_scored_honeypot(
    path: Path = SCORED_OUTPUT_PATH,
    time_mode: str = "original",  # "original", "sequential", "compressed"
    requests_per_second: float = 10.0,
) -> Iterator[LabelledRequest]:
    """
    Load pre-scored honeypot data for evaluation.

    Args:
        path: Path to scored CSV
        time_mode: How to handle timestamps
            - "original": Use original timestamps
            - "sequential": Assign sequential times at fixed rate
            - "compressed": Compress timeline to speed up replay
        requests_per_second: Rate for sequential mode

    Yields:
        LabelledRequest objects
    """
    import pandas as pd

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} scored entries from {path}")

    # Handle timestamps
    if time_mode == "sequential":
        interval = 1.0 / requests_per_second
        df["stream_time_s"] = df.index * interval
    elif time_mode == "compressed":
        # Compress to 1 hour total
        min_ts = df["timestamp_s"].min()
        max_ts = df["timestamp_s"].max()
        duration = max_ts - min_ts
        if duration > 0:
            df["stream_time_s"] = (df["timestamp_s"] - min_ts) / duration * 3600.0
        else:
            df["stream_time_s"] = df.index * 0.1
    else:  # original
        min_ts = df["timestamp_s"].min()
        df["stream_time_s"] = df["timestamp_s"] - min_ts

    # Sort by time
    df = df.sort_values("stream_time_s").reset_index(drop=True)

    for idx, row in df.iterrows():
        yield LabelledRequest(
            request_idx=int(idx),
            stream_time_s=float(row["stream_time_s"]),
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
    """
    Split scored data into validation and test sets.

    Validation is used for fitting theta_global.
    Test is used for final evaluation.

    Args:
        path: Path to scored CSV
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        (validation_list, test_list)
    """
    import numpy as np

    all_requests = list(load_scored_honeypot(path, time_mode="sequential"))
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


def run_evaluation_on_honeypot(
    scored_path: Path = SCORED_OUTPUT_PATH,
    output_dir: Path = Path("evaluation/results/honeypot"),
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run full evaluation on honeypot data.

    1. Split into validation/test
    2. Fit theta_global on validation
    3. Run all strategies on test
    4. Generate reports

    Args:
        scored_path: Path to pre-scored CSV
        output_dir: Output directory
        val_ratio: Validation split ratio
        seed: Random seed

    Returns:
        Evaluation summary
    """
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
            enable_anti_poisoning=False,  # Unbounded for ablation
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
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    for name, stats in summary["strategies"].items():
        status = "PASS" if stats["passes_recall_floor"] else "FAIL"
        logger.info(
            f"{name}:\n"
            f"  F1={stats['f1']:.4f}, Precision={stats['precision']:.4f}, "
            f"Recall={stats['recall']:.4f} [{status}]\n"
            f"  FPR={stats['fpr']:.4f}, FNR={stats['fnr']:.4f}\n"
            f"  Latency: mean={stats['mean_latency_us']:.0f}μs, "
            f"p99={stats['p99_latency_us']:.0f}μs"
        )

    # Generate summary report
    generate_full_report(output_dir, output_dir / "summary", recall_floor=0.95)

    return summary


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Honeypot data loader for evaluation")
    parser.add_argument("--precompute", action="store_true", help="Pre-compute s_app scores")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--input", type=Path, default=HONEYPOT_LOG_PATH, help="Input honeypot log")
    parser.add_argument("--scored", type=Path, default=SCORED_OUTPUT_PATH, help="Scored CSV path")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/honeypot"))
    parser.add_argument("--attack-threshold", type=float, default=0.4, help="Tag confidence threshold")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.precompute:
        precompute_scores(args.input, args.scored, args.attack_threshold)

    if args.evaluate:
        if not args.scored.exists():
            logger.error(f"Scored data not found: {args.scored}")
            logger.error("Run with --precompute first")
            sys.exit(1)
        run_evaluation_on_honeypot(args.scored, args.output_dir, seed=args.seed)

    if not args.precompute and not args.evaluate:
        parser.print_help()
