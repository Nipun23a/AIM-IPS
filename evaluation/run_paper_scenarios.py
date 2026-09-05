#!/usr/bin/env python3
"""
Run IEEE Paper Scenarios on Real Scored Dataset

Scenarios:
1. Traffic Shift - payload distribution change mid-stream
2. Slow Poisoning - with and without anti-poisoning bound

Uses payload_scored.csv with real LightGBM scores.
"""

import json
import logging
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.harness import ReplayHarness, LabelledRequest
from evaluation.strategies import (
    FixedThresholdStrategy,
    ValidationOptimisedStrategy,
    ContextAdaptiveStrategy,
)
from evaluation.strategies.base import derive_context_key
from evaluation.data_loader import fit_validation_threshold
from evaluation.summarize import generate_full_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCORED_PATH = Path("evaluation/data/payload_scored.csv")


def load_scored_data(path: Path = SCORED_PATH) -> pd.DataFrame:
    """Load the pre-scored dataset."""
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} scored entries")
    return df


def create_strategies(theta_global: float, include_unbounded: bool = True) -> List:
    """Create all evaluation strategies."""
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
    ]
    if include_unbounded:
        strategies.append(
            ContextAdaptiveStrategy(
                theta_global=theta_global,
                alpha=0.02,
                margin=0.02,
                delta=0.15,
                n_min=50,
                window_size=500,
                enable_anti_poisoning=False,
            )
        )
    return strategies


# =============================================================================
# SCENARIO 1: TRAFFIC SHIFT
# =============================================================================

def run_traffic_shift_scenario(
    df: pd.DataFrame,
    theta_global: float,
    output_dir: Path,
    shift_point: float = 0.5,
    shift_magnitude: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Traffic Shift Scenario

    Simulates a payload distribution shift mid-stream:
    - Phase 1 (0 to shift_point): Normal distribution
    - Phase 2 (shift_point to end): Benign scores shifted UP by shift_magnitude

    This simulates:
    - New client library with different encoding
    - API version migration
    - Legitimate traffic pattern change

    Expected outcome:
    - Fixed/ValidationOptimised: FPR increases after shift (can't adapt)
    - ContextAdaptive: FPR recovers as window updates with new distribution
    """
    logger.info("=" * 70)
    logger.info("SCENARIO: Traffic Shift (Payload Distribution Change)")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Shuffle and prepare data
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    shift_idx = int(n * shift_point)

    logger.info(f"Total samples: {n}")
    logger.info(f"Shift at sample: {shift_idx} ({shift_point*100:.0f}%)")
    logger.info(f"Shift magnitude: +{shift_magnitude} to benign scores")

    # Generate request stream with shift
    requests = []
    interval = 0.1  # 10 req/sec

    for idx, row in df_shuffled.iterrows():
        stream_time_s = idx * interval
        s_app = float(row["s_app"])
        is_attack = bool(row["is_attack"])

        # Apply shift to benign traffic after shift_point
        if idx >= shift_idx and not is_attack:
            # Shift benign scores UP (simulating encoding change)
            s_app = min(1.0, s_app + shift_magnitude + rng.normal(0, 0.02))

        requests.append(LabelledRequest(
            request_idx=idx,
            stream_time_s=stream_time_s,
            s_app=s_app,
            is_attack=is_attack,
            path=str(row["path"]),
            method=str(row["method"]),
        ))

    # Create strategies
    strategies = create_strategies(theta_global, include_unbounded=True)

    # Run harness
    harness = ReplayHarness(
        strategies=strategies,
        confirmation_delay_s=60.0,
        window_size_s=300.0,
        recall_floor=0.95,
        output_dir=output_dir,
    )

    summary = harness.run(iter(requests))
    harness.save_results()

    # Analyze pre/post shift performance
    shift_analysis = analyze_shift_performance(harness, strategies, shift_idx, n)

    # Save analysis
    with open(output_dir / "shift_analysis.json", "w") as f:
        json.dump({
            "config": {
                "shift_point": shift_point,
                "shift_magnitude": shift_magnitude,
                "shift_idx": shift_idx,
                "n_samples": n,
            },
            "summary": {name: {k: v for k, v in stats.items() if k != "config"}
                       for name, stats in summary["strategies"].items()},
            "shift_analysis": shift_analysis,
        }, f, indent=2)

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("TRAFFIC SHIFT RESULTS")
    logger.info("=" * 70)

    for name, stats in summary["strategies"].items():
        status = "PASS" if stats["passes_recall_floor"] else "FAIL"
        logger.info(f"\n{name}:")
        logger.info(f"  Overall: F1={stats['f1']:.4f}, Recall={stats['recall']:.4f} [{status}], FPR={stats['fpr']:.4f}")

        if name in shift_analysis:
            pre = shift_analysis[name]["pre_shift"]
            post = shift_analysis[name]["post_shift"]
            logger.info(f"  Pre-shift:  FPR={pre['fpr']:.4f}, Precision={pre['precision']:.4f}")
            logger.info(f"  Post-shift: FPR={post['fpr']:.4f}, Precision={post['precision']:.4f}")
            logger.info(f"  FPR change: {post['fpr'] - pre['fpr']:+.4f}")

    # Generate report
    generate_full_report(output_dir, output_dir / "summary", recall_floor=0.95)

    return {"summary": summary, "shift_analysis": shift_analysis}


def analyze_shift_performance(harness, strategies, shift_idx, n_total) -> Dict:
    """Analyze performance before and after shift point."""
    analysis = {}

    for strategy in strategies:
        metrics = harness.get_per_request_metrics(strategy.name)

        pre_metrics = [m for m in metrics if m.request_idx < shift_idx]
        post_metrics = [m for m in metrics if m.request_idx >= shift_idx]

        def compute_rates(mlist):
            if not mlist:
                return {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "precision": 0, "recall": 0, "fpr": 0}
            tp = sum(1 for m in mlist if m.is_tp)
            fp = sum(1 for m in mlist if m.is_fp)
            tn = sum(1 for m in mlist if m.is_tn)
            fn = sum(1 for m in mlist if m.is_fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "precision": precision, "recall": recall, "fpr": fpr,
                    "n_samples": len(mlist)}

        analysis[strategy.name] = {
            "pre_shift": compute_rates(pre_metrics),
            "post_shift": compute_rates(post_metrics),
        }

    return analysis


# =============================================================================
# SCENARIO 2: SLOW POISONING
# =============================================================================

def run_slow_poisoning_scenario(
    df: pd.DataFrame,
    theta_global: float,
    output_dir: Path,
    target_context: str = "(/api/users/{id}, POST)",
    poison_start: float = 0.2,
    poison_end: float = 0.7,
    poison_score_start: float = 0.25,
    poison_score_end: float = 0.85,
    attack_score: float = 0.88,
    n_attack_requests: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Slow Poisoning Scenario

    Adversarial attack that gradually drifts the adaptive threshold:
    - Phase 1 (0 to poison_start): Normal traffic, establish baseline
    - Phase 2 (poison_start to poison_end): Inject gradually increasing
      "benign" scores to target endpoint
    - Phase 3 (poison_end to end): Launch attack with scores that would
      have been caught before poisoning

    Expected outcome:
    - Bounded (δ=0.15): Threshold capped, attacks detected
    - Unbounded: Threshold drifts high, attacks missed
    """
    logger.info("=" * 70)
    logger.info("SCENARIO: Slow Poisoning Attack")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Shuffle data
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)

    poison_start_idx = int(n * poison_start)
    poison_end_idx = int(n * poison_end)
    poison_duration = poison_end_idx - poison_start_idx

    logger.info(f"Total samples: {n}")
    logger.info(f"Target context: {target_context}")
    logger.info(f"Poisoning: samples {poison_start_idx} to {poison_end_idx}")
    logger.info(f"Poison scores: {poison_score_start} → {poison_score_end}")
    logger.info(f"Attack score: {attack_score}")
    logger.info(f"Attack requests: {n_attack_requests}")

    # Generate request stream with poisoning
    requests = []
    interval = 0.1
    poison_injected = 0
    attacks_injected = 0
    poison_interval = max(1, poison_duration // 200)  # ~200 poison requests

    for idx, row in df_shuffled.iterrows():
        stream_time_s = idx * interval
        s_app = float(row["s_app"])
        is_attack = bool(row["is_attack"])
        path = str(row["path"])
        method = str(row["method"])
        context_key = derive_context_key(path, method)

        # Phase 2: Inject poison to target context
        if poison_start_idx <= idx < poison_end_idx:
            if idx % poison_interval == 0:
                progress = (idx - poison_start_idx) / poison_duration
                poison_score = poison_score_start + progress * (poison_score_end - poison_score_start)
                poison_score += rng.normal(0, 0.02)
                poison_score = np.clip(poison_score, 0, 0.99)

                requests.append(LabelledRequest(
                    request_idx=idx,
                    stream_time_s=stream_time_s,
                    s_app=poison_score,
                    is_attack=False,  # Labelled benign!
                    path="/api/users/123",  # Target endpoint
                    method="POST",
                ))
                poison_injected += 1
                continue

        # Phase 3: Launch attacks to target context
        if idx >= poison_end_idx and attacks_injected < n_attack_requests:
            if idx % 50 == 0:  # Spread attacks
                attack_s = attack_score + rng.normal(0, 0.02)
                requests.append(LabelledRequest(
                    request_idx=idx,
                    stream_time_s=stream_time_s,
                    s_app=np.clip(attack_s, 0, 1),
                    is_attack=True,
                    path="/api/users/123",
                    method="POST",
                ))
                attacks_injected += 1
                continue

        # Normal traffic
        requests.append(LabelledRequest(
            request_idx=idx,
            stream_time_s=stream_time_s,
            s_app=s_app,
            is_attack=is_attack,
            path=path,
            method=method,
        ))

    logger.info(f"Poison requests injected: {poison_injected}")
    logger.info(f"Attack requests injected: {attacks_injected}")

    # Run with BOUNDED adaptive
    logger.info("\n--- Running with BOUNDED (δ=0.15) ---")
    strategies_bounded = [
        FixedThresholdStrategy(theta=0.5),
        ValidationOptimisedStrategy(theta_global=theta_global),
        ContextAdaptiveStrategy(
            theta_global=theta_global,
            alpha=0.02, margin=0.02, delta=0.15,
            n_min=50, window_size=500,
            enable_anti_poisoning=True,
        ),
    ]

    harness_bounded = ReplayHarness(
        strategies=strategies_bounded,
        confirmation_delay_s=60.0,
        window_size_s=300.0,
        recall_floor=0.95,
        output_dir=output_dir / "bounded",
    )

    summary_bounded = harness_bounded.run(iter(requests))
    harness_bounded.save_results()

    # Analyze poisoning effect for bounded
    poison_analysis_bounded = analyze_poisoning(
        harness_bounded, strategies_bounded,
        target_context, poison_end_idx, theta_global
    )

    # Reset strategies and run UNBOUNDED
    logger.info("\n--- Running with UNBOUNDED ---")
    strategies_unbounded = [
        FixedThresholdStrategy(theta=0.5),
        ValidationOptimisedStrategy(theta_global=theta_global),
        ContextAdaptiveStrategy(
            theta_global=theta_global,
            alpha=0.02, margin=0.02, delta=0.15,
            n_min=50, window_size=500,
            enable_anti_poisoning=False,  # UNBOUNDED
        ),
    ]

    harness_unbounded = ReplayHarness(
        strategies=strategies_unbounded,
        confirmation_delay_s=60.0,
        window_size_s=300.0,
        recall_floor=0.95,
        output_dir=output_dir / "unbounded",
    )

    summary_unbounded = harness_unbounded.run(iter(requests))
    harness_unbounded.save_results()

    # Analyze poisoning effect for unbounded
    poison_analysis_unbounded = analyze_poisoning(
        harness_unbounded, strategies_unbounded,
        target_context, poison_end_idx, theta_global
    )

    # Save combined analysis
    combined = {
        "config": {
            "target_context": target_context,
            "poison_start": poison_start,
            "poison_end": poison_end,
            "poison_score_range": [poison_score_start, poison_score_end],
            "attack_score": attack_score,
            "n_attack_requests": n_attack_requests,
            "poison_injected": poison_injected,
            "theta_global": theta_global,
        },
        "bounded": {
            "summary": {k: {kk: vv for kk, vv in v.items() if kk != "config"}
                       for k, v in summary_bounded["strategies"].items()},
            "poisoning_analysis": poison_analysis_bounded,
        },
        "unbounded": {
            "summary": {k: {kk: vv for kk, vv in v.items() if kk != "config"}
                       for k, v in summary_unbounded["strategies"].items()},
            "poisoning_analysis": poison_analysis_unbounded,
        },
    }

    with open(output_dir / "poisoning_analysis.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("SLOW POISONING RESULTS")
    logger.info("=" * 70)

    logger.info("\n--- BOUNDED (δ=0.15) ---")
    for name, stats in summary_bounded["strategies"].items():
        status = "PASS" if stats["passes_recall_floor"] else "FAIL"
        logger.info(f"{name}: F1={stats['f1']:.4f}, Recall={stats['recall']:.4f} [{status}], FPR={stats['fpr']:.4f}")

    if "ContextAdaptive" in str(poison_analysis_bounded):
        for name, pa in poison_analysis_bounded.items():
            if "Adaptive" in name:
                logger.info(f"  {name}: threshold_drift={pa['threshold_drift']:.4f}, "
                           f"attacks_missed={pa['attacks_missed']}/{pa['total_attacks']}")

    logger.info("\n--- UNBOUNDED ---")
    for name, stats in summary_unbounded["strategies"].items():
        status = "PASS" if stats["passes_recall_floor"] else "FAIL"
        logger.info(f"{name}: F1={stats['f1']:.4f}, Recall={stats['recall']:.4f} [{status}], FPR={stats['fpr']:.4f}")

    if "ContextAdaptive" in str(poison_analysis_unbounded):
        for name, pa in poison_analysis_unbounded.items():
            if "Adaptive" in name:
                logger.info(f"  {name}: threshold_drift={pa['threshold_drift']:.4f}, "
                           f"attacks_missed={pa['attacks_missed']}/{pa['total_attacks']}")

    # Generate reports
    generate_full_report(output_dir / "bounded", output_dir / "bounded" / "summary")
    generate_full_report(output_dir / "unbounded", output_dir / "unbounded" / "summary")

    return combined


def analyze_poisoning(harness, strategies, target_context, poison_end_idx, theta_global) -> Dict:
    """Analyze poisoning effectiveness."""
    analysis = {}
    target_ctx_normalized = "(/api/users/{id}, POST)"

    for strategy in strategies:
        metrics = harness.get_per_request_metrics(strategy.name)

        # Find metrics for target context after poisoning
        target_metrics = [m for m in metrics
                         if m.context_key == target_ctx_normalized
                         and m.request_idx >= poison_end_idx]

        attack_metrics = [m for m in target_metrics if m.ground_truth_is_attack]

        if attack_metrics:
            attacks_missed = sum(1 for m in attack_metrics if m.is_fn)
            total_attacks = len(attack_metrics)

            # Get threshold used for attacks
            thresholds = [m.threshold_used for m in attack_metrics]
            mean_threshold = np.mean(thresholds) if thresholds else theta_global
            threshold_drift = mean_threshold - theta_global
        else:
            attacks_missed = 0
            total_attacks = 0
            threshold_drift = 0.0
            mean_threshold = theta_global

        analysis[strategy.name] = {
            "total_attacks": total_attacks,
            "attacks_missed": attacks_missed,
            "attack_miss_rate": attacks_missed / total_attacks if total_attacks > 0 else 0,
            "mean_threshold": mean_threshold,
            "threshold_drift": threshold_drift,
            "theta_global": theta_global,
        }

    return analysis


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run IEEE paper scenarios")
    parser.add_argument("--scenario", type=str, choices=["shift", "poison", "all"], default="all")
    parser.add_argument("--scored", type=Path, default=SCORED_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/paper"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load data
    df = load_scored_data(args.scored)

    # Fit theta_global on 20% validation split
    logger.info("Fitting theta_global on validation split...")
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(df))
    val_end = int(len(df) * 0.2)
    val_indices = indices[:val_end]
    test_indices = indices[val_end:]  # FIXED: separate test indices

    val_requests = [
        LabelledRequest(
            request_idx=i,
            stream_time_s=float(i),
            s_app=float(df.iloc[i]["s_app"]),
            is_attack=bool(df.iloc[i]["is_attack"]),
            path=str(df.iloc[i]["path"]),
            method=str(df.iloc[i]["method"]),
        )
        for i in val_indices
    ]

    theta_global, fit_info = fit_validation_threshold(val_requests, metric="f1")
    logger.info(f"Fitted theta_global = {theta_global:.4f} (F1 = {fit_info['best_score']:.4f})")

    # FIXED: Use only test split for scenarios (no validation leakage)
    df_test = df.iloc[test_indices].reset_index(drop=True)
    logger.info(f"Using TEST split only: {len(df_test)} samples (excluded {len(val_indices)} validation samples)")

    results = {}

    if args.scenario in ["shift", "all"]:
        results["traffic_shift"] = run_traffic_shift_scenario(
            df_test, theta_global,  # FIXED: use df_test instead of df
            args.output_dir / "traffic_shift",
            shift_point=0.5,
            shift_magnitude=0.15,
            seed=args.seed,
        )

    if args.scenario in ["poison", "all"]:
        results["slow_poisoning"] = run_slow_poisoning_scenario(
            df_test, theta_global,  # FIXED: use df_test instead of df
            args.output_dir / "slow_poisoning",
            poison_start=0.2,
            poison_end=0.7,
            poison_score_start=0.25,
            poison_score_end=0.85,
            attack_score=0.88,
            n_attack_requests=100,
            seed=args.seed,
        )

    # Save master results
    with open(args.output_dir / "master_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "theta_global": theta_global,
            "seed": args.seed,
        }, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("ALL SCENARIOS COMPLETE")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
