#!/usr/bin/env python3
"""
Delta Sweep for IEEE Paper

Runs the slow poisoning scenario with multiple delta values:
δ ∈ {0.05, 0.10, 0.15, 0.20, ∞ (unbounded)}

This provides real experimental data for the trade-off analysis.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.harness import ReplayHarness, LabelledRequest
from evaluation.strategies import ContextAdaptiveStrategy
from evaluation.strategies.base import derive_context_key
from evaluation.data_loader import fit_validation_threshold
from evaluation.summarize import generate_full_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCORED_PATH = Path("evaluation/data/payload_scored.csv")

# Delta values to test
DELTA_VALUES = [0.05, 0.10, 0.15, 0.20, None]  # None = unbounded


def load_scored_data(path: Path = SCORED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} scored entries")
    return df


def generate_poisoning_stream(
    df: pd.DataFrame,
    target_context: str = "(/api/users/{id}, POST)",
    poison_start: float = 0.2,
    poison_end: float = 0.7,
    poison_score_start: float = 0.25,
    poison_score_end: float = 0.85,
    attack_score: float = 0.88,
    n_attack_requests: int = 100,
    seed: int = 42,
):
    """Generate the poisoning request stream (same as run_paper_scenarios)."""
    rng = np.random.default_rng(seed)
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)

    poison_start_idx = int(n * poison_start)
    poison_end_idx = int(n * poison_end)
    poison_duration = poison_end_idx - poison_start_idx

    requests = []
    interval = 0.1
    poison_injected = 0
    attacks_injected = 0
    poison_interval = max(1, poison_duration // 200)

    for idx, row in df_shuffled.iterrows():
        stream_time_s = idx * interval
        s_app = float(row["s_app"])
        is_attack = bool(row["is_attack"])
        path = str(row["path"])
        method = str(row["method"])

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
                    is_attack=False,
                    path="/api/users/123",
                    method="POST",
                ))
                poison_injected += 1
                continue

        if idx >= poison_end_idx and attacks_injected < n_attack_requests:
            if idx % 50 == 0:
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

        requests.append(LabelledRequest(
            request_idx=idx,
            stream_time_s=stream_time_s,
            s_app=s_app,
            is_attack=is_attack,
            path=path,
            method=method,
        ))

    return requests, poison_end_idx, poison_injected


def run_delta_sweep(
    df: pd.DataFrame,
    theta_global: float,
    output_dir: Path,
    seed: int = 42,
):
    """Run poisoning scenario across multiple delta values."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate request stream once
    requests, poison_end_idx, poison_injected = generate_poisoning_stream(df, seed=seed)
    logger.info(f"Generated {len(requests)} requests with {poison_injected} poison samples")

    results = {}
    target_ctx = "(/api/users/{id}, POST)"

    for delta in DELTA_VALUES:
        delta_name = f"delta_{delta:.2f}" if delta else "unbounded"
        enable_bound = delta is not None

        logger.info(f"\n{'='*60}")
        logger.info(f"Running δ = {delta if delta else '∞ (unbounded)'}")
        logger.info(f"{'='*60}")

        strategy = ContextAdaptiveStrategy(
            theta_global=theta_global,
            alpha=0.02,
            margin=0.02,
            delta=delta if delta else 0.15,  # delta value doesn't matter if unbounded
            n_min=50,
            window_size=500,
            enable_anti_poisoning=enable_bound,
        )

        run_dir = output_dir / delta_name
        harness = ReplayHarness(
            strategies=[strategy],
            confirmation_delay_s=60.0,
            window_size_s=300.0,
            recall_floor=0.95,
            output_dir=run_dir,
        )

        summary = harness.run(iter(requests))
        harness.save_results()

        # Analyze poisoning effect
        metrics = harness.get_per_request_metrics(strategy.name)

        # Get all target context metrics
        target_metrics = [m for m in metrics if m.context_key == target_ctx]
        all_thresholds = [m.threshold_used for m in target_metrics]

        # Get attack metrics (post-poisoning)
        attack_metrics = [m for m in metrics
                         if m.context_key == target_ctx
                         and m.request_idx >= poison_end_idx
                         and m.ground_truth_is_attack]

        # Compute metrics
        attacks_missed = sum(1 for m in attack_metrics if m.is_fn)
        total_attacks = len(attack_metrics)
        miss_rate = attacks_missed / total_attacks if total_attacks > 0 else 0

        # Get FP at target context
        target_fp = sum(1 for m in target_metrics if m.is_fp)
        target_tn = sum(1 for m in target_metrics if m.is_tn)
        target_fpr = target_fp / (target_fp + target_tn) if (target_fp + target_tn) > 0 else 0

        # Get background FP (all other contexts)
        background_metrics = [m for m in metrics if m.context_key != target_ctx]
        background_fp = sum(1 for m in background_metrics if m.is_fp)
        background_tn = sum(1 for m in background_metrics if m.is_tn)
        background_fpr = background_fp / (background_fp + background_tn) if (background_fp + background_tn) > 0 else 0

        stats = summary["strategies"][strategy.name]

        results[delta_name] = {
            "delta": delta,
            "enable_anti_poisoning": enable_bound,
            "precision": stats["precision"],
            "recall": stats["recall"],
            "f1": stats["f1"],
            "fpr": stats["fpr"],
            "passes_recall_floor": stats["passes_recall_floor"],
            "total_attacks": total_attacks,
            "attacks_missed": attacks_missed,
            "attack_miss_rate": miss_rate,
            "target_fp": target_fp,
            "target_fpr": target_fpr,
            "background_fp": background_fp,
            "background_fpr": background_fpr,
            "mean_threshold_global": np.mean([m.threshold_used for m in metrics]),
            "mean_threshold_target": np.mean(all_thresholds) if all_thresholds else theta_global,
            "min_threshold_target": min(all_thresholds) if all_thresholds else theta_global,
            "max_threshold_target": max(all_thresholds) if all_thresholds else theta_global,
            "theta_global": theta_global,
        }

        logger.info(f"  F1={stats['f1']:.4f}, FPR={stats['fpr']:.4f}")
        logger.info(f"  Attacks missed: {attacks_missed}/{total_attacks} ({miss_rate*100:.2f}%)")
        logger.info(f"  Target FP: {target_fp}, Background FP: {background_fp}")
        logger.info(f"  Min threshold at target: {results[delta_name]['min_threshold_target']:.4f}")

    # Save results
    with open(output_dir / "delta_sweep_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "theta_global": theta_global,
            "seed": seed,
            "delta_values_tested": [d if d else "unbounded" for d in DELTA_VALUES],
            "results": results,
        }, f, indent=2)

    # Create summary table
    summary_df = pd.DataFrame([
        {
            "delta": r["delta"] if r["delta"] else float('inf'),
            "f1": r["f1"],
            "precision": r["precision"],
            "recall": r["recall"],
            "fpr": r["fpr"],
            "target_fpr": r["target_fpr"],
            "background_fpr": r["background_fpr"],
            "attacks_missed": r["attacks_missed"],
            "attack_miss_rate": r["attack_miss_rate"],
            "target_fp": r["target_fp"],
            "background_fp": r["background_fp"],
            "min_threshold": r["min_threshold_target"],
            "mean_threshold": r["mean_threshold_target"],
        }
        for name, r in results.items()
    ])
    summary_df.to_csv(output_dir / "delta_sweep_summary.csv", index=False)

    logger.info("\n" + "=" * 60)
    logger.info("DELTA SWEEP COMPLETE")
    logger.info("=" * 60)
    print(summary_df.to_string())

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run delta sweep for IEEE paper")
    parser.add_argument("--scored", type=Path, default=SCORED_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/paper/delta_sweep"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load data
    df = load_scored_data(args.scored)

    # Fit theta_global on validation split
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
    logger.info(f"Fitted theta_global = {theta_global:.4f}")

    # FIXED: Use only test split for evaluation (no validation leakage)
    df_test = df.iloc[test_indices].reset_index(drop=True)
    logger.info(f"Using TEST split only: {len(df_test)} samples (excluded {len(val_indices)} validation samples)")

    # Run sweep on test data only
    results = run_delta_sweep(df_test, theta_global, args.output_dir, args.seed)

    logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
