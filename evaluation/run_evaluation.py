#!/usr/bin/env python3
"""
Main Evaluation Runner

Runs all scenarios against all strategies in lockstep.
Produces CSV outputs for IEEE paper figures.

Usage:
    python -m evaluation.run_evaluation --output-dir results/exp1
    python -m evaluation.run_evaluation --scenario stationary --duration 1800
    python -m evaluation.run_evaluation --help

All figures come from this harness. No imported numbers.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from .strategies import (
    DecisionStrategy,
    FixedThresholdStrategy,
    ValidationOptimisedStrategy,
    ContextAdaptiveStrategy,
)
from .harness import ReplayHarness, LabelledRequest
from .scenarios import (
    StationaryScenario,
    NewEndpointScenario,
    PayloadDistributionShiftScenario,
    VolumeChangeScenario,
    SlowPoisoningScenario,
)
from .scenarios.stationary import StationaryConfig
from .scenarios.traffic_shift import (
    NewEndpointConfig,
    PayloadDistributionShiftConfig,
    VolumeChangeConfig,
)
from .scenarios.slow_poisoning import SlowPoisoningConfig, analyze_poisoning_results
from .summarize import generate_full_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_strategies(
    theta_fixed: float = 0.5,
    theta_global: float = 0.35,
    alpha: float = 0.02,
    margin: float = 0.02,
    delta: float = 0.15,
    n_min: int = 50,
    window_size: int = 500,
    include_unbounded: bool = True,
) -> List[DecisionStrategy]:
    """
    Create all three strategies (plus unbounded ablation).

    Args:
        theta_fixed: Fixed threshold baseline
        theta_global: Validation-optimised global threshold
        alpha: Target per-context FPR for adaptive
        margin: Safety margin for adaptive
        delta: Anti-poisoning bound for adaptive
        n_min: Cold start minimum samples
        window_size: Benign window size
        include_unbounded: Include unbounded adaptive for ablation

    Returns:
        List of strategy instances
    """
    strategies = [
        FixedThresholdStrategy(theta=theta_fixed),
        ValidationOptimisedStrategy(theta_global=theta_global),
        ContextAdaptiveStrategy(
            theta_global=theta_global,
            alpha=alpha,
            margin=margin,
            delta=delta,
            n_min=n_min,
            window_size=window_size,
            enable_anti_poisoning=True,
        ),
    ]

    if include_unbounded:
        strategies.append(
            ContextAdaptiveStrategy(
                theta_global=theta_global,
                alpha=alpha,
                margin=margin,
                delta=delta,
                n_min=n_min,
                window_size=window_size,
                enable_anti_poisoning=False,  # Unbounded for ablation
            )
        )

    return strategies


def run_stationary_scenario(
    strategies: List[DecisionStrategy],
    output_dir: Path,
    duration_s: float = 3600.0,
    requests_per_second: float = 10.0,
    attack_rate: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run stationary baseline scenario."""
    logger.info("=" * 60)
    logger.info("SCENARIO: Stationary (Baseline)")
    logger.info("=" * 60)

    config = StationaryConfig(
        duration_s=duration_s,
        requests_per_second=requests_per_second,
        attack_rate=attack_rate,
        seed=seed,
    )
    scenario = StationaryScenario(config)

    harness = ReplayHarness(
        strategies=strategies,
        confirmation_delay_s=60.0,
        window_size_s=300.0,
        recall_floor=0.95,
        output_dir=output_dir / "stationary",
    )

    summary = harness.run(scenario.generate())
    harness.save_results()

    logger.info("\nStationary scenario results:")
    for name, stats in summary["strategies"].items():
        status = "PASS" if stats["passes_recall_floor"] else "FAIL"
        logger.info(
            f"  {name}: F1={stats['f1']:.3f}, Recall={stats['recall']:.3f} [{status}], "
            f"FPR={stats['fpr']:.3f}, p99_latency={stats['p99_latency_us']:.0f}μs"
        )

    return summary


def run_new_endpoint_scenario(
    strategies: List[DecisionStrategy],
    output_dir: Path,
    duration_s: float = 3600.0,
    appear_at: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run new endpoint appearing mid-stream scenario."""
    logger.info("=" * 60)
    logger.info("SCENARIO: New Endpoint Mid-Stream")
    logger.info("=" * 60)

    config = NewEndpointConfig(
        duration_s=duration_s,
        new_endpoint_appear_at=appear_at,
        new_endpoint_path="/api/v2/recommendations",
        new_endpoint_traffic_fraction=0.3,
        new_endpoint_benign_mean=0.20,
        seed=seed,
    )
    scenario = NewEndpointScenario(config)

    harness = ReplayHarness(
        strategies=strategies,
        confirmation_delay_s=60.0,
        window_size_s=300.0,
        recall_floor=0.95,
        output_dir=output_dir / "new_endpoint",
    )

    summary = harness.run(scenario.generate())
    harness.save_results()

    logger.info("\nNew endpoint scenario results:")
    for name, stats in summary["strategies"].items():
        status = "PASS" if stats["passes_recall_floor"] else "FAIL"
        logger.info(f"  {name}: F1={stats['f1']:.3f}, Recall={stats['recall']:.3f} [{status}]")

    return summary


def run_payload_shift_scenario(
    strategies: List[DecisionStrategy],
    output_dir: Path,
    duration_s: float = 3600.0,
    shift_start_at: float = 0.4,
    shifted_mean: float = 0.30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run payload distribution shift scenario."""
    logger.info("=" * 60)
    logger.info("SCENARIO: Payload Distribution Shift")
    logger.info("=" * 60)

    config = PayloadDistributionShiftConfig(
        duration_s=duration_s,
        shift_start_at=shift_start_at,
        transition_duration=0.1,
        target_endpoint="/api/search",
        shifted_benign_mean=shifted_mean,
        shifted_benign_std=0.12,
        seed=seed,
    )
    scenario = PayloadDistributionShiftScenario(config)

    harness = ReplayHarness(
        strategies=strategies,
        confirmation_delay_s=60.0,
        window_size_s=300.0,
        recall_floor=0.95,
        output_dir=output_dir / "payload_shift",
    )

    summary = harness.run(scenario.generate())
    harness.save_results()

    logger.info("\nPayload shift scenario results:")
    for name, stats in summary["strategies"].items():
        status = "PASS" if stats["passes_recall_floor"] else "FAIL"
        logger.info(f"  {name}: F1={stats['f1']:.3f}, Recall={stats['recall']:.3f} [{status}]")

    return summary


def run_volume_change_scenario(
    strategies: List[DecisionStrategy],
    output_dir: Path,
    duration_s: float = 3600.0,
    volume_multiplier: float = 3.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run traffic volume change scenario."""
    logger.info("=" * 60)
    logger.info("SCENARIO: Volume Change (3x spike)")
    logger.info("=" * 60)

    config = VolumeChangeConfig(
        duration_s=duration_s,
        volume_change_at=0.5,
        volume_multiplier=volume_multiplier,
        gradual_transition=True,
        seed=seed,
    )
    scenario = VolumeChangeScenario(config)

    harness = ReplayHarness(
        strategies=strategies,
        confirmation_delay_s=60.0,
        window_size_s=300.0,
        recall_floor=0.95,
        output_dir=output_dir / "volume_change",
    )

    summary = harness.run(scenario.generate())
    harness.save_results()

    logger.info("\nVolume change scenario results:")
    for name, stats in summary["strategies"].items():
        status = "PASS" if stats["passes_recall_floor"] else "FAIL"
        logger.info(f"  {name}: F1={stats['f1']:.3f}, Recall={stats['recall']:.3f} [{status}]")

    return summary


def run_slow_poisoning_scenario(
    strategies: List[DecisionStrategy],
    output_dir: Path,
    duration_s: float = 3600.0,
    poison_score_end: float = 0.48,
    attack_score: float = 0.52,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run slow-poisoning attack scenario."""
    logger.info("=" * 60)
    logger.info("SCENARIO: Slow Poisoning Attack")
    logger.info("=" * 60)

    config = SlowPoisoningConfig(
        duration_s=duration_s,
        target_endpoint="/api/users/{id}",
        poisoning_start_at=0.2,
        poisoning_end_at=0.7,
        poison_score_start=0.25,
        poison_score_end=poison_score_end,
        attack_score=attack_score,
        n_attack_requests=20,
        seed=seed,
    )
    scenario = SlowPoisoningScenario(config)

    harness = ReplayHarness(
        strategies=strategies,
        confirmation_delay_s=60.0,
        window_size_s=300.0,
        recall_floor=0.95,
        output_dir=output_dir / "slow_poisoning",
    )

    summary = harness.run(scenario.generate())
    harness.save_results()

    # Analyze poisoning effectiveness
    poisoning_analyses = []
    for strategy in strategies:
        if "Adaptive" in strategy.name:
            metrics = harness.get_per_request_metrics(strategy.name)
            analysis = analyze_poisoning_results(metrics, config, strategy.get_config())
            poisoning_analyses.append({
                "strategy_name": strategy.name,
                **analysis.to_dict(),
            })

    # Save poisoning analysis
    analysis_path = output_dir / "slow_poisoning" / "poisoning_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(poisoning_analyses, f, indent=2)

    logger.info("\nSlow poisoning scenario results:")
    for name, stats in summary["strategies"].items():
        status = "PASS" if stats["passes_recall_floor"] else "FAIL"
        logger.info(f"  {name}: F1={stats['f1']:.3f}, Recall={stats['recall']:.3f} [{status}]")

    logger.info("\nPoisoning analysis:")
    for analysis in poisoning_analyses:
        logger.info(
            f"  {analysis['strategy_name']}: "
            f"drift={analysis['max_threshold_drift']:.4f}, "
            f"bounded={analysis['drift_bounded_by_delta']}, "
            f"attacks_missed={analysis['n_attacks_missed']}/{analysis['n_attack_requests']}"
        )

    summary["poisoning_analyses"] = poisoning_analyses
    return summary


def run_all_scenarios(
    output_dir: Path,
    theta_global: float = 0.35,
    delta: float = 0.15,
    duration_s: float = 3600.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run all scenarios and generate full report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create strategies
    strategies = create_strategies(
        theta_fixed=0.5,
        theta_global=theta_global,
        delta=delta,
        include_unbounded=True,
    )

    logger.info(f"Running evaluation with {len(strategies)} strategies")
    for s in strategies:
        logger.info(f"  - {s.name}")

    all_summaries = {}

    # Run each scenario
    all_summaries["stationary"] = run_stationary_scenario(
        strategies, output_dir, duration_s=duration_s, seed=seed
    )

    # Reset strategies between scenarios
    for s in strategies:
        s.reset()

    all_summaries["new_endpoint"] = run_new_endpoint_scenario(
        strategies, output_dir, duration_s=duration_s, seed=seed
    )

    for s in strategies:
        s.reset()

    all_summaries["payload_shift"] = run_payload_shift_scenario(
        strategies, output_dir, duration_s=duration_s, seed=seed
    )

    for s in strategies:
        s.reset()

    all_summaries["volume_change"] = run_volume_change_scenario(
        strategies, output_dir, duration_s=duration_s, seed=seed
    )

    for s in strategies:
        s.reset()

    all_summaries["slow_poisoning"] = run_slow_poisoning_scenario(
        strategies, output_dir, duration_s=duration_s, seed=seed
    )

    # Save master summary
    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "theta_global": theta_global,
            "delta": delta,
            "duration_s": duration_s,
            "seed": seed,
            "strategies": [s.get_config() for s in strategies],
        },
        "scenarios": all_summaries,
    }

    summary_path = output_dir / "master_summary.json"
    with open(summary_path, "w") as f:
        json.dump(master_summary, f, indent=2)

    logger.info(f"\nMaster summary saved to {summary_path}")

    # Generate summary reports for each scenario
    for scenario_name in all_summaries:
        scenario_dir = output_dir / scenario_name
        if scenario_dir.exists():
            generate_full_report(scenario_dir, scenario_dir / "summary")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    return master_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run threshold strategy evaluation for IEEE paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all scenarios (default)
    python -m evaluation.run_evaluation

    # Run specific scenario
    python -m evaluation.run_evaluation --scenario stationary

    # Custom parameters
    python -m evaluation.run_evaluation --theta-global 0.4 --delta 0.10

    # Quick test run
    python -m evaluation.run_evaluation --duration 300 --seed 123
        """,
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/results"),
        help="Output directory for results (default: evaluation/results)"
    )
    parser.add_argument(
        "--scenario", type=str, default="all",
        choices=["all", "stationary", "new_endpoint", "payload_shift", "volume_change", "slow_poisoning"],
        help="Which scenario to run (default: all)"
    )
    parser.add_argument(
        "--theta-global", type=float, default=0.35,
        help="Validation-optimised global threshold (default: 0.35)"
    )
    parser.add_argument(
        "--delta", type=float, default=0.15,
        help="Anti-poisoning bound (default: 0.15)"
    )
    parser.add_argument(
        "--duration", type=float, default=3600.0,
        help="Scenario duration in seconds (default: 3600)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-unbounded", action="store_true",
        help="Skip unbounded adaptive strategy ablation"
    )

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp

    if args.scenario == "all":
        run_all_scenarios(
            output_dir=output_dir,
            theta_global=args.theta_global,
            delta=args.delta,
            duration_s=args.duration,
            seed=args.seed,
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        strategies = create_strategies(
            theta_global=args.theta_global,
            delta=args.delta,
            include_unbounded=not args.no_unbounded,
        )

        scenario_runners = {
            "stationary": run_stationary_scenario,
            "new_endpoint": run_new_endpoint_scenario,
            "payload_shift": run_payload_shift_scenario,
            "volume_change": run_volume_change_scenario,
            "slow_poisoning": run_slow_poisoning_scenario,
        }

        runner = scenario_runners[args.scenario]
        runner(strategies, output_dir, duration_s=args.duration, seed=args.seed)

        # Generate summary
        scenario_dir = output_dir / args.scenario
        if scenario_dir.exists():
            generate_full_report(scenario_dir, scenario_dir / "summary")


if __name__ == "__main__":
    main()
