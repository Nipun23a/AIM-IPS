"""
Summary Script for Threshold Strategy Evaluation

Produces:
1. Comparison tables (per strategy, per scenario)
2. Threshold stability plots over time
3. Recall floor compliance report
4. Latency distribution analysis

All figures come from the harness - no imported numbers.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyScenarioResult:
    """Results for one strategy on one scenario."""
    strategy_name: str
    scenario_name: str

    # Overall metrics
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float

    # Recall floor compliance
    recall_floor: float
    passes_recall_floor: bool

    # Latency (microseconds)
    mean_latency_us: float
    p50_latency_us: float
    p99_latency_us: float
    max_latency_us: float

    # Request counts
    n_requests: int
    tp: int
    fp: int
    tn: int
    fn: int

    # Threshold stats (for adaptive)
    mean_threshold: float
    threshold_std: float
    n_contexts: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "scenario": self.scenario_name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "fnr": round(self.fnr, 4),
            "recall_floor": self.recall_floor,
            "passes_recall_floor": self.passes_recall_floor,
            "mean_latency_us": round(self.mean_latency_us, 2),
            "p50_latency_us": round(self.p50_latency_us, 2),
            "p99_latency_us": round(self.p99_latency_us, 2),
            "max_latency_us": round(self.max_latency_us, 2),
            "n_requests": self.n_requests,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "mean_threshold": round(self.mean_threshold, 4),
            "threshold_std": round(self.threshold_std, 4),
            "n_contexts": self.n_contexts,
        }


def load_per_window_metrics(csv_path: Path) -> List[Dict[str, Any]]:
    """Load per-window metrics from CSV."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key not in ["strategy"]:
                    try:
                        if "." in str(row[key]):
                            row[key] = float(row[key])
                        else:
                            row[key] = int(row[key])
                    except (ValueError, TypeError):
                        pass
            rows.append(row)
    return rows


def load_per_request_metrics(csv_path: Path) -> List[Dict[str, Any]]:
    """Load per-request metrics from CSV."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                if key not in ["context_key", "decision"]:
                    try:
                        if "." in str(row[key]):
                            row[key] = float(row[key])
                        else:
                            row[key] = int(row[key])
                    except (ValueError, TypeError):
                        pass
            rows.append(row)
    return rows


def compute_strategy_scenario_result(
    per_request_rows: List[Dict[str, Any]],
    strategy_name: str,
    scenario_name: str,
    recall_floor: float = 0.95,
) -> StrategyScenarioResult:
    """Compute summary statistics for one strategy on one scenario."""
    if not per_request_rows:
        raise ValueError("No data provided")

    tp = sum(r["is_tp"] for r in per_request_rows)
    fp = sum(r["is_fp"] for r in per_request_rows)
    tn = sum(r["is_tn"] for r in per_request_rows)
    fn = sum(r["is_fn"] for r in per_request_rows)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    latencies = [r["decision_latency_us"] for r in per_request_rows]
    latencies_sorted = sorted(latencies)

    thresholds = [r["threshold_used"] for r in per_request_rows]
    contexts = set(r["context_key"] for r in per_request_rows)

    return StrategyScenarioResult(
        strategy_name=strategy_name,
        scenario_name=scenario_name,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        fnr=fnr,
        recall_floor=recall_floor,
        passes_recall_floor=recall >= recall_floor,
        mean_latency_us=float(np.mean(latencies)),
        p50_latency_us=float(latencies_sorted[len(latencies_sorted) // 2]),
        p99_latency_us=float(latencies_sorted[int(len(latencies_sorted) * 0.99)]),
        max_latency_us=float(max(latencies)),
        n_requests=len(per_request_rows),
        tp=tp, fp=fp, tn=tn, fn=fn,
        mean_threshold=float(np.mean(thresholds)),
        threshold_std=float(np.std(thresholds)),
        n_contexts=len(contexts),
    )


def generate_comparison_table(
    results: List[StrategyScenarioResult],
    output_path: Path,
) -> None:
    """Generate comparison table CSV."""
    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "scenario", "strategy", "precision", "recall", "f1", "fpr", "fnr",
            "passes_recall_floor", "mean_latency_us", "p99_latency_us",
            "mean_threshold", "threshold_std", "n_contexts",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                "scenario": r.scenario_name,
                "strategy": r.strategy_name,
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
                "f1": round(r.f1, 4),
                "fpr": round(r.fpr, 4),
                "fnr": round(r.fnr, 4),
                "passes_recall_floor": r.passes_recall_floor,
                "mean_latency_us": round(r.mean_latency_us, 2),
                "p99_latency_us": round(r.p99_latency_us, 2),
                "mean_threshold": round(r.mean_threshold, 4),
                "threshold_std": round(r.threshold_std, 4),
                "n_contexts": r.n_contexts,
            })

    logger.info(f"Comparison table saved to {output_path}")


def generate_latex_table(
    results: List[StrategyScenarioResult],
    output_path: Path,
    caption: str = "Strategy comparison across scenarios",
) -> None:
    """Generate LaTeX table for IEEE paper."""
    # Group by scenario
    scenarios = sorted(set(r.scenario_name for r in results))
    strategies = sorted(set(r.strategy_name for r in results))

    with open(output_path, "w") as f:
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write("\\label{tab:strategy_comparison}\n")
        f.write("\\begin{tabular}{ll" + "c" * 6 + "}\n")
        f.write("\\toprule\n")
        f.write("Scenario & Strategy & Precision & Recall & F1 & FPR & $\\tau$-Pass & Latency ($\\mu$s) \\\\\n")
        f.write("\\midrule\n")

        for scenario in scenarios:
            scenario_results = [r for r in results if r.scenario_name == scenario]
            first = True
            for r in scenario_results:
                scenario_col = scenario if first else ""
                first = False
                tau_pass = "\\checkmark" if r.passes_recall_floor else "\\texttimes"
                f.write(
                    f"{scenario_col} & {r.strategy_name} & "
                    f"{r.precision:.3f} & {r.recall:.3f} & {r.f1:.3f} & "
                    f"{r.fpr:.3f} & {tau_pass} & {r.p99_latency_us:.0f} \\\\\n"
                )
            f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    logger.info(f"LaTeX table saved to {output_path}")


def generate_threshold_stability_plot(
    per_window_rows: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Threshold Stability Over Time",
) -> None:
    """
    Generate threshold stability plot.

    Shows threshold values over time for each strategy.
    Uses matplotlib if available, falls back to ASCII plot.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # Group by strategy
        strategies = sorted(set(r["strategy"] for r in per_window_rows))

        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy in strategies:
            strategy_rows = [r for r in per_window_rows if r["strategy"] == strategy]
            strategy_rows.sort(key=lambda x: x["window_idx"])

            times = [r["window_start_s"] / 60.0 for r in strategy_rows]  # Convert to minutes
            thresholds = [r["mean_threshold"] for r in strategy_rows]

            ax.plot(times, thresholds, label=strategy, linewidth=1.5)

            # Add min/max band for adaptive strategies
            if "Adaptive" in strategy:
                min_thresholds = [r["min_threshold"] for r in strategy_rows]
                max_thresholds = [r["max_threshold"] for r in strategy_rows]
                ax.fill_between(times, min_thresholds, max_thresholds, alpha=0.2)

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Threshold")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Set y-axis limits to show relevant range
        all_thresholds = [r["mean_threshold"] for r in per_window_rows]
        y_min = max(0, min(all_thresholds) - 0.1)
        y_max = min(1, max(all_thresholds) + 0.1)
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Threshold stability plot saved to {output_path}")

    except ImportError:
        logger.warning("matplotlib not available, generating CSV data instead")
        csv_path = output_path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["strategy", "window_idx", "time_min", "mean_threshold", "min_threshold", "max_threshold"])
            for r in per_window_rows:
                writer.writerow([
                    r["strategy"], r["window_idx"], r["window_start_s"] / 60.0,
                    r["mean_threshold"], r["min_threshold"], r["max_threshold"]
                ])
        logger.info(f"Threshold data saved to {csv_path}")


def generate_recall_fpr_tradeoff_plot(
    results: List[StrategyScenarioResult],
    output_path: Path,
    title: str = "Recall vs FPR Trade-off",
) -> None:
    """Generate recall vs FPR scatter plot."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        # Different markers for different strategies
        markers = {"Fixed": "o", "Validation": "s", "Adaptive": "^"}
        colors = {"Fixed": "blue", "Validation": "green", "Adaptive": "red"}

        for r in results:
            strategy_type = "Adaptive" if "Adaptive" in r.strategy_name else \
                           "Validation" if "Validation" in r.strategy_name else "Fixed"
            marker = markers.get(strategy_type, "x")
            color = colors.get(strategy_type, "gray")

            ax.scatter(
                r.fpr, r.recall,
                marker=marker, c=color, s=100,
                label=f"{r.strategy_name} ({r.scenario_name})",
                alpha=0.7,
            )

        # Add recall floor line
        recall_floor = results[0].recall_floor if results else 0.95
        ax.axhline(y=recall_floor, color="red", linestyle="--", alpha=0.5, label=f"Recall floor ({recall_floor})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("Recall")
        ax.set_title(title)
        ax.set_xlim(-0.01, 0.3)
        ax.set_ylim(0.8, 1.01)
        ax.grid(True, alpha=0.3)

        # Legend outside plot
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Recall-FPR plot saved to {output_path}")

    except ImportError:
        logger.warning("matplotlib not available for plotting")


def generate_latency_distribution_plot(
    per_request_rows: List[Dict[str, Any]],
    strategies: List[str],
    output_path: Path,
) -> None:
    """Generate latency distribution comparison."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        data = []
        labels = []

        for strategy in strategies:
            latencies = [r["decision_latency_us"] for r in per_request_rows
                        if strategy in str(r.get("strategy", ""))]
            if latencies:
                data.append(latencies)
                labels.append(strategy[:20])  # Truncate long names

        if data:
            ax.boxplot(data, labels=labels, vert=True)
            ax.set_ylabel("Decision Latency (μs)")
            ax.set_title("Decision Latency Distribution by Strategy")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Latency distribution plot saved to {output_path}")

    except ImportError:
        logger.warning("matplotlib not available for plotting")


def generate_poisoning_analysis_table(
    poisoning_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate poisoning analysis table."""
    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "strategy", "delta", "theta_global",
            "threshold_before", "threshold_after", "max_drift",
            "drift_bounded", "n_attacks", "n_missed", "miss_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in poisoning_results:
            writer.writerow({
                "strategy": r.get("strategy_name", ""),
                "delta": r.get("delta", 0),
                "theta_global": round(r.get("theta_global", 0), 4),
                "threshold_before": round(r.get("threshold_before_poisoning", 0), 4),
                "threshold_after": round(r.get("threshold_after_poisoning", 0), 4),
                "max_drift": round(r.get("max_threshold_drift", 0), 4),
                "drift_bounded": r.get("drift_bounded_by_delta", False),
                "n_attacks": r.get("n_attack_requests", 0),
                "n_missed": r.get("n_attacks_missed", 0),
                "miss_rate": round(r.get("attack_miss_rate", 0), 4),
            })

    logger.info(f"Poisoning analysis table saved to {output_path}")


def generate_full_report(
    results_dir: Path,
    output_dir: Path,
    recall_floor: float = 0.95,
) -> Dict[str, Path]:
    """
    Generate full summary report from harness output.

    Args:
        results_dir: Directory containing harness CSV output
        output_dir: Directory for summary output
        recall_floor: Recall floor for compliance checking

    Returns:
        Dict mapping report type to file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Load per-window metrics
    per_window_path = results_dir / "per_window_metrics.csv"
    if per_window_path.exists():
        per_window_rows = load_per_window_metrics(per_window_path)

        # Generate threshold stability plot
        plot_path = output_dir / "threshold_stability.png"
        generate_threshold_stability_plot(per_window_rows, plot_path)
        paths["threshold_stability_plot"] = plot_path

    # Find all per-request CSVs and compute results
    all_results = []
    per_request_files = list(results_dir.glob("per_request_*.csv"))

    for csv_file in per_request_files:
        rows = load_per_request_metrics(csv_file)
        if rows:
            # Extract strategy name from filename
            strategy_name = csv_file.stem.replace("per_request_", "")
            scenario_name = results_dir.name  # Use directory name as scenario

            result = compute_strategy_scenario_result(
                rows, strategy_name, scenario_name, recall_floor
            )
            all_results.append(result)

    if all_results:
        # Generate comparison table
        table_path = output_dir / "comparison_table.csv"
        generate_comparison_table(all_results, table_path)
        paths["comparison_table"] = table_path

        # Generate LaTeX table
        latex_path = output_dir / "comparison_table.tex"
        generate_latex_table(all_results, latex_path)
        paths["latex_table"] = latex_path

        # Generate recall-FPR plot
        tradeoff_path = output_dir / "recall_fpr_tradeoff.png"
        generate_recall_fpr_tradeoff_plot(all_results, tradeoff_path)
        paths["recall_fpr_plot"] = tradeoff_path

        # Save JSON summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "recall_floor": recall_floor,
                    "n_strategies": len(set(r.strategy_name for r in all_results)),
                    "results": [r.to_dict() for r in all_results],
                },
                f, indent=2
            )
        paths["summary_json"] = summary_path

    logger.info(f"Full report generated in {output_dir}")
    return paths


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Generate summary report from harness output")
    parser.add_argument("results_dir", type=Path, help="Directory containing harness CSV output")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for report")
    parser.add_argument("--recall-floor", type=float, default=0.95, help="Recall floor (tau)")

    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir / "summary"

    try:
        paths = generate_full_report(args.results_dir, output_dir, args.recall_floor)
        print(f"\nGenerated {len(paths)} output files:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
