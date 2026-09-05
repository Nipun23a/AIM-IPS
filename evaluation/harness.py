"""
Replay Harness for Threshold Strategy Evaluation

Feeds one labelled request stream through all strategies in lockstep.
Logs per-time-window metrics to CSV with microsecond decision timing.

Key features:
- Identical s_app score fed to all strategies per request
- Delayed label confirmation (D=60s stream time) for adaptive strategies
- Microsecond timer around decide() only (excludes classifier inference, I/O)
- Per-window precision, recall, F1, FPR, FNR, threshold values
- Per-request decision latency logging

Label source: Ground-truth labels only. Classifier scores are NEVER used
as benign confirmation signal to avoid feedback loops.
"""

import csv
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from collections import deque
import numpy as np

from .strategies.base import (
    DecisionStrategy,
    Decision,
    StrategyMetrics,
    derive_context_key,
)

logger = logging.getLogger(__name__)


@dataclass
class LabelledRequest:
    """
    A single labelled request for replay evaluation.

    This is the minimal record needed for threshold evaluation.
    The s_app score is pre-computed by the (frozen) LightGBM classifier.
    """
    request_idx: int
    stream_time_s: float      # Simulated stream time (seconds from start)
    s_app: float              # Classifier score: 1 - P(norm|x), in [0, 1]
    is_attack: bool           # Ground-truth label (True = attack)
    path: str                 # Original request path (e.g., /users/123)
    method: str               # HTTP method (GET, POST, etc.)

    @property
    def context_key(self) -> str:
        return derive_context_key(self.path, self.method)


@dataclass
class PendingConfirmation:
    """A benign request awaiting delayed confirmation."""
    request_idx: int
    s_app: float
    context_key: str
    confirm_at_time_s: float  # Stream time when confirmation becomes available


@dataclass
class WindowMetrics:
    """Aggregated metrics for a time window."""
    window_idx: int
    window_start_s: float
    window_end_s: float
    strategy_name: str

    # Counts
    n_requests: int
    n_attacks: int
    n_benign: int
    tp: int
    fp: int
    tn: int
    fn: int

    # Metrics (computed)
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float

    # Threshold info
    mean_threshold: float
    min_threshold: float
    max_threshold: float
    n_contexts: int

    # Latency (microseconds)
    mean_latency_us: float
    p50_latency_us: float
    p99_latency_us: float
    max_latency_us: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_idx": self.window_idx,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "strategy": self.strategy_name,
            "n_requests": self.n_requests,
            "n_attacks": self.n_attacks,
            "n_benign": self.n_benign,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "mean_threshold": self.mean_threshold,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "n_contexts": self.n_contexts,
            "mean_latency_us": self.mean_latency_us,
            "p50_latency_us": self.p50_latency_us,
            "p99_latency_us": self.p99_latency_us,
            "max_latency_us": self.max_latency_us,
        }


class ReplayHarness:
    """
    Replays labelled request stream through multiple strategies in lockstep.

    Features:
    - All strategies receive identical s_app scores
    - Microsecond timing around decide() only
    - Delayed label confirmation for adaptive strategies
    - Per-window and per-request metric logging
    - Recall floor checking

    Usage:
        harness = ReplayHarness(
            strategies=[fixed, validated, adaptive],
            confirmation_delay_s=60.0,
            window_size_s=300.0,
            recall_floor=0.95,
        )
        results = harness.run(request_stream)
        harness.save_results(output_dir)
    """

    def __init__(
        self,
        strategies: List[DecisionStrategy],
        confirmation_delay_s: float = 60.0,
        window_size_s: float = 300.0,  # 5-minute windows
        recall_floor: float = 0.95,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize replay harness.

        Args:
            strategies: List of decision strategies to evaluate in lockstep
            confirmation_delay_s: Delay before benign label is confirmed (D=60s)
            window_size_s: Window size for aggregated metrics
            recall_floor: Minimum acceptable recall (tau=0.95)
            output_dir: Directory for CSV output files
        """
        self.strategies = strategies
        self.confirmation_delay_s = confirmation_delay_s
        self.window_size_s = window_size_s
        self.recall_floor = recall_floor
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation/results")

        # Per-strategy state
        self._per_request_metrics: Dict[str, List[StrategyMetrics]] = {
            s.name: [] for s in strategies
        }
        self._window_metrics: Dict[str, List[WindowMetrics]] = {
            s.name: [] for s in strategies
        }
        self._pending_confirmations: Dict[str, deque] = {
            s.name: deque() for s in strategies
        }
        self._threshold_snapshots: Dict[str, List[Dict[str, float]]] = {
            s.name: [] for s in strategies
        }

        # Global state
        self._current_window_idx = 0
        self._window_start_s = 0.0

    def reset(self) -> None:
        """Reset harness and all strategies for fresh run."""
        for s in self.strategies:
            s.reset()

        self._per_request_metrics = {s.name: [] for s in self.strategies}
        self._window_metrics = {s.name: [] for s in self.strategies}
        self._pending_confirmations = {s.name: deque() for s in self.strategies}
        self._threshold_snapshots = {s.name: [] for s in self.strategies}
        self._current_window_idx = 0
        self._window_start_s = 0.0

    def _process_pending_confirmations(
        self,
        strategy: DecisionStrategy,
        current_time_s: float,
    ) -> None:
        """Process any pending confirmations that have become available."""
        pending = self._pending_confirmations[strategy.name]

        while pending and pending[0].confirm_at_time_s <= current_time_s:
            conf = pending.popleft()
            strategy.confirm_benign(
                s_app=conf.s_app,
                context_key=conf.context_key,
                stream_time_s=conf.confirm_at_time_s,
            )

    def _schedule_confirmation(
        self,
        strategy: DecisionStrategy,
        request: LabelledRequest,
    ) -> None:
        """Schedule a benign confirmation for later (if ground-truth is benign)."""
        if request.is_attack:
            # Never confirm attacks as benign
            return

        conf = PendingConfirmation(
            request_idx=request.request_idx,
            s_app=request.s_app,
            context_key=request.context_key,
            confirm_at_time_s=request.stream_time_s + self.confirmation_delay_s,
        )
        self._pending_confirmations[strategy.name].append(conf)

    def _evaluate_single_request(
        self,
        strategy: DecisionStrategy,
        request: LabelledRequest,
    ) -> StrategyMetrics:
        """
        Evaluate a single request against a strategy.

        Timing is around decide() only, using time.perf_counter_ns() for
        microsecond precision.
        """
        # Process any pending confirmations first
        self._process_pending_confirmations(strategy, request.stream_time_s)

        # Time the decision (microseconds)
        # Using perf_counter_ns for highest precision
        start_ns = time.perf_counter_ns()
        decision, threshold = strategy.decide(request.s_app, request.context_key)
        end_ns = time.perf_counter_ns()

        latency_us = (end_ns - start_ns) // 1000  # Convert ns to us

        # Schedule confirmation for benign requests
        self._schedule_confirmation(strategy, request)

        # Get current window size
        window_size = strategy.get_window_size(request.context_key)

        return StrategyMetrics(
            request_idx=request.request_idx,
            stream_time_s=request.stream_time_s,
            context_key=request.context_key,
            s_app=request.s_app,
            ground_truth_is_attack=request.is_attack,
            decision=decision,
            threshold_used=threshold,
            decision_latency_us=latency_us,
            window_size=window_size,
        )

    def _compute_window_metrics(
        self,
        strategy: DecisionStrategy,
        metrics: List[StrategyMetrics],
        window_idx: int,
        window_start_s: float,
        window_end_s: float,
    ) -> WindowMetrics:
        """Compute aggregated metrics for a window."""
        if not metrics:
            return WindowMetrics(
                window_idx=window_idx,
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                strategy_name=strategy.name,
                n_requests=0, n_attacks=0, n_benign=0,
                tp=0, fp=0, tn=0, fn=0,
                precision=0.0, recall=0.0, f1=0.0, fpr=0.0, fnr=0.0,
                mean_threshold=0.0, min_threshold=0.0, max_threshold=0.0,
                n_contexts=0,
                mean_latency_us=0.0, p50_latency_us=0.0, p99_latency_us=0.0, max_latency_us=0.0,
            )

        # Count outcomes
        tp = sum(1 for m in metrics if m.is_tp)
        fp = sum(1 for m in metrics if m.is_fp)
        tn = sum(1 for m in metrics if m.is_tn)
        fn = sum(1 for m in metrics if m.is_fn)

        n_attacks = tp + fn
        n_benign = tn + fp

        # Compute rates (with zero-division protection)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Threshold stats
        thresholds = [m.threshold_used for m in metrics]
        contexts = set(m.context_key for m in metrics)

        # Latency stats (in microseconds)
        latencies = [m.decision_latency_us for m in metrics]
        latencies_sorted = sorted(latencies)

        return WindowMetrics(
            window_idx=window_idx,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            strategy_name=strategy.name,
            n_requests=len(metrics),
            n_attacks=n_attacks,
            n_benign=n_benign,
            tp=tp, fp=fp, tn=tn, fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            fpr=fpr,
            fnr=fnr,
            mean_threshold=float(np.mean(thresholds)),
            min_threshold=float(np.min(thresholds)),
            max_threshold=float(np.max(thresholds)),
            n_contexts=len(contexts),
            mean_latency_us=float(np.mean(latencies)),
            p50_latency_us=float(latencies_sorted[len(latencies_sorted) // 2]),
            p99_latency_us=float(latencies_sorted[int(len(latencies_sorted) * 0.99)]),
            max_latency_us=float(max(latencies)),
        )

    def run(
        self,
        request_stream: Iterator[LabelledRequest],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run evaluation on request stream.

        Args:
            request_stream: Iterator of LabelledRequest objects
            verbose: Print progress updates

        Returns:
            Summary dict with overall metrics per strategy
        """
        self.reset()

        # Convert to list for multi-pass (strategies evaluated in lockstep)
        requests = list(request_stream)
        if not requests:
            raise ValueError("Empty request stream")

        logger.info(f"Running evaluation on {len(requests)} requests")

        # Track current window bounds
        self._window_start_s = requests[0].stream_time_s
        window_end_s = self._window_start_s + self.window_size_s
        current_window_metrics: Dict[str, List[StrategyMetrics]] = {
            s.name: [] for s in self.strategies
        }

        for req_idx, request in enumerate(requests):
            # Check if we've moved to a new window
            while request.stream_time_s >= window_end_s:
                # Finalize current window for all strategies
                for strategy in self.strategies:
                    wm = self._compute_window_metrics(
                        strategy,
                        current_window_metrics[strategy.name],
                        self._current_window_idx,
                        self._window_start_s,
                        window_end_s,
                    )
                    self._window_metrics[strategy.name].append(wm)

                    # Snapshot thresholds for this window
                    self._threshold_snapshots[strategy.name].append(
                        strategy.get_all_context_thresholds()
                    )

                    # Clear window metrics
                    current_window_metrics[strategy.name] = []

                # Move to next window
                self._current_window_idx += 1
                self._window_start_s = window_end_s
                window_end_s = self._window_start_s + self.window_size_s

            # Evaluate request against all strategies (lockstep)
            for strategy in self.strategies:
                metrics = self._evaluate_single_request(strategy, request)
                self._per_request_metrics[strategy.name].append(metrics)
                current_window_metrics[strategy.name].append(metrics)

            # Progress logging
            if verbose and (req_idx + 1) % 10000 == 0:
                logger.info(f"Processed {req_idx + 1}/{len(requests)} requests")

        # Finalize last window
        for strategy in self.strategies:
            if current_window_metrics[strategy.name]:
                wm = self._compute_window_metrics(
                    strategy,
                    current_window_metrics[strategy.name],
                    self._current_window_idx,
                    self._window_start_s,
                    requests[-1].stream_time_s,
                )
                self._window_metrics[strategy.name].append(wm)
                self._threshold_snapshots[strategy.name].append(
                    strategy.get_all_context_thresholds()
                )

        # Compute overall summary
        return self._compute_summary()

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute overall summary statistics."""
        summary = {
            "recall_floor": self.recall_floor,
            "confirmation_delay_s": self.confirmation_delay_s,
            "window_size_s": self.window_size_s,
            "strategies": {},
        }

        for strategy in self.strategies:
            all_metrics = self._per_request_metrics[strategy.name]
            if not all_metrics:
                continue

            tp = sum(1 for m in all_metrics if m.is_tp)
            fp = sum(1 for m in all_metrics if m.is_fp)
            tn = sum(1 for m in all_metrics if m.is_tn)
            fn = sum(1 for m in all_metrics if m.is_fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            latencies = [m.decision_latency_us for m in all_metrics]
            latencies_sorted = sorted(latencies)

            passes_recall_floor = recall >= self.recall_floor

            summary["strategies"][strategy.name] = {
                "config": strategy.get_config(),
                "n_requests": len(all_metrics),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr,
                "passes_recall_floor": passes_recall_floor,
                "recall_floor": self.recall_floor,
                "mean_latency_us": float(np.mean(latencies)),
                "p50_latency_us": float(latencies_sorted[len(latencies_sorted) // 2]),
                "p99_latency_us": float(latencies_sorted[int(len(latencies_sorted) * 0.99)]),
                "max_latency_us": float(max(latencies)),
            }

        return summary

    def save_results(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save all results to CSV files.

        Returns dict mapping result type to file path.
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save per-request metrics (one file per strategy)
        for strategy in self.strategies:
            safe_name = strategy.name.replace("(", "_").replace(")", "_").replace("=", "_").replace(",", "_").replace(" ", "")
            path = output_dir / f"per_request_{safe_name}.csv"
            paths[f"per_request_{strategy.name}"] = path

            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "request_idx", "stream_time_s", "context_key", "s_app",
                    "ground_truth_is_attack", "decision", "threshold_used",
                    "decision_latency_us", "window_size", "is_correct",
                    "is_tp", "is_fp", "is_tn", "is_fn"
                ])
                for m in self._per_request_metrics[strategy.name]:
                    writer.writerow([
                        m.request_idx, m.stream_time_s, m.context_key, m.s_app,
                        int(m.ground_truth_is_attack), m.decision.name, m.threshold_used,
                        m.decision_latency_us, m.window_size, int(m.is_correct),
                        int(m.is_tp), int(m.is_fp), int(m.is_tn), int(m.is_fn)
                    ])

        # Save per-window metrics (all strategies in one file)
        path = output_dir / "per_window_metrics.csv"
        paths["per_window"] = path

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "window_idx", "window_start_s", "window_end_s", "strategy",
                "n_requests", "n_attacks", "n_benign",
                "tp", "fp", "tn", "fn",
                "precision", "recall", "f1", "fpr", "fnr",
                "mean_threshold", "min_threshold", "max_threshold", "n_contexts",
                "mean_latency_us", "p50_latency_us", "p99_latency_us", "max_latency_us",
            ])
            writer.writeheader()
            for strategy in self.strategies:
                for wm in self._window_metrics[strategy.name]:
                    writer.writerow(wm.to_dict())

        # Save threshold snapshots for adaptive strategies
        for strategy in self.strategies:
            if hasattr(strategy, 'get_all_context_thresholds'):
                snapshots = self._threshold_snapshots[strategy.name]
                if snapshots and any(s for s in snapshots):
                    safe_name = strategy.name.replace("(", "_").replace(")", "_").replace("=", "_").replace(",", "_").replace(" ", "")
                    path = output_dir / f"threshold_snapshots_{safe_name}.csv"
                    paths[f"threshold_snapshots_{strategy.name}"] = path

                    # Collect all context keys
                    all_contexts = set()
                    for snap in snapshots:
                        all_contexts.update(snap.keys())
                    all_contexts = sorted(all_contexts)

                    with open(path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["window_idx"] + list(all_contexts))
                        for idx, snap in enumerate(snapshots):
                            row = [idx] + [snap.get(ctx, "") for ctx in all_contexts]
                            writer.writerow(row)

        logger.info(f"Results saved to {output_dir}")
        return paths

    def get_per_request_metrics(self, strategy_name: str) -> List[StrategyMetrics]:
        """Get per-request metrics for a strategy."""
        return self._per_request_metrics.get(strategy_name, [])

    def get_window_metrics(self, strategy_name: str) -> List[WindowMetrics]:
        """Get per-window metrics for a strategy."""
        return self._window_metrics.get(strategy_name, [])
