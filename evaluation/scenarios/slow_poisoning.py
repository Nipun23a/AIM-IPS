"""
Slow-Poisoning Attack Scenario

Adversarial scenario where an attacker:
1. Gradually feeds borderline-benign requests to one endpoint
2. Drifts the adaptive threshold upward
3. Launches an attack that exploits the elevated threshold

Records:
- How far the threshold drifted
- Whether the attack was detected or missed
- Time-to-detection if detected

This scenario tests the anti-poisoning bound (delta) effectiveness.
"""

from dataclasses import dataclass, field
from typing import Iterator, Dict, Any, List, Tuple
import numpy as np

from .base import ScenarioConfig, ScenarioGenerator
from ..harness import LabelledRequest


@dataclass
class SlowPoisoningConfig(ScenarioConfig):
    """Configuration for slow-poisoning attack scenario."""

    # Target endpoint for poisoning
    target_endpoint: str = "/api/users/{id}"
    target_method: str = "GET"

    # Poisoning timeline
    poisoning_start_at: float = 0.2      # When poisoning begins
    poisoning_end_at: float = 0.7        # When poisoning ends, attack begins

    # Poisoning parameters
    poison_requests_per_window: int = 10  # Poison requests per confirmation window
    poison_score_start: float = 0.25      # Starting score for poison requests
    poison_score_end: float = 0.48        # Ending score (just below attack threshold)

    # The attacker's attack score (what they're trying to sneak through)
    attack_score: float = 0.52            # Just above the drifted threshold

    # Number of attack requests after poisoning
    n_attack_requests: int = 20

    # Regular background traffic continues throughout
    background_traffic_fraction: float = 0.7  # Fraction going to other endpoints

    # Endpoint for background (non-target) traffic
    background_endpoints: List[str] = field(default_factory=lambda: [
        "/api/orders/{id}",
        "/api/products",
        "/api/search",
    ])


class SlowPoisoningScenario(ScenarioGenerator):
    """
    Slow-poisoning attack scenario.

    Attack timeline:
    1. Warmup (0 to poisoning_start): Normal traffic establishes baselines
    2. Poisoning (poisoning_start to poisoning_end): Attacker sends gradually
       increasing borderline-benign scores to target endpoint
    3. Attack (poisoning_end to end): Attacker sends attack requests
       with scores that would have been caught pre-poisoning

    The scenario tracks:
    - Threshold drift over time
    - Attack success rate
    - Whether anti-poisoning bounds (delta) prevented the attack

    Background traffic to other endpoints continues throughout, and
    regular attacks at the configured rate are interleaved to ensure
    the evaluation captures overall performance, not just on the
    targeted endpoint.
    """

    def __init__(self, config: SlowPoisoningConfig):
        super().__init__(config)
        self.config: SlowPoisoningConfig = config
        self._poison_scores_sent: List[Tuple[float, float]] = []  # (time, score)
        self._attack_results: List[Tuple[float, float, bool]] = []  # (time, score, is_attack)

    def _get_poison_score(self, progress: float) -> float:
        """
        Get poison score based on progress through poisoning phase.
        Linear interpolation from start to end score.
        """
        return (
            self.config.poison_score_start +
            progress * (self.config.poison_score_end - self.config.poison_score_start)
        )

    def generate(self) -> Iterator[LabelledRequest]:
        interval = 1.0 / self.config.requests_per_second
        n_requests = int(self.config.duration_s * self.config.requests_per_second)

        poison_start_time = self.config.duration_s * self.config.poisoning_start_at
        poison_end_time = self.config.duration_s * self.config.poisoning_end_at
        poison_duration = poison_end_time - poison_start_time

        # Calculate poison injection timing
        # We want poison_requests_per_window requests per confirmation window (60s)
        poison_interval = 60.0 / self.config.poison_requests_per_window
        next_poison_time = poison_start_time

        # Attack tracking
        attacks_sent = 0
        attack_start_time = poison_end_time

        for idx in range(n_requests):
            stream_time_s = idx * interval

            # Phase 1: Warmup - normal traffic
            if stream_time_s < poison_start_time:
                is_attack = self._is_attack_request(idx, stream_time_s)
                yield self._generate_request(idx, stream_time_s, is_attack)
                continue

            # Phase 2: Poisoning - inject poison requests to target endpoint
            if stream_time_s < poison_end_time:
                # Check if it's time for a poison request
                if stream_time_s >= next_poison_time:
                    progress = (stream_time_s - poison_start_time) / poison_duration
                    poison_score = self._get_poison_score(progress)

                    self._poison_scores_sent.append((stream_time_s, poison_score))

                    yield LabelledRequest(
                        request_idx=idx,
                        stream_time_s=stream_time_s,
                        s_app=poison_score,
                        is_attack=False,  # Poison requests are labelled benign!
                        path=self.config.target_endpoint,
                        method=self.config.target_method,
                    )

                    next_poison_time += poison_interval
                    continue

                # Background traffic (to other endpoints or as attacks)
                is_attack = self._is_attack_request(idx, stream_time_s)
                if is_attack:
                    yield self._generate_request(idx, stream_time_s, True)
                else:
                    # Route to background endpoints
                    bg_endpoint = self.rng.choice(self.config.background_endpoints)
                    yield self._generate_request(
                        idx, stream_time_s, False, endpoint=bg_endpoint
                    )
                continue

            # Phase 3: Attack - send attack requests to target endpoint
            if attacks_sent < self.config.n_attack_requests:
                # Send attack request with the score we're trying to sneak through
                yield LabelledRequest(
                    request_idx=idx,
                    stream_time_s=stream_time_s,
                    s_app=self.config.attack_score,
                    is_attack=True,
                    path=self.config.target_endpoint,
                    method=self.config.target_method,
                )
                attacks_sent += 1
                continue

            # After attack phase: normal traffic resumes
            is_attack = self._is_attack_request(idx, stream_time_s)
            yield self._generate_request(idx, stream_time_s, is_attack)

    def get_poison_scores(self) -> List[Tuple[float, float]]:
        """Get list of (time, score) for poison requests sent."""
        return self._poison_scores_sent.copy()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "scenario_type": "slow_poisoning",
            "duration_s": self.config.duration_s,
            "target_endpoint": self.config.target_endpoint,
            "poisoning_start_at": self.config.poisoning_start_at,
            "poisoning_end_at": self.config.poisoning_end_at,
            "poison_score_start": self.config.poison_score_start,
            "poison_score_end": self.config.poison_score_end,
            "attack_score": self.config.attack_score,
            "n_attack_requests": self.config.n_attack_requests,
            "poison_requests_per_window": self.config.poison_requests_per_window,
            "attack_rate": self.config.attack_rate,
            "seed": self.config.seed,
        }


@dataclass
class PoisoningAnalysis:
    """Analysis results for slow-poisoning scenario."""
    target_context: str
    theta_global: float
    delta: float  # Anti-poisoning bound

    # Threshold drift
    threshold_before_poisoning: float
    threshold_after_poisoning: float
    max_threshold_drift: float
    drift_bounded_by_delta: bool

    # Attack outcomes
    n_attack_requests: int
    n_attacks_missed: int  # False negatives
    attack_miss_rate: float
    attack_score: float

    # Would the attack have succeeded without poisoning?
    attack_would_miss_without_poisoning: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_context": self.target_context,
            "theta_global": self.theta_global,
            "delta": self.delta,
            "threshold_before_poisoning": self.threshold_before_poisoning,
            "threshold_after_poisoning": self.threshold_after_poisoning,
            "max_threshold_drift": self.max_threshold_drift,
            "drift_bounded_by_delta": self.drift_bounded_by_delta,
            "n_attack_requests": self.n_attack_requests,
            "n_attacks_missed": self.n_attacks_missed,
            "attack_miss_rate": self.attack_miss_rate,
            "attack_score": self.attack_score,
            "attack_would_miss_without_poisoning": self.attack_would_miss_without_poisoning,
        }


def analyze_poisoning_results(
    per_request_metrics: List,  # List[StrategyMetrics]
    scenario_config: SlowPoisoningConfig,
    strategy_config: Dict[str, Any],
) -> PoisoningAnalysis:
    """
    Analyze poisoning scenario results.

    Args:
        per_request_metrics: Per-request metrics from harness
        scenario_config: Poisoning scenario configuration
        strategy_config: Strategy configuration (from get_config())

    Returns:
        PoisoningAnalysis with drift and attack outcome information
    """
    from ..strategies.base import derive_context_key

    target_context = derive_context_key(
        scenario_config.target_endpoint,
        scenario_config.target_method,
    )

    theta_global = strategy_config.get("theta_global", 0.5)
    delta = strategy_config.get("delta", 0.15)

    poison_start_time = scenario_config.duration_s * scenario_config.poisoning_start_at
    poison_end_time = scenario_config.duration_s * scenario_config.poisoning_end_at

    # Find thresholds before/after poisoning
    thresholds_before = [
        m.threshold_used for m in per_request_metrics
        if m.context_key == target_context and m.stream_time_s < poison_start_time
    ]
    thresholds_after = [
        m.threshold_used for m in per_request_metrics
        if m.context_key == target_context and m.stream_time_s >= poison_end_time
    ]

    threshold_before = np.mean(thresholds_before) if thresholds_before else theta_global
    threshold_after = np.mean(thresholds_after) if thresholds_after else theta_global
    max_drift = threshold_after - theta_global

    # Check if drift was bounded
    drift_bounded = abs(max_drift) <= delta + 0.001  # Small epsilon for float comparison

    # Analyze attack outcomes
    attack_metrics = [
        m for m in per_request_metrics
        if m.context_key == target_context
        and m.ground_truth_is_attack
        and m.stream_time_s >= poison_end_time
    ]

    n_attacks = len(attack_metrics)
    n_missed = sum(1 for m in attack_metrics if m.is_fn)
    miss_rate = n_missed / n_attacks if n_attacks > 0 else 0.0

    # Would attack have been caught without poisoning?
    # (i.e., is attack_score >= theta_global?)
    would_miss_without = scenario_config.attack_score < theta_global

    return PoisoningAnalysis(
        target_context=target_context,
        theta_global=theta_global,
        delta=delta,
        threshold_before_poisoning=float(threshold_before),
        threshold_after_poisoning=float(threshold_after),
        max_threshold_drift=float(max_drift),
        drift_bounded_by_delta=drift_bounded,
        n_attack_requests=n_attacks,
        n_attacks_missed=n_missed,
        attack_miss_rate=miss_rate,
        attack_score=scenario_config.attack_score,
        attack_would_miss_without_poisoning=would_miss_without,
    )
