"""
Stationary Traffic Scenario (Baseline)

Generates traffic with stable distributions throughout the evaluation.
No distribution shifts, no new endpoints, no poisoning.
Used as baseline for comparison.
"""

from dataclasses import dataclass
from typing import Iterator, Dict, Any

from .base import ScenarioConfig, ScenarioGenerator
from ..harness import LabelledRequest


@dataclass
class StationaryConfig(ScenarioConfig):
    """Configuration for stationary traffic scenario."""
    pass  # Uses all defaults from ScenarioConfig


class StationaryScenario(ScenarioGenerator):
    """
    Stationary traffic scenario.

    Generates traffic with:
    - Fixed endpoint distribution
    - Fixed benign/attack score distributions
    - Fixed attack rate
    - No distribution shifts

    This is the baseline scenario for evaluating threshold strategies
    under stable conditions.
    """

    def __init__(self, config: StationaryConfig):
        super().__init__(config)

    def generate(self) -> Iterator[LabelledRequest]:
        """Generate stationary traffic stream."""
        interval = 1.0 / self.config.requests_per_second
        n_requests = int(self.config.duration_s * self.config.requests_per_second)

        for idx in range(n_requests):
            stream_time_s = idx * interval
            is_attack = self._is_attack_request(idx, stream_time_s)
            yield self._generate_request(idx, stream_time_s, is_attack)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "scenario_type": "stationary",
            "duration_s": self.config.duration_s,
            "requests_per_second": self.config.requests_per_second,
            "attack_rate": self.config.attack_rate,
            "benign_score_mean": self.config.benign_score_mean,
            "benign_score_std": self.config.benign_score_std,
            "attack_score_mean": self.config.attack_score_mean,
            "attack_score_std": self.config.attack_score_std,
            "n_endpoints": len(self.config.endpoints),
            "seed": self.config.seed,
        }
