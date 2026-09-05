"""
Fixed Threshold Strategy (Baseline)

Uses a static threshold theta for all requests regardless of context.
Default theta=0.5 as per specification.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import DecisionStrategy, Decision


@dataclass
class FixedThresholdStrategy(DecisionStrategy):
    """
    Fixed threshold baseline strategy.

    Decision rule: s_app >= theta => ATTACK, else BENIGN

    This is the simplest baseline with no adaptation or context awareness.
    """

    theta: float = 0.5

    def __post_init__(self):
        self.name = f"FixedThreshold(theta={self.theta})"
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {self.theta}")

    def decide(self, s_app: float, context_key: str) -> tuple[Decision, float]:
        """
        Fixed threshold decision.

        Complexity: O(1)
        No I/O, no state mutation.
        """
        decision = Decision.ATTACK if s_app >= self.theta else Decision.BENIGN
        return decision, self.theta

    def confirm_benign(self, s_app: float, context_key: str, stream_time_s: float) -> None:
        """No-op for fixed strategy."""
        pass

    def get_threshold(self, context_key: str) -> float:
        """Returns fixed theta for all contexts."""
        return self.theta

    def get_window_size(self, context_key: str) -> int:
        """No window for fixed strategy."""
        return 0

    def reset(self) -> None:
        """No state to reset."""
        pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "strategy_type": "fixed",
            "theta": self.theta,
        }
