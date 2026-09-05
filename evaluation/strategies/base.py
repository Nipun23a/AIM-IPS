"""
Decision Strategy Abstract Base Class

All strategies consume the identical LightGBM application score s_app = 1 - P(norm|x)
and produce a binary attack/benign decision. The five-tier fused-score engine is
out of scope for this experiment.

Decision latency is timed around decide() only, excluding classifier inference and I/O.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import re


class Decision(Enum):
    """Binary decision output."""
    BENIGN = 0
    ATTACK = 1


@dataclass
class StrategyMetrics:
    """
    Per-request metrics collected during replay.
    All timing in microseconds (integer) to avoid float precision issues.
    """
    request_idx: int
    stream_time_s: float          # Simulated stream time in seconds
    context_key: str              # (path_template, method)
    s_app: float                  # Classifier score (input)
    ground_truth_is_attack: bool  # True label
    decision: Decision            # Strategy output
    threshold_used: float         # Threshold at decision time
    decision_latency_us: int      # Microseconds for decide() call only
    window_size: int              # Current benign window size for context (0 for non-adaptive)

    @property
    def is_correct(self) -> bool:
        predicted_attack = (self.decision == Decision.ATTACK)
        return predicted_attack == self.ground_truth_is_attack

    @property
    def is_tp(self) -> bool:
        return self.ground_truth_is_attack and self.decision == Decision.ATTACK

    @property
    def is_fp(self) -> bool:
        return (not self.ground_truth_is_attack) and self.decision == Decision.ATTACK

    @property
    def is_tn(self) -> bool:
        return (not self.ground_truth_is_attack) and self.decision == Decision.BENIGN

    @property
    def is_fn(self) -> bool:
        return self.ground_truth_is_attack and self.decision == Decision.BENIGN


def normalise_path(path: str) -> str:
    """
    Normalise path to template form for context key derivation.

    Rules:
    - Numeric segments -> {id}
    - UUID segments -> {uuid}
    - Preserves structure for endpoint grouping

    Examples:
        /users/123 -> /users/{id}
        /users/456 -> /users/{id}
        /orders/550e8400-e29b-41d4-a716-446655440000 -> /orders/{uuid}
        /api/v1/items/42/reviews -> /api/v1/items/{id}/reviews
    """
    if not path or path == "/":
        return "/"

    # UUID pattern (8-4-4-4-12 hex)
    uuid_pattern = re.compile(
        r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    )
    # Pure numeric pattern
    numeric_pattern = re.compile(r'^[0-9]+$')
    # Hex ID pattern (common in MongoDB ObjectIds, etc.)
    hex_id_pattern = re.compile(r'^[0-9a-fA-F]{24}$')

    segments = path.strip("/").split("/")
    normalised = []

    for seg in segments:
        if not seg:
            continue
        if uuid_pattern.match(seg):
            normalised.append("{uuid}")
        elif hex_id_pattern.match(seg):
            normalised.append("{id}")
        elif numeric_pattern.match(seg):
            normalised.append("{id}")
        else:
            normalised.append(seg)

    return "/" + "/".join(normalised) if normalised else "/"


def derive_context_key(path: str, method: str) -> str:
    """
    Derive context key from request path and HTTP method.

    Returns: "(normalised_path, METHOD)" string
    """
    norm_path = normalise_path(path)
    return f"({norm_path}, {method.upper()})"


@dataclass
class DecisionStrategy(ABC):
    """
    Abstract base class for decision strategies.

    All strategies:
    - Receive identical s_app score (1 - P(norm|x) from LightGBM)
    - Produce binary ATTACK/BENIGN decision
    - Track their own state (if any)
    - Support delayed label confirmation for adaptive strategies
    """

    name: str = field(init=False)

    @abstractmethod
    def decide(self, s_app: float, context_key: str) -> tuple[Decision, float]:
        """
        Make a decision given classifier score and context.

        Args:
            s_app: Classifier score in [0, 1], where higher = more likely attack
            context_key: Derived from (normalised_path, method)

        Returns:
            (decision, threshold_used)

        NOTE: This method is timed. Do not include I/O or heavy computation
        that wouldn't be in the production decision path.
        """
        pass

    @abstractmethod
    def confirm_benign(self, s_app: float, context_key: str, stream_time_s: float) -> None:
        """
        Confirm a request as benign (ground-truth label, after delay).

        For adaptive strategies: add s_app to the context's benign window.
        For fixed strategies: no-op.

        This is called by the harness ONLY for requests with ground_truth=benign,
        ONLY after the confirmation delay D has elapsed in stream time.
        """
        pass

    @abstractmethod
    def get_threshold(self, context_key: str) -> float:
        """
        Get current threshold for a context (for logging/plotting).
        """
        pass

    @abstractmethod
    def get_window_size(self, context_key: str) -> int:
        """
        Get current benign window size for a context.
        Returns 0 for non-adaptive strategies.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset all internal state for a fresh evaluation run.
        """
        pass

    def get_all_context_thresholds(self) -> Dict[str, float]:
        """
        Get all current per-context thresholds (for snapshot logging).
        Default returns empty dict; override in adaptive strategies.
        """
        return {}

    def get_config(self) -> Dict[str, Any]:
        """
        Return strategy configuration for reproducibility logging.
        """
        return {"name": self.name}
