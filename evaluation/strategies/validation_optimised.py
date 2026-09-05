"""
Validation-Optimised Global Threshold Strategy

Uses theta_global fitted on held-out validation split (from threshold_optimizer.py).
Never fitted on the replay stream - determined once before evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from .base import DecisionStrategy, Decision


# Path to the threshold file produced by threshold_optimizer.py
DEFAULT_THRESHOLD_PATH = Path("models/network_level/ensemble_threshold.pkl")
VALIDATION_THRESHOLD_PATH = Path("anomly_detector/src/network_level_attacks_anomality/validation_threshold.pkl")


def load_validation_threshold(path: Optional[Path] = None) -> float:
    """
    Load the validation-optimised threshold from disk.

    Falls back to default if file not found.
    """
    import joblib

    search_paths = [
        path,
        VALIDATION_THRESHOLD_PATH,
        DEFAULT_THRESHOLD_PATH,
        Path("models/application_layer/validation_threshold.pkl"),
    ]

    for p in search_paths:
        if p is not None and p.exists():
            try:
                data = joblib.load(p)
                if isinstance(data, dict) and "threshold" in data:
                    return float(data["threshold"])
                elif isinstance(data, dict) and "best_threshold" in data:
                    return float(data["best_threshold"])
                elif isinstance(data, (int, float)):
                    return float(data)
            except Exception:
                continue

    # Fallback: return 0.5 if no threshold file found
    return 0.5


@dataclass
class ValidationOptimisedStrategy(DecisionStrategy):
    """
    Validation-optimised global threshold strategy.

    Uses theta_global fitted on a held-out validation split during training.
    This threshold is determined ONCE before evaluation and never updated
    during the replay stream.

    Decision rule: s_app >= theta_global => ATTACK, else BENIGN
    """

    theta_global: Optional[float] = None
    threshold_path: Optional[Path] = None

    def __post_init__(self):
        # Load or use provided threshold
        if self.theta_global is None:
            self.theta_global = load_validation_threshold(self.threshold_path)

        if not 0.0 <= self.theta_global <= 1.0:
            raise ValueError(f"theta_global must be in [0, 1], got {self.theta_global}")

        self.name = f"ValidationOptimised(theta={self.theta_global:.4f})"

    def decide(self, s_app: float, context_key: str) -> tuple[Decision, float]:
        """
        Global threshold decision (same for all contexts).

        Complexity: O(1)
        No I/O, no state mutation.
        """
        decision = Decision.ATTACK if s_app >= self.theta_global else Decision.BENIGN
        return decision, self.theta_global

    def confirm_benign(self, s_app: float, context_key: str, stream_time_s: float) -> None:
        """No-op for validation-optimised strategy (no runtime adaptation)."""
        pass

    def get_threshold(self, context_key: str) -> float:
        """Returns theta_global for all contexts."""
        return self.theta_global

    def get_window_size(self, context_key: str) -> int:
        """No window for global strategy."""
        return 0

    def reset(self) -> None:
        """No runtime state to reset (threshold is fixed from validation)."""
        pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "strategy_type": "validation_optimised",
            "theta_global": self.theta_global,
            "threshold_path": str(self.threshold_path) if self.threshold_path else None,
        }


def fit_threshold_on_validation(
    s_app_scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Fit optimal threshold on validation data.

    This function should be called ONCE on a held-out validation split,
    NEVER on the replay/test stream.

    Args:
        s_app_scores: Array of classifier scores [0, 1]
        labels: Binary labels (1 = attack, 0 = benign)
        metric: Optimization target ("f1", "f2", "balanced_accuracy")

    Returns:
        Optimal threshold value
    """
    from sklearn.metrics import f1_score, fbeta_score, balanced_accuracy_score

    best_threshold = 0.5
    best_score = 0.0

    # Grid search over thresholds
    for threshold in np.arange(0.05, 0.95, 0.01):
        preds = (s_app_scores >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(labels, preds, zero_division=0)
        elif metric == "f2":
            score = fbeta_score(labels, preds, beta=2, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(labels, preds)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold
