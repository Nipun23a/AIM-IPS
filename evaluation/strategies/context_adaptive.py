"""
Context-Aware Adaptive Threshold Strategy

Per-context adaptive thresholding based on recent confirmed-benign traffic.

Design:
- Context key: (normalised path template, HTTP method)
- Per-context state: sliding window of last N=500 confirmed-benign s_app values
- Threshold rule: theta_ctx = clip(Q_{1-alpha}(window) + margin,
                                   theta_global - delta, theta_global + delta)
- Cold start: if context has < n_min samples, use theta_global
- Anti-poisoning bound: delta limits drift in either direction

Label source: Ground-truth labels only, available after delay D (simulating
production 60-second no-incident auto-confirm rule). Never uses classifier
score as benign signal.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Any, Optional
import numpy as np

from .base import DecisionStrategy, Decision
from .validation_optimised import load_validation_threshold


@dataclass
class ContextState:
    """Per-context sliding window state."""
    window: deque = field(default_factory=lambda: deque(maxlen=500))

    def add_benign_score(self, s_app: float) -> None:
        """Add a confirmed-benign score to the window."""
        self.window.append(s_app)

    def size(self) -> int:
        return len(self.window)

    def quantile(self, q: float) -> float:
        """Compute quantile of window values."""
        if not self.window:
            return 0.5  # Fallback
        return float(np.quantile(list(self.window), q))


@dataclass
class ContextAdaptiveStrategy(DecisionStrategy):
    """
    Context-aware adaptive threshold strategy.

    Threshold rule:
        theta_ctx = clip(Q_{1-alpha}(window) + margin,
                         theta_global - delta, theta_global + delta)

    Where:
        - Q_{1-alpha} is the (1-alpha) quantile of the benign window
        - alpha = target per-context FPR (default 0.02)
        - margin = safety margin added to quantile (default 0.02)
        - theta_global = validation-optimised global threshold
        - delta = anti-poisoning bound (default 0.15)

    Cold start:
        If context has < n_min (default 50) benign samples, use theta_global.

    Parameters:
        theta_global: Validation-optimised global threshold (loaded if None)
        alpha: Target per-context FPR (default 0.02)
        margin: Safety margin above quantile (default 0.02)
        delta: Anti-poisoning bound on drift (default 0.15)
        n_min: Minimum samples before adaptation (default 50)
        window_size: Maximum benign samples per context (default 500)
        enable_anti_poisoning: If False, delta bound is disabled (for ablation)
    """

    theta_global: Optional[float] = None
    alpha: float = 0.02
    margin: float = 0.02
    delta: float = 0.15
    n_min: int = 50
    window_size: int = 500
    enable_anti_poisoning: bool = True

    # Internal state (not constructor args)
    _contexts: Dict[str, ContextState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        # Load global threshold if not provided
        if self.theta_global is None:
            self.theta_global = load_validation_threshold()

        # Validate parameters
        if not 0.0 <= self.theta_global <= 1.0:
            raise ValueError(f"theta_global must be in [0, 1], got {self.theta_global}")
        if not 0.0 < self.alpha < 0.5:
            raise ValueError(f"alpha must be in (0, 0.5), got {self.alpha}")
        if not 0.0 <= self.margin <= 0.5:
            raise ValueError(f"margin must be in [0, 0.5], got {self.margin}")
        if not 0.0 < self.delta <= 0.5:
            raise ValueError(f"delta must be in (0, 0.5], got {self.delta}")
        if self.n_min < 1:
            raise ValueError(f"n_min must be >= 1, got {self.n_min}")

        anti_poison_str = f", delta={self.delta}" if self.enable_anti_poisoning else ", unbounded"
        self.name = f"ContextAdaptive(alpha={self.alpha}, margin={self.margin}{anti_poison_str})"

    def _get_or_create_context(self, context_key: str) -> ContextState:
        """Get or create context state."""
        if context_key not in self._contexts:
            self._contexts[context_key] = ContextState(
                window=deque(maxlen=self.window_size)
            )
        return self._contexts[context_key]

    def _compute_adaptive_threshold(self, ctx: ContextState) -> float:
        """
        Compute adaptive threshold for a context.

        Returns theta_global if cold start (< n_min samples).
        """
        if ctx.size() < self.n_min:
            # Cold start: use global threshold
            return self.theta_global

        # Compute (1 - alpha) quantile of benign window
        q = 1.0 - self.alpha
        quantile_value = ctx.quantile(q)

        # Add safety margin
        raw_threshold = quantile_value + self.margin

        # Apply anti-poisoning bounds
        if self.enable_anti_poisoning:
            lower_bound = self.theta_global - self.delta
            upper_bound = self.theta_global + self.delta
            theta_ctx = np.clip(raw_threshold, lower_bound, upper_bound)
        else:
            # Unbounded (for ablation study)
            theta_ctx = np.clip(raw_threshold, 0.0, 1.0)

        return float(theta_ctx)

    def decide(self, s_app: float, context_key: str) -> tuple[Decision, float]:
        """
        Context-adaptive threshold decision.

        Complexity: O(1) amortized (quantile is O(N) but N is bounded by window_size)

        NOTE: The quantile computation on a 500-element deque is ~microseconds.
        For production, consider maintaining a sorted structure or approximate quantile.
        """
        ctx = self._get_or_create_context(context_key)
        threshold = self._compute_adaptive_threshold(ctx)
        decision = Decision.ATTACK if s_app >= threshold else Decision.BENIGN
        return decision, threshold

    def confirm_benign(self, s_app: float, context_key: str, stream_time_s: float) -> None:
        """
        Add confirmed-benign score to context window.

        Called by harness ONLY for ground-truth benign requests,
        ONLY after confirmation delay D has elapsed.

        NOTE: stream_time_s is provided for logging/debugging but not used
        in the threshold computation.
        """
        ctx = self._get_or_create_context(context_key)
        ctx.add_benign_score(s_app)

    def get_threshold(self, context_key: str) -> float:
        """Get current threshold for a context."""
        if context_key not in self._contexts:
            return self.theta_global
        ctx = self._contexts[context_key]
        return self._compute_adaptive_threshold(ctx)

    def get_window_size(self, context_key: str) -> int:
        """Get current benign window size for a context."""
        if context_key not in self._contexts:
            return 0
        return self._contexts[context_key].size()

    def reset(self) -> None:
        """Clear all context state for fresh evaluation run."""
        self._contexts.clear()

    def get_all_context_thresholds(self) -> Dict[str, float]:
        """Get all current per-context thresholds."""
        return {
            key: self._compute_adaptive_threshold(ctx)
            for key, ctx in self._contexts.items()
        }

    def get_context_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed stats per context (for analysis)."""
        stats = {}
        for key, ctx in self._contexts.items():
            if ctx.size() > 0:
                window_list = list(ctx.window)
                stats[key] = {
                    "window_size": ctx.size(),
                    "threshold": self._compute_adaptive_threshold(ctx),
                    "is_cold_start": ctx.size() < self.n_min,
                    "quantile_value": ctx.quantile(1.0 - self.alpha) if ctx.size() >= self.n_min else None,
                    "window_mean": float(np.mean(window_list)),
                    "window_std": float(np.std(window_list)),
                    "window_max": float(np.max(window_list)),
                }
        return stats

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "strategy_type": "context_adaptive",
            "theta_global": self.theta_global,
            "alpha": self.alpha,
            "margin": self.margin,
            "delta": self.delta,
            "n_min": self.n_min,
            "window_size": self.window_size,
            "enable_anti_poisoning": self.enable_anti_poisoning,
        }

    def get_drift_from_global(self, context_key: str) -> float:
        """
        Get how far the context threshold has drifted from global.

        Useful for slow-poisoning analysis.
        """
        current = self.get_threshold(context_key)
        return current - self.theta_global
