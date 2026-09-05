"""
Decision Strategy Interface and Implementations

Three strategies consuming identical classifier scores:
1. FixedThresholdStrategy - static threshold (baseline)
2. ValidationOptimisedStrategy - global threshold fitted on held-out split
3. ContextAdaptiveStrategy - per-context adaptive threshold
"""

from .base import DecisionStrategy, Decision, StrategyMetrics
from .fixed import FixedThresholdStrategy
from .validation_optimised import ValidationOptimisedStrategy
from .context_adaptive import ContextAdaptiveStrategy

__all__ = [
    "DecisionStrategy",
    "Decision",
    "StrategyMetrics",
    "FixedThresholdStrategy",
    "ValidationOptimisedStrategy",
    "ContextAdaptiveStrategy",
]
