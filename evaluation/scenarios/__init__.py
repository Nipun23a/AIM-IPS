"""
Scenario Generators for Threshold Strategy Evaluation

Configurable scenarios:
1. Legitimate traffic shifts (new endpoint, payload distribution, volume)
2. Slow-poisoning attacks
3. Baseline (stationary traffic)

Each scenario generates a stream of LabelledRequest objects with
pre-computed s_app scores from the frozen LightGBM classifier.
"""

from .base import ScenarioConfig, ScenarioGenerator
from .traffic_shift import (
    NewEndpointScenario,
    PayloadDistributionShiftScenario,
    VolumeChangeScenario,
)
from .slow_poisoning import SlowPoisoningScenario
from .stationary import StationaryScenario

__all__ = [
    "ScenarioConfig",
    "ScenarioGenerator",
    "NewEndpointScenario",
    "PayloadDistributionShiftScenario",
    "VolumeChangeScenario",
    "SlowPoisoningScenario",
    "StationaryScenario",
]
