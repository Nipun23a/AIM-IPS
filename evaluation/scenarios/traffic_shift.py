"""
Legitimate Traffic Shift Scenarios

Three types of distribution shifts with attacks interleaved at fixed rate:
1. New endpoint appearing mid-stream
2. Payload-length/encoding distribution shift on existing endpoint
3. Traffic volume change (rate increase/decrease)

The classifier is NEVER modified - only the traffic characteristics change.
"""

from dataclasses import dataclass, field
from typing import Iterator, Dict, Any, List, Optional
import numpy as np

from .base import ScenarioConfig, ScenarioGenerator
from ..harness import LabelledRequest


@dataclass
class NewEndpointConfig(ScenarioConfig):
    """Configuration for new endpoint scenario."""
    # When the new endpoint appears (fraction of total duration)
    new_endpoint_appear_at: float = 0.5

    # The new endpoint path
    new_endpoint_path: str = "/api/v2/recommendations"
    new_endpoint_method: str = "GET"

    # Fraction of traffic to new endpoint after it appears
    new_endpoint_traffic_fraction: float = 0.3

    # Score distribution for new endpoint (may differ from existing)
    new_endpoint_benign_mean: float = 0.20
    new_endpoint_benign_std: float = 0.10


class NewEndpointScenario(ScenarioGenerator):
    """
    New endpoint appearing mid-stream scenario.

    Simulates a deployment of new API functionality. The adaptive
    strategy must handle cold start for the new context.

    Timeline:
    - Phase 1 (0 to appear_at): Traffic to existing endpoints only
    - Phase 2 (appear_at to end): New endpoint receives traffic_fraction
      of benign requests

    Attack rate remains constant throughout.
    """

    def __init__(self, config: NewEndpointConfig):
        super().__init__(config)
        self.config: NewEndpointConfig = config

    def generate(self) -> Iterator[LabelledRequest]:
        interval = 1.0 / self.config.requests_per_second
        n_requests = int(self.config.duration_s * self.config.requests_per_second)
        appear_time = self.config.duration_s * self.config.new_endpoint_appear_at

        for idx in range(n_requests):
            stream_time_s = idx * interval
            is_attack = self._is_attack_request(idx, stream_time_s)

            # After appearance, route some benign traffic to new endpoint
            if not is_attack and stream_time_s >= appear_time:
                if self.rng.random() < self.config.new_endpoint_traffic_fraction:
                    # Traffic to new endpoint
                    s_app = self._sample_benign_score(
                        mean=self.config.new_endpoint_benign_mean,
                        std=self.config.new_endpoint_benign_std,
                    )
                    yield LabelledRequest(
                        request_idx=idx,
                        stream_time_s=stream_time_s,
                        s_app=s_app,
                        is_attack=False,
                        path=self.config.new_endpoint_path,
                        method=self.config.new_endpoint_method,
                    )
                    continue

            # Regular traffic (existing endpoints or attacks)
            yield self._generate_request(idx, stream_time_s, is_attack)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "scenario_type": "new_endpoint",
            "duration_s": self.config.duration_s,
            "new_endpoint_appear_at": self.config.new_endpoint_appear_at,
            "new_endpoint_path": self.config.new_endpoint_path,
            "new_endpoint_traffic_fraction": self.config.new_endpoint_traffic_fraction,
            "new_endpoint_benign_mean": self.config.new_endpoint_benign_mean,
            "attack_rate": self.config.attack_rate,
            "seed": self.config.seed,
        }


@dataclass
class PayloadDistributionShiftConfig(ScenarioConfig):
    """Configuration for payload distribution shift scenario."""
    # When the shift begins (fraction of total duration)
    shift_start_at: float = 0.4

    # How long the transition takes (fraction of total duration)
    transition_duration: float = 0.1

    # Target endpoint for the shift
    target_endpoint: str = "/api/search"

    # New score distribution after shift
    # (e.g., new encoding, longer payloads → higher baseline scores)
    shifted_benign_mean: float = 0.30
    shifted_benign_std: float = 0.12


class PayloadDistributionShiftScenario(ScenarioGenerator):
    """
    Payload distribution shift scenario.

    Simulates legitimate changes in traffic patterns:
    - New client library with different encoding
    - Feature flag enabling longer request bodies
    - API version migration

    The shift affects benign score distribution for a specific endpoint.
    Attack scores remain unchanged.

    Timeline:
    - Phase 1 (0 to shift_start): Original distribution
    - Phase 2 (shift_start to shift_start + transition): Linear blend
    - Phase 3 (transition_end to end): Shifted distribution
    """

    def __init__(self, config: PayloadDistributionShiftConfig):
        super().__init__(config)
        self.config: PayloadDistributionShiftConfig = config

    def _get_blend_factor(self, stream_time_s: float) -> float:
        """
        Get blending factor for distribution transition.
        Returns 0.0 before shift, 1.0 after transition, linear blend during.
        """
        shift_start = self.config.duration_s * self.config.shift_start_at
        transition_end = shift_start + (self.config.duration_s * self.config.transition_duration)

        if stream_time_s < shift_start:
            return 0.0
        elif stream_time_s >= transition_end:
            return 1.0
        else:
            # Linear interpolation
            progress = (stream_time_s - shift_start) / (transition_end - shift_start)
            return progress

    def generate(self) -> Iterator[LabelledRequest]:
        interval = 1.0 / self.config.requests_per_second
        n_requests = int(self.config.duration_s * self.config.requests_per_second)

        for idx in range(n_requests):
            stream_time_s = idx * interval
            is_attack = self._is_attack_request(idx, stream_time_s)
            endpoint = self._sample_endpoint()

            if not is_attack and endpoint == self.config.target_endpoint:
                # Apply distribution shift
                blend = self._get_blend_factor(stream_time_s)

                if blend == 0.0:
                    s_app = self._sample_benign_score()
                elif blend == 1.0:
                    s_app = self._sample_benign_score(
                        mean=self.config.shifted_benign_mean,
                        std=self.config.shifted_benign_std,
                    )
                else:
                    # Blend: sample from both, then interpolate
                    score_old = self._sample_benign_score()
                    score_new = self._sample_benign_score(
                        mean=self.config.shifted_benign_mean,
                        std=self.config.shifted_benign_std,
                    )
                    s_app = (1 - blend) * score_old + blend * score_new

                yield LabelledRequest(
                    request_idx=idx,
                    stream_time_s=stream_time_s,
                    s_app=s_app,
                    is_attack=False,
                    path=endpoint,
                    method=self._sample_method(),
                )
            else:
                yield self._generate_request(
                    idx, stream_time_s, is_attack, endpoint=endpoint
                )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "scenario_type": "payload_distribution_shift",
            "duration_s": self.config.duration_s,
            "shift_start_at": self.config.shift_start_at,
            "transition_duration": self.config.transition_duration,
            "target_endpoint": self.config.target_endpoint,
            "shifted_benign_mean": self.config.shifted_benign_mean,
            "shifted_benign_std": self.config.shifted_benign_std,
            "attack_rate": self.config.attack_rate,
            "seed": self.config.seed,
        }


@dataclass
class VolumeChangeConfig(ScenarioConfig):
    """Configuration for volume change scenario."""
    # When volume change begins
    volume_change_at: float = 0.5

    # Volume multiplier (>1 = increase, <1 = decrease)
    volume_multiplier: float = 3.0

    # Gradual or instant change
    gradual_transition: bool = True
    transition_duration: float = 0.1  # fraction of total duration


class VolumeChangeScenario(ScenarioGenerator):
    """
    Traffic volume change scenario.

    Simulates traffic spikes or drops:
    - Marketing campaign driving traffic
    - Time-of-day variations
    - Service degradation causing retries

    Volume changes affect all endpoints proportionally.
    Attack rate (proportion) remains constant.

    Timeline:
    - Phase 1: Base volume
    - Phase 2 (optional transition): Gradual change
    - Phase 3: New volume level
    """

    def __init__(self, config: VolumeChangeConfig):
        super().__init__(config)
        self.config: VolumeChangeConfig = config

    def _get_volume_multiplier(self, stream_time_s: float) -> float:
        """Get current volume multiplier."""
        change_start = self.config.duration_s * self.config.volume_change_at

        if stream_time_s < change_start:
            return 1.0

        if not self.config.gradual_transition:
            return self.config.volume_multiplier

        transition_end = change_start + (self.config.duration_s * self.config.transition_duration)

        if stream_time_s >= transition_end:
            return self.config.volume_multiplier

        # Linear interpolation
        progress = (stream_time_s - change_start) / (transition_end - change_start)
        return 1.0 + progress * (self.config.volume_multiplier - 1.0)

    def generate(self) -> Iterator[LabelledRequest]:
        # We need to generate with variable rate
        # Use a time-stepping approach

        stream_time_s = 0.0
        request_idx = 0
        base_interval = 1.0 / self.config.requests_per_second

        while stream_time_s < self.config.duration_s:
            multiplier = self._get_volume_multiplier(stream_time_s)
            current_interval = base_interval / multiplier

            is_attack = self._is_attack_request(request_idx, stream_time_s)
            yield self._generate_request(request_idx, stream_time_s, is_attack)

            request_idx += 1
            stream_time_s += current_interval

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "scenario_type": "volume_change",
            "duration_s": self.config.duration_s,
            "volume_change_at": self.config.volume_change_at,
            "volume_multiplier": self.config.volume_multiplier,
            "gradual_transition": self.config.gradual_transition,
            "transition_duration": self.config.transition_duration,
            "attack_rate": self.config.attack_rate,
            "seed": self.config.seed,
        }
