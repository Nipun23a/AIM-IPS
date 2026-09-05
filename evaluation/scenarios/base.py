"""
Base classes for scenario generation.

Scenarios generate streams of LabelledRequest objects with pre-computed
classifier scores. The classifier is NEVER modified during evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Dict, Any, Optional, Callable
import numpy as np

from ..harness import LabelledRequest


@dataclass
class ScenarioConfig:
    """
    Base configuration for all scenarios.

    Shared parameters across scenario types.
    """
    # Duration and timing
    duration_s: float = 3600.0           # Total scenario duration (1 hour default)
    requests_per_second: float = 10.0    # Base request rate

    # Attack interleaving (fixed rate)
    attack_rate: float = 0.05            # Fraction of requests that are attacks
    attack_distribution: str = "uniform" # "uniform", "bursty", "periodic"

    # Score distributions (for synthetic data)
    benign_score_mean: float = 0.15      # Mean s_app for benign traffic
    benign_score_std: float = 0.08       # Std dev for benign scores
    attack_score_mean: float = 0.75      # Mean s_app for attack traffic
    attack_score_std: float = 0.15       # Std dev for attack scores

    # Random seed for reproducibility
    seed: int = 42

    # Endpoint distribution (default)
    endpoints: List[str] = field(default_factory=lambda: [
        "/api/users/{id}",
        "/api/orders/{id}",
        "/api/products",
        "/api/search",
        "/api/auth/login",
    ])
    endpoint_weights: Optional[List[float]] = None  # None = uniform

    methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    method_weights: Optional[List[float]] = None


class ScenarioGenerator(ABC):
    """
    Abstract base class for scenario generators.

    Generates a stream of LabelledRequest objects with pre-computed
    s_app scores. The classifier is frozen during evaluation.
    """

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    @abstractmethod
    def generate(self) -> Iterator[LabelledRequest]:
        """Generate stream of labelled requests."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return scenario metadata for logging."""
        pass

    def _sample_endpoint(self) -> str:
        """Sample an endpoint from the distribution."""
        weights = self.config.endpoint_weights
        if weights is None:
            weights = [1.0 / len(self.config.endpoints)] * len(self.config.endpoints)
        return self.rng.choice(self.config.endpoints, p=weights)

    def _sample_method(self) -> str:
        """Sample an HTTP method from the distribution."""
        weights = self.config.method_weights
        if weights is None:
            weights = [1.0 / len(self.config.methods)] * len(self.config.methods)
        return self.rng.choice(self.config.methods, p=weights)

    def _sample_benign_score(self, mean: Optional[float] = None, std: Optional[float] = None) -> float:
        """Sample a benign traffic score."""
        mean = mean if mean is not None else self.config.benign_score_mean
        std = std if std is not None else self.config.benign_score_std
        score = self.rng.normal(mean, std)
        return float(np.clip(score, 0.0, 1.0))

    def _sample_attack_score(self, mean: Optional[float] = None, std: Optional[float] = None) -> float:
        """Sample an attack traffic score."""
        mean = mean if mean is not None else self.config.attack_score_mean
        std = std if std is not None else self.config.attack_score_std
        score = self.rng.normal(mean, std)
        return float(np.clip(score, 0.0, 1.0))

    def _is_attack_request(self, request_idx: int, stream_time_s: float) -> bool:
        """Determine if this request should be an attack."""
        if self.config.attack_distribution == "uniform":
            return self.rng.random() < self.config.attack_rate
        elif self.config.attack_distribution == "bursty":
            # Attacks come in bursts
            burst_period = 300.0  # 5-minute cycle
            burst_duration = 30.0  # 30-second burst
            in_burst = (stream_time_s % burst_period) < burst_duration
            if in_burst:
                return self.rng.random() < (self.config.attack_rate * 5)
            return self.rng.random() < (self.config.attack_rate * 0.2)
        elif self.config.attack_distribution == "periodic":
            # Fixed attack every N requests
            period = int(1.0 / self.config.attack_rate) if self.config.attack_rate > 0 else 1000
            return request_idx % period == 0
        else:
            return self.rng.random() < self.config.attack_rate

    def _generate_request(
        self,
        request_idx: int,
        stream_time_s: float,
        is_attack: bool,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        score_override: Optional[float] = None,
    ) -> LabelledRequest:
        """Generate a single labelled request."""
        endpoint = endpoint or self._sample_endpoint()
        method = method or self._sample_method()

        if score_override is not None:
            s_app = score_override
        elif is_attack:
            s_app = self._sample_attack_score()
        else:
            s_app = self._sample_benign_score()

        return LabelledRequest(
            request_idx=request_idx,
            stream_time_s=stream_time_s,
            s_app=s_app,
            is_attack=is_attack,
            path=endpoint,
            method=method,
        )


def load_real_dataset(
    path: str,
    score_column: str = "s_app",
    label_column: str = "label",
    path_column: str = "path",
    method_column: str = "method",
) -> Iterator[LabelledRequest]:
    """
    Load a real dataset with pre-computed classifier scores.

    Expected format: CSV or parquet with columns for score, label, path, method.
    Time is assigned sequentially based on row order.
    """
    import pandas as pd

    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)

    for idx, row in df.iterrows():
        yield LabelledRequest(
            request_idx=int(idx),
            stream_time_s=float(idx),  # 1 request per second by default
            s_app=float(row[score_column]),
            is_attack=bool(row[label_column]),
            path=str(row[path_column]),
            method=str(row[method_column]) if method_column in row else "GET",
        )
