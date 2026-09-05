"""
Data Loader for Real Datasets

Loads labelled datasets with pre-computed LightGBM s_app scores.
Supports CSV and parquet formats.

IMPORTANT: The classifier is FROZEN. We load pre-computed scores, never
run inference during evaluation. This ensures the replay harness measures
only threshold strategy performance.
"""

import csv
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict, Any
import numpy as np

from .harness import LabelledRequest

logger = logging.getLogger(__name__)


def load_scored_dataset(
    path: Path,
    score_column: str = "s_app",
    label_column: str = "label",
    path_column: str = "path",
    method_column: str = "method",
    time_column: Optional[str] = None,
    requests_per_second: float = 10.0,
    max_requests: Optional[int] = None,
) -> Iterator[LabelledRequest]:
    """
    Load a dataset with pre-computed classifier scores.

    The scores must be pre-computed by the frozen LightGBM classifier.
    This loader does NOT run inference.

    Args:
        path: Path to CSV or parquet file
        score_column: Column containing s_app score (1 - P(norm|x))
        label_column: Column containing ground-truth label (1=attack, 0=benign)
        path_column: Column containing request path
        method_column: Column containing HTTP method
        time_column: Optional column with timestamps (seconds)
        requests_per_second: Rate for synthetic time if time_column is None
        max_requests: Maximum number of requests to load (None = all)

    Yields:
        LabelledRequest objects
    """
    path = Path(path)

    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
    elif path.suffix in [".csv", ".gz"]:
        import pandas as pd
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"Loaded {len(df)} rows from {path}")

    # Validate required columns
    required = [score_column, label_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use defaults for optional columns
    if path_column not in df.columns:
        logger.warning(f"No '{path_column}' column, using '/api/default'")
        df[path_column] = "/api/default"

    if method_column not in df.columns:
        logger.warning(f"No '{method_column}' column, using 'GET'")
        df[method_column] = "GET"

    # Limit rows if requested
    if max_requests is not None and len(df) > max_requests:
        df = df.head(max_requests)

    # Generate stream
    interval = 1.0 / requests_per_second

    for idx, row in df.iterrows():
        # Compute stream time
        if time_column and time_column in df.columns:
            stream_time_s = float(row[time_column])
        else:
            stream_time_s = int(idx) * interval

        # Parse label
        label_val = row[label_column]
        if isinstance(label_val, str):
            is_attack = label_val.lower() not in ["benign", "normal", "norm", "clean", "0", "false"]
        else:
            is_attack = bool(label_val)

        yield LabelledRequest(
            request_idx=int(idx),
            stream_time_s=stream_time_s,
            s_app=float(row[score_column]),
            is_attack=is_attack,
            path=str(row[path_column]),
            method=str(row[method_column]),
        )


def split_train_validation_test(
    path: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    **kwargs,
) -> Tuple[List[LabelledRequest], List[LabelledRequest], List[LabelledRequest]]:
    """
    Split dataset into train/validation/test sets.

    Validation set is used for fitting theta_global.
    Test set is used for final evaluation (replay harness).

    Args:
        path: Path to dataset
        train_ratio: Fraction for training (unused in threshold eval)
        val_ratio: Fraction for validation threshold fitting
        seed: Random seed for reproducibility
        **kwargs: Passed to load_scored_dataset

    Returns:
        (train_list, validation_list, test_list)
    """
    # Load all data
    all_requests = list(load_scored_dataset(path, **kwargs))
    n = len(all_requests)

    # Shuffle with seed
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    # Split indices
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_list = [all_requests[i] for i in sorted(train_indices)]
    val_list = [all_requests[i] for i in sorted(val_indices)]
    test_list = [all_requests[i] for i in sorted(test_indices)]

    logger.info(f"Split: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

    return train_list, val_list, test_list


def fit_validation_threshold(
    validation_requests: List[LabelledRequest],
    metric: str = "f1",
) -> Tuple[float, Dict[str, Any]]:
    """
    Fit optimal threshold on validation set.

    This is called ONCE before evaluation. The threshold is NEVER
    updated during the replay stream.

    Args:
        validation_requests: Validation split (NOT test data)
        metric: Optimization target ("f1", "f2", "balanced_accuracy")

    Returns:
        (optimal_threshold, fitting_info)
    """
    from sklearn.metrics import f1_score, fbeta_score, balanced_accuracy_score, precision_score, recall_score

    scores = np.array([r.s_app for r in validation_requests])
    labels = np.array([int(r.is_attack) for r in validation_requests])

    best_threshold = 0.5
    best_score = 0.0
    all_results = []

    # Grid search
    for threshold in np.arange(0.05, 0.95, 0.01):
        preds = (scores >= threshold).astype(int)

        if metric == "f1":
            score_val = f1_score(labels, preds, zero_division=0)
        elif metric == "f2":
            score_val = fbeta_score(labels, preds, beta=2, zero_division=0)
        elif metric == "balanced_accuracy":
            score_val = balanced_accuracy_score(labels, preds)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)

        all_results.append({
            "threshold": threshold,
            "metric_value": score_val,
            "precision": precision,
            "recall": recall,
        })

        if score_val > best_score:
            best_score = score_val
            best_threshold = threshold

    fitting_info = {
        "metric": metric,
        "best_threshold": best_threshold,
        "best_score": best_score,
        "n_validation_samples": len(validation_requests),
        "n_attacks": int(labels.sum()),
        "n_benign": int(len(labels) - labels.sum()),
        "all_results": all_results,
    }

    logger.info(
        f"Fitted threshold: {best_threshold:.4f} ({metric}={best_score:.4f}) "
        f"on {len(validation_requests)} validation samples"
    )

    return best_threshold, fitting_info


def generate_endpoint_distribution(
    n_endpoints: int = 10,
    seed: int = 42,
) -> Tuple[List[str], List[float]]:
    """
    Generate realistic endpoint distribution.

    Returns endpoints following a power-law distribution
    (few endpoints get most traffic).
    """
    rng = np.random.default_rng(seed)

    # Common API endpoint patterns
    patterns = [
        "/api/users/{id}",
        "/api/orders/{id}",
        "/api/products",
        "/api/search",
        "/api/auth/login",
        "/api/auth/logout",
        "/api/cart/{id}",
        "/api/checkout",
        "/api/reviews/{id}",
        "/api/recommendations",
        "/api/notifications",
        "/api/settings",
        "/api/payments/{id}",
        "/api/shipping/{id}",
        "/api/inventory",
    ]

    endpoints = patterns[:n_endpoints]

    # Power-law weights (zipf distribution)
    weights = 1.0 / np.arange(1, n_endpoints + 1)
    weights = weights / weights.sum()

    return endpoints, weights.tolist()


def precompute_scores_from_classifier(
    input_path: Path,
    output_path: Path,
    classifier_path: Path,
    payload_column: str = "payload",
    path_column: str = "path",
    method_column: str = "method",
    label_column: str = "label",
) -> None:
    """
    Pre-compute s_app scores using the frozen LightGBM classifier.

    This should be run ONCE before evaluation to create the scored dataset.
    The classifier is NOT modified or retrained.

    Args:
        input_path: Raw dataset with payloads
        output_path: Output path for scored dataset
        classifier_path: Path to trained LightGBM model
        payload_column: Column containing request payload
        path_column: Column containing request path
        method_column: Column containing HTTP method
        label_column: Column containing ground-truth label
    """
    import pandas as pd
    import joblib
    import sys

    # Add project root to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from threat_classifier.lgbm_classifier import LGBMAppClassifier
    from shared.schemas import RequestContext

    # Load classifier
    classifier = LGBMAppClassifier(model_dir=classifier_path.parent)
    classifier.load()

    # Load raw data
    df = pd.read_csv(input_path)
    logger.info(f"Computing scores for {len(df)} samples...")

    scores = []
    for idx, row in df.iterrows():
        ctx = RequestContext(
            ip="0.0.0.0",
            method=str(row.get(method_column, "GET")),
            path=str(row.get(path_column, "/")),
            body=str(row.get(payload_column, "")),
        )

        try:
            layer_score = classifier.predict(ctx)
            s_app = layer_score.score
        except Exception as e:
            logger.warning(f"Error scoring row {idx}: {e}")
            s_app = 0.5  # Default on error

        scores.append(s_app)

        if (idx + 1) % 1000 == 0:
            logger.info(f"  Scored {idx + 1}/{len(df)}")

    df["s_app"] = scores

    # Save
    df.to_csv(output_path, index=False)
    logger.info(f"Scored dataset saved to {output_path}")
