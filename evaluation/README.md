# Adaptive Threshold Evaluation Framework

Evaluation harness for IEEE paper: *Context-Aware Adaptive Thresholding for
Web-Application Intrusion Detection Classifiers*

## Quick Start

```bash
# Run all scenarios with default parameters
python -m evaluation.run_evaluation

# Run specific scenario
python -m evaluation.run_evaluation --scenario slow_poisoning

# Custom parameters
python -m evaluation.run_evaluation --theta-global 0.35 --delta 0.15 --duration 1800

# View help
python -m evaluation.run_evaluation --help
```

## Design Principles

1. **Frozen Classifier**: The LightGBM classifier is never modified. All strategies
   consume identical pre-computed `s_app` scores (1 - P(norm|x)).

2. **No Feedback Loops**: Benign confirmation uses ground-truth labels only, never
   the classifier's output. Labels are delayed by D=60s stream time.

3. **Microsecond Timing**: Decision latency is measured with `time.perf_counter_ns()`
   around `decide()` only, excluding I/O and classifier inference.

4. **Recall Floor**: All strategies report compliance with τ=0.95 recall floor.
   Configurations below floor are flagged as failing.

## Decision Strategies

### 1. FixedThresholdStrategy
```python
theta = 0.5  # Static threshold
decision = ATTACK if s_app >= theta else BENIGN
```

### 2. ValidationOptimisedStrategy
```python
theta_global = fit_on_validation_split()  # Fitted ONCE, never updated
decision = ATTACK if s_app >= theta_global else BENIGN
```

### 3. ContextAdaptiveStrategy
```python
# Per-context state
window[ctx] = deque(maxlen=500)  # Recent benign s_app values

# Threshold computation
if len(window[ctx]) < n_min:
    theta_ctx = theta_global  # Cold start
else:
    Q = quantile(window[ctx], 1 - alpha)  # alpha=0.02
    theta_ctx = clip(Q + margin, theta_global - delta, theta_global + delta)

decision = ATTACK if s_app >= theta_ctx else BENIGN

# After confirmation delay D=60s, if ground_truth=benign:
window[ctx].append(s_app)
```

## Scenarios

| Scenario | Description | Key Metrics |
|----------|-------------|-------------|
| `stationary` | Stable traffic, no shifts | Baseline F1, FPR |
| `new_endpoint` | New endpoint appears mid-stream | Cold start behavior |
| `payload_shift` | Encoding/length distribution changes | Adaptation speed |
| `volume_change` | Traffic volume spike (3x) | Stability under load |
| `slow_poisoning` | Adversarial threshold drift attack | Anti-poisoning effectiveness |

## Output Files

```
results/
└── 20260904_120000/
    ├── master_summary.json           # All scenarios
    ├── stationary/
    │   ├── per_request_*.csv         # Per-request decisions
    │   ├── per_window_metrics.csv    # Aggregated by window
    │   └── summary/
    │       ├── comparison_table.csv
    │       ├── comparison_table.tex  # LaTeX for paper
    │       └── threshold_stability.png
    ├── slow_poisoning/
    │   ├── poisoning_analysis.json   # Drift analysis
    │   └── ...
    └── ...
```

## CSV Columns

### per_request_*.csv
```
request_idx, stream_time_s, context_key, s_app, ground_truth_is_attack,
decision, threshold_used, decision_latency_us, window_size,
is_correct, is_tp, is_fp, is_tn, is_fn
```

### per_window_metrics.csv
```
window_idx, window_start_s, window_end_s, strategy, n_requests,
n_attacks, n_benign, tp, fp, tn, fn, precision, recall, f1, fpr, fnr,
mean_threshold, min_threshold, max_threshold, n_contexts,
mean_latency_us, p50_latency_us, p99_latency_us, max_latency_us
```

## Defensibility Checklist

Issues that would make results hard to defend:

| Issue | Status | Notes |
|-------|--------|-------|
| Unlabelled feedback loops | ✅ CLEAN | Ground-truth labels only |
| Thresholds fitted on test data | ✅ CLEAN | Validation split only |
| Timer includes I/O | ✅ CLEAN | `decide()` only |
| Classifier modified during eval | ✅ CLEAN | Pre-computed scores |
| Missing recall floor check | ✅ CLEAN | τ=0.95 enforced |
| Synthetic data only | ⚠️ | Use real dataset for final |

## Using Real Data

```python
from evaluation.data_loader import (
    load_scored_dataset,
    split_train_validation_test,
    fit_validation_threshold,
)

# Load pre-scored dataset
requests = list(load_scored_dataset(
    "data/cicids_scored.csv",
    score_column="s_app",
    label_column="label",
))

# Split for validation threshold fitting
train, val, test = split_train_validation_test("data/cicids_scored.csv")

# Fit theta_global on validation (ONCE, before evaluation)
theta_global, info = fit_validation_threshold(val, metric="f1")

# Create strategies with fitted threshold
strategies = create_strategies(theta_global=theta_global)

# Run harness on TEST split only
harness = ReplayHarness(strategies=strategies)
results = harness.run(iter(test))
```

## Pre-computing Scores

If you have raw payloads without scores:

```python
from evaluation.data_loader import precompute_scores_from_classifier

precompute_scores_from_classifier(
    input_path="data/raw_requests.csv",
    output_path="data/scored_requests.csv",
    classifier_path="models/application_layer/threat_classifier_lgb.pkl",
)
```

## Ablation Studies

### Anti-Poisoning Bound

```bash
# With delta=0.15 (bounded)
python -m evaluation.run_evaluation --delta 0.15 --scenario slow_poisoning

# Unbounded (included by default in all runs)
# Look for "ContextAdaptive(..., unbounded)" in results
```

### Confirmation Delay

Modify in code:
```python
harness = ReplayHarness(
    strategies=strategies,
    confirmation_delay_s=30.0,  # Try 30s, 60s, 120s
)
```

## Paper Figures

All figures must come from this harness. Example generation:

```python
from evaluation.summarize import (
    generate_comparison_table,
    generate_latex_table,
    generate_threshold_stability_plot,
    generate_recall_fpr_tradeoff_plot,
)

# Load results
results = [...]  # From harness output

# Generate LaTeX table for paper
generate_latex_table(results, Path("figures/table1.tex"))

# Generate threshold stability plot
generate_threshold_stability_plot(per_window_rows, Path("figures/threshold_drift.pdf"))
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta_global` | 0.35 | Validation-optimised global threshold |
| `alpha` | 0.02 | Target per-context FPR |
| `margin` | 0.02 | Safety margin above quantile |
| `delta` | 0.15 | Anti-poisoning bound |
| `n_min` | 50 | Cold start minimum samples |
| `window_size` | 500 | Benign window size per context |
| `D` | 60s | Label confirmation delay |
| `τ` | 0.95 | Recall floor |
