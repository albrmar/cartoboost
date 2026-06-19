# Fair Benchmarking Program

CartoBoost benchmark claims must come from public task contracts, not from a
winner table assembled after a run. The benchmark program is organized into
four tracks:

| Track | Purpose | Required comparator types |
| --- | --- | --- |
| Tabular supervised | Public tabular regression and classification tasks where tree ensembles are serious baselines. | GBDT baseline and deep tabular baseline. |
| Spatial and repeated-ID | NYC taxi fare, duration, and demand tasks with IID, temporal, spatial, and cold-zone or cold-route splits. | GBDT with equivalent train-only spatial features and a deep or metric-learning baseline. |
| Graph structured | Public graph tasks with fixed splits and standardized evaluators. | Graph neural network baseline and tabularized graph-feature baseline. |
| Forecasting | Public forecasting repository data plus taxi lane-demand panels under rolling-origin evaluation. | Statistical forecasting baseline and global or neural forecasting baseline. |

The machine-readable contracts live under `benchmarks/tracks/`. Shared seed,
budget, hardware, and required-baseline rules live under `benchmarks/configs/`.
Report and dataset-card templates live under `benchmarks/reports/`.

## Claim Rules

A benchmark result can support a public quality claim only when it satisfies all
of these rules:

- The task is defined by a committed benchmark manifest with dataset, split,
  metric, required-baseline, and search-space entries.
- The test set is evaluated once per outer fold after model selection.
- Hyperparameter optimization uses train/validation data only.
- Every required model family receives the same trial budget, wall-clock
  ceiling, seed list, and early-stopping policy.
- Missing required baselines make the result incomplete, not a CartoBoost win.
- Repeated seeds or repeated outer splits are reported as distributions, not
  only as a single point estimate.
- Reports include confidence intervals or paired uncertainty estimates.
- Forecasting reports compute real interval coverage or omit interval claims.
  Placeholder coverage values are not acceptable benchmark evidence.
- Compute tables include hardware, thread count, fit time, prediction time, and
  peak memory.
- Dataset cards record source, extraction date, content hash, filters, target
  transforms, joins, dropped rows, and leakage checks.

Synthetic fixtures remain useful for mechanism diagnostics and regression
tests. They are not sufficient evidence for broad model superiority claims.

## Protocol

| Element | Rule |
| --- | --- |
| Train/validation/test | HPO uses train/validation only; final test is evaluated exactly once per outer fold. |
| IID tasks | Use repeated outer folds, with the public seed list in `benchmarks/configs/seeds.json`. |
| Spatial tasks | Use spatial or group-aware validation for both tuning and outer evaluation. |
| Cold-start tasks | Use group-aware outer splits and report cold-group prevalence. |
| Forecasting tasks | Use rolling-origin evaluation with fixed horizons and horizon-wise metrics. |
| Hyperparameter tuning | Use the same budget ID across model families in the same comparison. |
| Timing | Compare timings only with hardware metadata attached. |
| Baseline failures | Mark the benchmark incomplete unless the failure itself is the claim being audited. |

## Metrics

Regression tasks should report RMSE and MAE as primary point metrics, with R2
when it is meaningful. Taxi tasks must also report subgroup slices such as
borough, airport/non-airport, rare versus common zones, cold zones, cold routes,
and time-of-day or day-of-week buckets.

Classification and link-prediction tasks should report AUROC or average
precision plus probability-quality metrics such as log loss, Brier score, or
calibration error when probabilities are available. Graph tasks should slice by
node degree, train-edge density, and cold-node exposure.

Forecasting tasks should report MASE, RMSE, MAE, WAPE, sMAPE, and horizon-wise
metrics. Probabilistic or interval claims require real coverage, interval width,
interval score, CRPS, or another proper probabilistic score.

## Reporting

Each release benchmark report should include:

- Task definition.
- Dataset-card summary.
- Split protocol.
- Feature-access rules.
- HPO budget.
- Required baselines.
- Compute environment.
- Aggregate metrics with intervals.
- Subgroup slices.
- Significance tests.
- Failure logs.
- Limitations.

Use per-task delta plots, critical-difference diagrams, reliability diagrams,
horizon-wise forecast plots, subgroup heatmaps, and quality-versus-cost Pareto
charts when they help readers understand ranking stability.

## Commands

Validate benchmark manifests:

```sh
python -m benchmarks.runners.manifest
```

Aggregate a JSONL result file:

```sh
python -m benchmarks.runners.aggregate_results \
  --input artifacts/results/spatial/results.jsonl \
  --output artifacts/results/spatial/summary.json
```

The runner accepts rows shaped like this:

```json
{"task_id": "fare_spatial_cold_zone", "model_family": "cartoboost", "metric": "rmse", "value": 2.14}
```

This aggregation utility is intentionally small. Full release reports should
add paired tests over the registered folds and seeds before making winner
claims.

