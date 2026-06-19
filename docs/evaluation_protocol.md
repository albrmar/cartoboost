# Evaluation Protocol

Use this protocol when deciding whether CartoBoost is the right model for a
taxi-domain regression or forecasting problem. The point is not to make
CartoBoost win every table. The point is to measure whether its spatial,
temporal, sparse-set, graph, and residual-ID structure improves generalization
under the same evidence standard used for serious baselines.

CartoBoost is worth choosing when it gives a reproducible quality gain on the
deployment split that matters, or when it matches the best baseline while
providing useful structure such as route geometry, sparse zone membership,
native forecasting artifacts, or leakage-aware diagnostics. It is not worth
choosing solely because it wins a random split, a synthetic fixture, or a
benchmark where another model received weaker features.

## Evidence Standard

Every comparison should name:

- Dataset source, row count, target, unit of observation, and filtering rules.
- Feature families available to each model.
- Split design, including whether rows are random, out of time, spatially held
  out, route grouped, cold-ID, or rolling-origin forecast folds.
- Model settings for CartoBoost and baselines such as LightGBM, XGBoost,
  StatsForecast, Prophet, or simple seasonal methods.
- Metrics: RMSE and MAE for regression error, R2 for explained variance when
  appropriate, and WAPE or horizon metrics for demand forecasting.
- Training time and prediction time when throughput affects model choice.
- Exact command, seed, dependency group, and artifact paths.

If any of those are missing, treat the result as exploratory. Do not use it as a
quality claim in docs, releases, benchmark narratives, or model-selection
recommendations.

Public benchmark claims must use the manifest-driven program described in
[Fair Benchmarking Program](benchmarks/fair-benchmarking.md). That program adds
fixed public task manifests, required baseline families, equal HPO budgets,
repeated seeds or outer folds, confidence intervals, subgroup slices, and
compute metadata on top of the per-run evidence listed here.

## Baseline Fairness

Compare CartoBoost against models a skeptical scientist would actually deploy:
LightGBM, XGBoost, strong seasonal-naive baselines, StatsForecast, Prophet, or
domain-specific linear/ridge baselines when they are appropriate. A mean
baseline is useful for calibration but not sufficient evidence.

Give baselines the same training rows, validation rows, target transformation,
sample weights, and broadly comparable feature information. If CartoBoost sees
pickup zone, dropoff zone, hour, route distance, or target-mean zone features,
the serious baselines should receive equivalent encoded columns whenever their
interfaces can represent them. If a feature family is CartoBoost-specific, such
as sparse set membership or native graph-derived route features, say so and
evaluate whether the gain is large enough to justify the extra surface.

Keep random, spatial, grouped, and out-of-time results separate. A random split
can answer whether the model interpolates within a familiar distribution. It
does not answer whether the model survives new pickup zones, future periods, or
cold route IDs.

## Leakage Checks

Temporal-spatial taxi data leaks easily. Before trusting a score, check:

- Future rows are not used to compute lags, rolling features, target encodings,
  residual embeddings, graph features, or shrinkage factors.
- Pickup/dropoff zone statistics are fit on training rows only.
- Forecast horizons are evaluated against timestamps strictly after the
  training window.
- Grouped and spatial holdouts keep all rows for a held-out zone, route, lane,
  or ID out of training.
- Neural residual embeddings are evaluated on both repeated-ID and cold-ID
  splits.
- Graph topology is materialized from train-side edges only when the deployment
  problem will not know validation edges.

If a benchmark cannot enforce these conditions, label it as a wiring check or
acceptance fixture rather than evidence for production accuracy.

## Blocked Validation Helpers

```python
from cartoboost import (
    grouped_blocked_cv,
    out_of_time_split,
    spatial_blocked_cv,
    temporal_blocked_cv,
)
```

The blocked CV helpers yield `(train_idx, test_idx)` NumPy index arrays.
`out_of_time_split` returns one pair directly.

- `out_of_time_split(times, validation_fraction=0.2)` returns one tail
  validation window. Use `validation_size=...` for an exact count or
  `cutoff=...` for rows strictly after a timestamp.
- `spatial_blocked_cv(coordinates, n_splits=5)` holds out spatial grid blocks.
- `temporal_blocked_cv(times, n_splits=5, gap=0)` holds out contiguous time
  blocks and can remove adjacent training rows with `gap`.
- `grouped_blocked_cv(groups, n_splits=5)` keeps all rows from a group in the
  same fold.

Use these helpers for both CartoBoost and external baselines. The split object
is part of the evidence, not an implementation detail.

## Out-Of-Time Example

Use out-of-time validation when the model will score rows from a later period
than the training data. This is the right first check for pickup demand,
dropoff demand, ETA residuals, pricing adjustments, and route-level staffing
forecasts.

```python
from cartoboost import CartoBoostRegressor, out_of_time_split

train_idx, validation_idx = out_of_time_split(
    pickup_times,
    validation_fraction=0.2,
)

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    splitters=["axis", "periodic:24", "diagonal_2d"],
)

model.fit(X[train_idx], y[train_idx])
prediction = model.predict(X[validation_idx])
```

The returned arrays are original row indices, so they can index NumPy arrays
and pandas objects with positional indexing:

```python
model.fit(X_df.iloc[train_idx], y_series.iloc[train_idx])
prediction = model.predict(X_df.iloc[validation_idx])
```

Use `gap` when adjacent time periods can leak information:

```python
train_idx, validation_idx = out_of_time_split(
    pickup_times,
    validation_fraction=0.2,
    gap=500,
)
```

For sparse pickup/dropoff zone features, slice dense features and sparse lists
with the same indices:

```python
taxi_zones_train = [taxi_zones[i] for i in train_idx]
taxi_zones_validation = [taxi_zones[i] for i in validation_idx]

model.fit(
    X_dense[train_idx],
    y[train_idx],
    sparse_sets={"taxi_zones": taxi_zones_train},
)
prediction = model.predict(
    X_dense[validation_idx],
    sparse_sets={"taxi_zones": taxi_zones_validation},
)
```

## Diagnostic Metrics

```python
from cartoboost import (
    calibrated_intervals,
    conformal_residual_quantile,
    interval_coverage,
    jitter_volatility,
    mean_interval_width,
    pinball_loss,
    residual_morans_i,
)
```

- `conformal_residual_quantile` and `calibrated_intervals` support
  split-conformal interval calibration on held-out residuals.
- `pinball_loss`, `interval_coverage`, and `mean_interval_width` summarize
  quantile and interval quality.
- `jitter_volatility` reports mean per-sample instability across repeated
  jittered prediction runs.
- `residual_morans_i` checks whether coordinates still explain residuals.

Use `residual_morans_i` after fitting a taxi fare, duration, or demand model.
Strong spatial residual autocorrelation means the model is missing geographic
structure or the validation split does not match the deployment risk.

## Model Choice Rules

Choose plain CartoBoost when dense taxi features, cyclic time, route geometry,
and zone membership explain the gain without neural or graph additions. Choose
neural residual embeddings only when repeated pickup/dropoff zones, lanes, or
route IDs recur in production and the repeated-ID split improves without
degrading the cold-ID split you care about. Choose graph features when the
target is genuinely about pickup/dropoff topology, source-target flow, or lane
relationships.

Choose a simpler baseline when it ties CartoBoost on the deployment split and
is easier to operate. In short panels, a seasonal-naive forecast can be the
honest winner. That is useful evidence, not a failed benchmark.

## Reporting Checklist

For every maintained result, report:

- Random split metrics, when interpolation quality is relevant.
- Spatial or grouped holdout metrics for withheld zones, cells, routes, or
  lanes.
- Out-of-time or rolling-origin metrics for future periods.
- Residual summaries by hour, pickup zone, dropoff zone, lane, or borough.
- The same split and target transformation for CartoBoost, LightGBM, XGBoost,
  and forecasting baselines.
- Exact commands and artifact paths.

Update the benchmark writeup in the same change as any refreshed result so the
numbers, command, data source, and interpretation stay aligned.
