# Evaluation Protocol Helpers

GeoBoost exposes lightweight Python helpers for regression diagnostics and
blocked validation. They are independent of the estimator and operate on arrays,
so they can be used with GeoBoost, XGBoost, LightGBM, scikit-learn, or any
external model predictions.

For temporal-spatial data, random validation is often too easy: nearby rows,
future rows, or rows from the same route can leak signal across the split. Use
blocked evaluation when the model will face new time windows, new locations, or
new route groups.

## Metrics

```python
from geoboost import (
    calibrated_intervals,
    conformal_residual_quantile,
    interval_coverage,
    jitter_volatility,
    mean_interval_width,
    pinball_loss,
    residual_morans_i,
)
```

- `conformal_residual_quantile(y_true, y_pred, alpha=0.1)` computes the
  finite-sample absolute residual quantile for split-conformal calibration.
- `calibrated_intervals(y_pred, residual_quantile=...)` returns symmetric
  lower and upper prediction intervals.
- `pinball_loss(...)`, `interval_coverage(...)`, and
  `mean_interval_width(...)` summarize quantile and interval quality.
- `jitter_volatility(predictions)` reports mean per-sample instability across
  repeated jittered prediction runs.
- `residual_morans_i(coordinates, residuals, weights=...)` reports spatial
  autocorrelation with inverse-distance or fixed-radius weights.

Use `residual_morans_i` after fitting to check whether location still explains
the errors. Strong residual autocorrelation is a sign that the model is missing
spatial structure or that the holdout needs a stronger spatial split.

## Blocked CV

```python
from geoboost import (
    grouped_blocked_cv,
    out_of_time_split,
    spatial_blocked_cv,
    temporal_blocked_cv,
)
```

The blocked CV helpers yield `(train_idx, test_idx)` NumPy index arrays.
`out_of_time_split` returns one pair directly.

- `out_of_time_split(times, validation_fraction=0.2)` returns one
  `(train_idx, validation_idx)` pair with validation rows taken from the latest
  time window. Use `validation_size=...` for an exact tail count or `cutoff=...`
  for rows strictly after a time boundary.
- `spatial_blocked_cv(coordinates, n_splits=5)` holds out spatial grid blocks.
- `temporal_blocked_cv(times, n_splits=5, gap=0)` holds out contiguous
  time-ordered blocks and can remove adjacent training rows with `gap`.
- `grouped_blocked_cv(groups, n_splits=5)` keeps all rows from a group in the
  same fold to avoid group leakage.

## Out-Of-Time Validation

Use out-of-time validation when the model will score rows from a later period
than the training data. This is the right first check for demand forecasting,
ETA residuals, pricing adjustments, staffing forecasts, and other temporal or
temporal-spatial targets.

```python
from geoboost import GeoBoostRegressor, out_of_time_split

train_idx, validation_idx = out_of_time_split(
    pickup_times,
    validation_fraction=0.2,
)

model = GeoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    splitters=["axis", "periodic:24", "diagonal_2d"],
)

model.fit(X[train_idx], y[train_idx])
prediction = model.predict(X[validation_idx])
```

By default, `out_of_time_split` sorts rows by `times` and uses the latest 20% as
validation. The returned arrays are original row indices, so they can index
NumPy arrays and pandas objects with positional indexing.

Choose the validation window by fraction, exact size, or cutoff:

```python
# Latest 20% of rows by time.
train_idx, validation_idx = out_of_time_split(times, validation_fraction=0.2)

# Latest 10,000 rows by time.
train_idx, validation_idx = out_of_time_split(
    times,
    validation_size=10_000,
    validation_fraction=None,
)

# Rows strictly after a time boundary.
train_idx, validation_idx = out_of_time_split(
    times,
    validation_fraction=None,
    cutoff="2025-01-01",
)
```

Use `gap` when adjacent time periods can leak information. A gap removes that
many sorted rows immediately before the validation window from training:

```python
train_idx, validation_idx = out_of_time_split(
    times,
    validation_fraction=0.2,
    gap=500,
)
```

For pandas inputs, use `.iloc` with the returned indices:

```python
model.fit(
    X_df.iloc[train_idx],
    y_series.iloc[train_idx],
)
prediction = model.predict(X_df.iloc[validation_idx])
```

For sparse route-cell features, slice the dense features and sparse lists with
the same indices:

```python
route_cells_train = [route_cells[i] for i in train_idx]
route_cells_validation = [route_cells[i] for i in validation_idx]

model.fit(
    X_dense[train_idx],
    y[train_idx],
    sparse_sets={"route_cells": route_cells_train},
)
prediction = model.predict(
    X_dense[validation_idx],
    sparse_sets={"route_cells": route_cells_validation},
)
```

Report out-of-time metrics separately from random or spatial holdouts. A model
can look good on a random split while failing on the latest period if demand,
traffic, pricing, weather, or market conditions drift.

## Recommended Comparisons

For temporal-spatial problems, report at least:

- Random holdout metrics.
- Spatial or grouped holdout metrics for withheld zones, cells, lanes, or
  routes.
- Out-of-time metrics for the latest period.
- Residual summaries by hour, zone, route cell, or lane.
- The same splits for GeoBoost and any XGBoost, LightGBM, or baseline model.
