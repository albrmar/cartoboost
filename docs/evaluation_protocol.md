# Evaluation Protocol Helpers

GeoBoost exposes lightweight Python helpers for regression diagnostics and
blocked validation protocols. They are independent of the estimator and operate
on arrays, so they can be used with native, fallback, or external model
predictions.

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

## Blocked CV

```python
from geoboost import grouped_blocked_cv, spatial_blocked_cv, temporal_blocked_cv
```

Each helper yields `(train_idx, test_idx)` NumPy index arrays.

- `spatial_blocked_cv(coordinates, n_splits=5)` holds out spatial grid blocks.
- `temporal_blocked_cv(times, n_splits=5, gap=0)` holds out contiguous
  time-ordered blocks and can remove adjacent training rows with `gap`.
- `grouped_blocked_cv(groups, n_splits=5)` keeps all rows from a group in the
  same fold to avoid group leakage.
