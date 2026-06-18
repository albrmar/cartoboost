# Forecasting Lag Features

CartoBoost lag-based forecasting is Rust-owned. The public Python class
`CartoBoostLagForecaster` is a thin wrapper over
`cartoboost._native.CartoBoostLagForecaster`.

Python may still expose configuration objects such as `LagFeatureConfig`,
`RollingFeatureConfig`, and `CalendarFeatureConfig`, but supervised lag matrix
construction, recursive prediction, model fitting, and model prediction must be
owned by Rust.

The Python `LagFeatureBuilder` is available for inspection and preflight feature
audits. Its target-derived features are panel-isolated and use only rows whose
timestamp is strictly earlier than the feature row timestamp. Rows sharing the
same panel and timestamp do not feed each other's lag, rolling, or expanding
features.

`CartoBoostLagForecaster` delegates to the native Rust model. Python config
objects are converted only when they match the native surface:

- `LagFeatureConfig(lags=[...])` maps to native `lags`.
- `LagFeatureConfig(difference_lags=[...], rolling_trend_windows=[...])` is
  available in the Python inspection builder, and the native
  `CartoBoostLagForecaster(trend_features=True)` enables the corresponding
  leakage-safe lag-delta and rolling-trend features for forecasting.
- `RollingFeatureConfig` maps to native `rolling_windows` only for complete
  rolling means.
- `CalendarFeatureConfig` maps to native `calendar_features=True` for
  `dayofweek`, `month`, and `day`.
- `regressor_params` maps fixed CartoBoost booster settings such as
  `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `min_gain`,
  and `splitters` to the native model. Unsupported regressor parameters fail
  explicitly.

Unsupported Python-only feature options, such as hourly calendar features,
expanding summaries, non-mean rolling aggregations, static columns, known-future
covariates, and custom booster options outside the documented native surface,
should fail clearly instead of being silently ignored by the native wrapper.

Taxi-domain lag feature contracts should use columns such as `pickup_hour`,
`pickup_trips`, `PULocationID`, `DOLocationID`, pickup/dropoff lane identifiers,
known-future calendar or dispatch plans, and historical-only observed queue or
trip-distance features.
