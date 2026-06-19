# Forecasting Artifacts

Forecasting artifacts turn a model run into reviewable evidence. A forecast CSV
alone is not enough to reproduce a scientific claim; reviewers also need the
model name, horizon, frequency, panel columns, feature configuration, interval
settings, and backtest metrics. CartoBoost writes those details into a manifest
beside the forecast rows.

Forecasting V1 writes portable artifacts as a directory with two files:

- `manifest.json`: model, schema, feature, interval, ensemble, reconciliation,
  and backtest metadata;
- `forecast.csv` by default, or `forecast.parquet` when saved with Parquet
  support.

The artifact is designed to avoid hidden process state. A loaded artifact
contains forecast rows and the manifest only; it does not restore live Python
model objects, closures, notebooks, or local process state.

## Manifest Contract

`ForecastArtifactManifest` records:

- `model_name`, for example `weighted_ensemble`, `seasonal_naive`, or
  `cartoboost_lag`;
- `horizon` and optional `freq`;
- `columns`, `forecast_path`, and `forecast_format`;
- optional `target_column`, `time_column`, and `panel_columns`;
- optional `lower_bound` and `upper_bound` used to clip forecasts and intervals;
- `feature_config`, such as taxi demand lag settings for `PULocationID`;
- `params`, `backtest_metrics`, `interval_metadata`, `ensemble_metadata`,
  `reconciliation_metadata`, and free-form `metadata`.

For a taxi demand forecast, rows commonly include `PULocationID`, `pickup_hour`,
`step`, `mean`, `lower`, and `upper`. The manifest `columns` field is
authoritative: every saved row must contain those columns.

When comparing runs, inspect the manifest first. If two artifacts differ in
panel columns, horizon, feature configuration, bounds, or validation metadata,
they are different experiments even if their forecast tables have the same
shape.

## Formats

CSV uses only the Python standard library and is always available. Parquet
requires the optional `pyarrow` package and hard-fails with an install hint when
it is missing. Use CSV for portable smoke tests and Parquet for larger taxi
forecast tables when the dependency is installed.

Optional integrations must stay optional. Artifact code should not silently
degrade to a different format or omit metadata when an optional package is
missing.

## Registry And Config

`ForecastRegistry.defaults()` registers these Forecasting V1 model names:

- `naive`
- `seasonal_naive`
- `theta`
- `optimized_theta`
- `ets`
- `auto_arima`
- `cartoboost_lag`
- `local_level_kalman`
- `local_linear_trend_kalman`
- `unobserved_components`
- `sarimax`
- `dynamic_regression`
- `croston`
- `sba`
- `tsb`
- `mstl_ets`
- `stl_arima`
- `quantile_carto_boost_lag`
- `conformal_forecaster`
- `bottom_up_reconciler`
- `min_trace_reconciler`
- `foundation_model_adapter_optional`

Forecasting model wrappers validate constructor parameters in Python where that
does not require model execution. Fitting and prediction are delegated to Rust
bindings for `naive`, `seasonal_naive`, `theta`, `optimized_theta`, `ets`,
`auto_arima`, `cartoboost_lag`, and the registry-only Forecasting V1 names
listed above.

`WeightedEnsembleForecaster`, `BottomUpReconciler`, and `MinTraceReconciler`
are available as direct Python classes when explicit native component models or
hierarchy metadata are supplied. Unsupported modes fail explicitly instead of
running Python fallback forecasting algorithms. Duplicate registry entries are
rejected unless `override=True` is passed.

`ForecastingConfig` parses TOML strictly. Unknown root or model fields raise by
default. Set `allow_unknown = true` to retain unknown fields under
manifest/config metadata instead of rejecting the file.

Hierarchical reconciliation settings can be carried in a `[reconciliation]`
table. Supported methods are `bottom_up_reconciler` and `min_trace_reconciler`,
with short aliases `bottom_up` and `min_trace`. The table accepts hierarchy
metadata such as `hierarchy`, `summing_matrix`, `series_id_column`,
`parent_column`, `child_column`, `non_negative`, and MinT covariance settings.
The Python config layer validates and passes these settings to native
reconcilers; it does not perform reconciliation itself.
