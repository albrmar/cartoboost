# Forecasting Artifacts

Forecasting V1 writes portable artifacts as a directory with two files:

- `manifest.json`: model, schema, feature, interval, ensemble, and backtest metadata.
- `forecast.csv` by default, or `forecast.parquet` when saved with Parquet support.

The artifact is designed to avoid hidden process state. A loaded artifact contains forecast rows
and the manifest only; it does not restore live Python model objects or local closures.

## Manifest Contract

`ForecastArtifactManifest` records:

- `model_name`: for example `weighted_ensemble`, `seasonal_naive`, or `cartoboost_lag`.
- `horizon` and optional `freq`.
- `columns`, `forecast_path`, and `forecast_format`.
- Optional `target_column`, `time_column`, and `panel_columns`.
- Optional `lower_bound` and `upper_bound` used to clip forecasts and intervals.
- `feature_config`, such as taxi demand lag settings for `PULocationID`.
- `params`, `backtest_metrics`, `interval_metadata`, `ensemble_metadata`, and free-form `metadata`.

For a taxi demand forecast, rows commonly include `PULocationID`, `pickup_hour`, `step`,
`mean`, `lower`, and `upper`. The manifest `columns` field is authoritative: every saved row must
contain those columns.

## Formats

CSV uses only the Python standard library and is always available. Parquet requires the optional
`pyarrow` package and hard-fails with an install hint when it is missing. Use CSV for portable
smoke tests and Parquet for larger forecast tables when the dependency is installed.

## Registry And Config

`ForecastRegistry.defaults()` registers these Forecasting V1 model names:

- `naive`
- `seasonal_naive`
- `theta`
- `optimized_theta`
- `ets`
- `auto_arima`
- `cartoboost_lag`

Forecasting model wrappers validate constructor parameters in Python where that
does not require model execution. Fitting and prediction are delegated to Rust
bindings for `naive`, `seasonal_naive`, `theta`, `optimized_theta`, `ets`,
`auto_arima`, and `cartoboost_lag`. `WeightedEnsembleForecaster` is available
as a direct Python class when explicit native component models are supplied.
Unsupported modes fail explicitly instead of running Python fallback forecasting
algorithms. Duplicate registry entries are rejected unless
`override=True` is passed.

`ForecastingConfig` parses TOML strictly. Unknown root or model fields raise by default. Set
`allow_unknown = true` to retain unknown fields under manifest/config metadata instead of rejecting
the file.
