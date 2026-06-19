# Forecasting Examples

The committed forecasting examples show how to run taxi-domain forecasting
experiments without downloading data or calling private services. Treat them as
workflow references: they demonstrate how to declare a panel, fit a native
forecasting wrapper, evaluate held-out taxi demand, save plots or artifacts
under `target/`, and fail clearly when a required native surface is unavailable.

The shared fixture is `examples/forecasting/forecast_cli_input.csv`. Examples
that fit `naive`, `seasonal_naive`, `theta`, `optimized_theta`, `ets`,
`auto_arima`, or `cartoboost_lag` call Rust-native forecasting bindings through
`cartoboost._native`. Python must not run fallback forecasting algorithms.

## How To Read The Examples

Use an example when it matches the scientific question:

- baseline behavior: start with naive, seasonal naive, theta, or ETS examples to
  establish a defensible comparator before adding more complex features;
- panel evidence: use panel examples when the question is pickup-zone or
  pickup/dropoff-lane demand rather than a single aggregate series;
- feature-rich forecasting: use lag examples when recent demand, rolling means,
  trend features, and calendar features should be part of the experiment;
- interval or probabilistic output: use interval examples when the decision
  needs uncertainty evidence, not only point forecasts;
- CLI reproducibility: use CLI examples when a command should be copied into a
  benchmark log, report, or CI job.

Keep taxi terminology in examples and reports: pickup/dropoff zones,
`PULocationID`, `DOLocationID`, taxi trips, fare, duration, trip distance, and
hour/day features.

## Example Map

| File | Coverage |
| --- | --- |
| `naive_seasonal_visualization.py` | Rust-backed `NaiveForecaster` and `SeasonalNaiveForecaster` on hourly taxi-zone pickup demand, with holdout metrics and a comparison plot under `target/examples/`. |
| `theta_optimized_visualization.py` | Rust-backed `ThetaForecaster` and `OptimizedThetaForecaster` with an explicit theta/alpha grid over taxi-zone demand, plus held-out metrics and a plot. |
| `ets_component_visualization.py` | Rust-backed `ETSForecaster` with fitted values, residuals, level/trend paths, seasonal components, held-out metrics, and a component plot. |
| `arima_example_visualization.py` | Python `ArimaForecaster` and `AutoARIMAForecaster` on a deterministic taxi lane panel, with held-out metrics and optional Matplotlib plot output under `target/`. |
| `cartoboost_lag_visualization.py` | Python `CartoBoostLagForecaster` wrapper with lag, rolling, calendar, trend, recursive forecasting, held-out metrics, and a residual plot. |
| `weighted_ensemble_visualization.py` | Rust-backed `WeightedEnsembleForecaster` combining seasonal naive, theta, and Kalman components for taxi-lane pickup demand. |
| `kriging_example_visualization.py` | Rust-backed kriging variogram fitting, interpolation surface, leave-one-out diagnostics, and committed documentation assets for example pickup-zone geometry. |
| `single_series_theta.py` | CLI `fit` shape for `theta` over taxi pickup demand. |
| `panel_forecasting.py` | CLI `fit` shape for panel `seasonal_naive` over `PULocationID`. |
| `rolling_origin_backtest.py` | CLI `backtest` shape and current clear failure when the Rust backtest binding is unavailable. |
| `probabilistic_intervals.py` | Python `ForecastResult` and `PredictionInterval` output columns for interval forecasts. |
| `kalman_diagnostics_visualization.py` | Rust-backed Kalman utility diagnostics with filtered/smoothed states, forecast intervals, standardized innovations, and a saved Matplotlib plot under `target/examples/`. |
| `carto_boost_lag_forecaster.py` | Python `CartoBoostLagForecaster` wrapper with lag, rolling, calendar, static, and known-future features. |

## Evidence To Preserve

When adapting an example into a benchmark or report, preserve:

- exact command or script path;
- data source and whether it is a fixture, generated acceptance data, or real
  taxi benchmark data;
- sample size, target, horizon, frequency, panel definition, and split name;
- model settings and feature configuration;
- RMSE, MAE, R2 when applicable, training time, prediction time, and interval
  metrics when intervals are produced;
- output artifact or plot path, if committed as evidence.

Generated outputs should go under `target/` or `/tmp` unless they are
intentionally committed evidence. Benchmark refreshes should update the
maintained report narrative in the same change so readers understand the
command, data source, metrics, winner, and why the result is structurally
meaningful.

## Forecasting V1 Names

The Forecasting V1 model-name surface includes `local_level_kalman`,
`local_linear_trend_kalman`, `unobserved_components`, `sarimax`,
`dynamic_regression`, `croston`, `sba`, `tsb`, `mstl_ets`, `stl_arima`,
`quantile_carto_boost_lag`, `conformal_forecaster`, `bottom_up_reconciler`,
`min_trace_reconciler`, and `foundation_model_adapter_optional`.

These names are accepted by the CLI so config and compare command shapes can be
validated early. Models that need additional parameters, hierarchy metadata,
calibration data, or optional adapters fail explicitly until their Rust/Python
CLI wrapper is available.

Weighted ensembles, conformal wrappers, quantile lag forecasting, hierarchical
reconciliation, and foundation-model adapters require explicit component models,
calibration data, hierarchy definitions, or optional adapter configuration. Keep
those examples in Python until the corresponding CLI options can represent the
full model contract without silent defaults.
