# Forecasting Examples

The committed forecasting examples use the taxi pickup-demand fixture
`examples/forecasting/forecast_cli_input.csv` and do not download data or call
private services.

Examples that fit `naive`, `seasonal_naive`, `theta`, `optimized_theta`, `ets`,
`auto_arima`, or `cartoboost_lag` call Rust-native forecasting bindings through
`cartoboost._native`. Python must not run fallback forecasting algorithms.

Use the examples as command-shape references for taxi pickup/dropoff demand,
taxi-zone panels, CLI runs, and wrapper APIs:

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
