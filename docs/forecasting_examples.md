# Forecasting Examples

The committed forecasting examples use the taxi pickup-demand fixture
`examples/forecasting/forecast_cli_input.csv` and do not download data or call
private services.

Examples that fit `naive`, `seasonal_naive`, `theta`, `optimized_theta`, `ets`,
`auto_arima`, or `cartoboost_lag` call Rust-native forecasting bindings through
`cartoboost._native`. Python must not run fallback forecasting algorithms.

Use the examples as command-shape references for taxi pickup/dropoff demand,
taxi-zone panels, CLI runs, and wrapper APIs. Weighted ensembles require
explicit component models.
