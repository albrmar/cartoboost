# Forecasting Examples

The committed forecasting examples use the taxi pickup-demand fixture
`examples/forecasting/forecast_cli_input.csv` and do not download data or call
private services.

During the Rust migration, examples that fit or predict forecasting models are
expected to fail clearly until the corresponding `cartoboost._native`
forecasting bindings are available. That failure is intentional: Python must not
run fallback forecasting algorithms.

Use the examples as command-shape references for the CLI and wrapper APIs. Once
Rust bindings land, the same examples can become executable smoke tests again.
