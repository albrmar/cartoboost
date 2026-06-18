# Forecasting CLI

The Forecasting V1 CLI is available through:

```bash
PYTHONPATH=python uv run --group dev python -m cartoboost.forecasting.cli fit \
  --input examples/forecasting/forecast_cli_input.csv \
  --timestamp-col timestamp \
  --target-col pickup_demand \
  --series-id-col PULocationID \
  --freq D \
  --model theta \
  --horizon 7 \
  --season-length 7 \
  --artifact-dir target/forecasting/theta
```

Commands are `fit`, `predict`, `backtest`, and `compare`. The script accepts
`--input`, `--timestamp-col`, `--target-col`, `--series-id-col`, `--freq`,
`--model`, `--horizon`, `--season-length`, `--output`, `--artifact-dir`, and
`--config`.

Available model names are `naive`, `seasonal_naive`, `theta`, `optimized_theta`,
`ets`, `auto_arima`, `cartoboost_lag`, and `weighted_ensemble`.

The CLI does not run Python fallback forecasters. It validates configuration and
input shape, then delegates to the Python wrapper for the selected Rust native
model. If the corresponding `cartoboost._native` binding is not present, the
command exits nonzero and prints a `Rust binding ... is not available` error.

`predict`, `backtest`, and `compare` also require Rust-side forecasting artifact,
backtest, and comparison bindings. Until those bindings are exposed, these
commands fail clearly rather than writing synthetic forecast CSVs or metrics.
