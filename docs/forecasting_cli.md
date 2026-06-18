# Forecasting CLI

CartoBoost Forecasting V1 provides a lightweight command surface for deterministic
taxi time-series smoke workflows:

```bash
uv run --group dev python scripts/forecast.py fit \
  --input examples/forecasting/forecast_cli_input.csv \
  --timestamp-col timestamp \
  --target-col pickup_demand \
  --series-id-col PULocationID \
  --freq D \
  --model theta \
  --horizon 7 \
  --season-length 7 \
  --artifact-dir target/forecasting/theta \
  --output target/forecasting/theta_forecast.csv
```

Commands are `fit`, `predict`, `backtest`, and `compare`. The script accepts
`--input`, `--timestamp-col`, `--target-col`, `--series-id-col`, `--freq`,
`--model`, `--horizon`, `--season-length`, `--output`, `--artifact-dir`, and
`--config`.

Available Forecasting V1 model names are `naive`, `seasonal_naive`, `theta`,
`optimized_theta`, `ets`, `auto_arima`, `cartoboost_lag`, and
`weighted_ensemble`. Compatibility aliases `mean`, `drift`, and `cartoboost`
are also accepted by the script. `compare --model all` evaluates every V1 model
name and writes metrics ordered by RMSE.

`fit` writes `model.json` and `resolved_config.json` under `--artifact-dir`.
When `--output` is supplied it also writes a forecast CSV with `series_id`,
`timestamp`, `model`, `horizon`, `forecast`, `lower_80`, and `upper_80`.

`predict` reads `model.json` from `--artifact-dir` and writes the forecast CSV:

```bash
uv run --group dev python scripts/forecast.py predict \
  --artifact-dir target/forecasting/theta \
  --horizon 3 \
  --output target/forecasting/predictions.csv
```

`backtest` writes JSON metrics for the final holdout window:

```bash
uv run --group dev python scripts/forecast.py backtest \
  --input examples/forecasting/forecast_cli_input.csv \
  --timestamp-col timestamp \
  --target-col pickup_demand \
  --series-id-col PULocationID \
  --model theta \
  --horizon 3 \
  --output target/forecasting/backtest_metrics.json
```

`compare` writes JSON metrics for one or more models:

```bash
uv run --group dev python scripts/forecast.py compare \
  --input examples/forecasting/forecast_cli_input.csv \
  --timestamp-col timestamp \
  --target-col pickup_demand \
  --series-id-col PULocationID \
  --model all \
  --horizon 3 \
  --output target/forecasting/compare_metrics.json
```

Config files may be JSON or simple TOML-style `key = value` files. CLI options
override config file values. Invalid model names, missing columns, non-finite
targets, unknown config keys, and missing artifacts exit nonzero with an error
message on stderr.
