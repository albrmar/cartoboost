# CLI Reference

## `train`

```sh
cartoboost train --data <csv> [--config <toml>] [--model-out <path>] [--output json|csv]
```

Trains a dense numeric CSV model. If `--model-out` is omitted, the CLI writes
`cartoboost-model.json`.

JSON output:

```json
{"ok": true, "command": "train", "rows": 100, "features": 8, "model_path": "model.json", "trees": 100}
```

## `predict`

```sh
cartoboost predict --model <path> --input <csv> [--predictions-out <path>] [--output json|csv]
```

Loads a model and predicts dense numeric rows. Prediction CSV output uses:

```csv
row,prediction
0,1.25
1,2.5
```

## `eval`

```sh
cartoboost eval --model <path> --data <csv> [--output json|csv]
```

Computes mean absolute error against the target column stored in the model, or
the last data column when the model has no target name.

## `inspect`

```sh
cartoboost inspect [--model <path>] [--config <toml>] [--data <csv>] [--output json|csv]
```

Summarizes model, config, and data inputs without training.

## Accepted Options

| Command | Options |
| --- | --- |
| `train` | `--data`, `--config`, `--model-out`, `--output`, `--help` |
| `predict` | `--model`, `--input`, `--predictions-out`, `--output`, `--help` |
| `eval` | `--model`, `--data`, `--output`, `--help` |
| `inspect` | `--model`, `--config`, `--data`, `--output`, `--help` |

Unknown options fail fast.

## Forecasting Script

Forecasting V1 is exposed through `scripts/forecast.py`:

```sh
python scripts/forecast.py fit \
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

Commands:

| Command | Purpose |
| --- | --- |
| `fit` | Reads CSV history, writes `model.json`, `resolved_config.json`, and optional forecast CSV. |
| `predict` | Reads a saved forecast artifact directory and writes a forecast CSV. |
| `backtest` | Runs a deterministic final-window backtest and writes JSON metrics. |
| `compare` | Scores one or more forecasting models on the same holdout. |

Forecasting options:

| Option | Notes |
| --- | --- |
| `--input` | CSV history. Required for `fit`, `backtest`, and `compare`. |
| `--timestamp-col` | Timestamp column such as `timestamp` or `pickup_hour`. |
| `--target-col` | Target column such as `pickup_demand`, `fare`, or `duration`. |
| `--series-id-col` | Optional panel id such as `PULocationID` or `lane_id`. |
| `--freq` | Frequency: `D`, `H`, `W`, or `M`. |
| `--model` | `naive`, `seasonal_naive`, `theta`, `optimized_theta`, `ets`, `auto_arima`, `cartoboost_lag`, or `all` for `compare`. |
| `--horizon` | Positive forecast horizon. |
| `--season-length` | Seasonal cycle for seasonal naive and theta-style models. |
| `--output` | Forecast CSV or JSON metrics path. |
| `--artifact-dir` | Directory for model/config/metrics artifacts. |
| `--config` | JSON or simple TOML-style config file; CLI flags override file values. |

Forecast CSVs include `series_id`, `timestamp`, `model`, `horizon`,
`forecast`, `lower_80`, and `upper_80`. Invalid configs, missing columns,
unknown model names, and missing artifact directories exit nonzero with a
message on stderr.
