# Forecasting CLI

The Forecasting V1 CLI is the reproducible command-line surface for taxi
forecast experiments. Use it when you want the input path, timestamp column,
target column, panel column, model, horizon, frequency, and artifact directory
captured in a shell command that can be rerun by another reviewer.

The CLI validates configuration and input shape, then delegates to the Python
wrapper for the selected Rust native model. It does not run Python fallback
forecasters. Missing native bindings, unavailable zero-argument wrappers, and
missing optional dependencies fail clearly.

```bash
PYTHONPATH=python uv run python -m cartoboost.forecasting.cli fit \
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

## Choosing A CLI Workflow

Use `fit` for a single declared model and artifact. Use `backtest` when the
question is comparative evidence across rolling-origin folds. Use `compare`
when several accepted model names should be evaluated under the same command
shape. Use `predict` when reading an existing artifact or model state through
the supported native path.

For benchmark-style claims, keep the command stable and record the exact command
with RMSE, MAE, R2 when applicable, training time, prediction time, sample size,
task name, split name, and whether the data were synthetic, generated acceptance
data, or real taxi benchmark data.

## Accepted Model Names

| Model | CLI coverage |
| --- | --- |
| `naive` | `fit` can instantiate the zero-argument Rust wrapper. |
| `seasonal_naive` | `fit` can instantiate the Rust wrapper with `--season-length`. |
| `theta` | `fit` can instantiate the zero-argument Rust wrapper. |
| `optimized_theta` | `fit` can instantiate the zero-argument Rust wrapper. |
| `ets` | `fit` can instantiate the zero-argument Rust wrapper. |
| `auto_arima` | `fit` can instantiate the zero-argument Rust wrapper. |
| `cartoboost_lag` | `fit` can instantiate the zero-argument Rust wrapper. |
| `local_level_kalman` | Accepted V1 model name; zero-argument CLI wrapper is pending. |
| `local_linear_trend_kalman` | Accepted V1 model name; zero-argument CLI wrapper is pending. |
| `unobserved_components` | Accepted V1 model name; zero-argument CLI wrapper is pending. |
| `sarimax` | Accepted V1 model name; zero-argument CLI wrapper is pending. |
| `dynamic_regression` | Accepted V1 model name; zero-argument CLI wrapper is pending. |
| `croston` | Accepted V1 model name; zero-argument CLI wrapper is pending. |
| `sba` | Accepted V1 model name; zero-argument CLI wrapper is pending. |
| `tsb` | Accepted V1 model name; zero-argument CLI wrapper is pending. |
| `mstl_ets` | Accepted V1 model name; zero-argument CLI wrapper is pending. |
| `stl_arima` | Accepted V1 model name; zero-argument CLI wrapper is pending. |
| `quantile_carto_boost_lag` | Accepted V1 model name; quantile configuration is exposed through Python examples until the CLI wrapper lands. |
| `conformal_forecaster` | Accepted V1 model name; conformal calibration is exposed through Python examples until the CLI wrapper lands. |
| `bottom_up_reconciler` | Accepted V1 model name; reconciliation needs hierarchy metadata and is exposed through Python examples until the CLI wrapper lands. |
| `min_trace_reconciler` | Accepted V1 model name; reconciliation needs hierarchy metadata and is exposed through Python examples until the CLI wrapper lands. |
| `foundation_model_adapter_optional` | Accepted optional-adapter name; it must hard-fail unless the adapter is explicitly installed and configured. |

`weighted_ensemble` is available from Python when explicit component models are
provided, but it is not a zero-argument CLI model.

`compare --model all` expands over the full accepted model-name list. A
comma-separated list such as `--model theta,sarimax,conformal_forecaster` is
also valid for `compare`.

## Failure Semantics

If the corresponding `cartoboost._native` binding is not present, the command
exits nonzero and prints a `Rust binding ... is not available` error. If the
model name is accepted for Forecasting V1 but has no zero-argument CLI wrapper
yet, `fit` exits nonzero and reports that the CLI wrapper is unavailable.

`predict`, `backtest`, and `compare` also require Rust-side forecasting
artifact, backtest, and comparison bindings. If a required binding is absent,
these commands fail clearly rather than writing synthetic forecast CSVs or
metrics.
