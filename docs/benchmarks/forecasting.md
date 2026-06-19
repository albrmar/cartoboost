# Forecasting Tool Benchmark

## Bottom Line

The forecasting benchmark has five maintained views:

- Real NYC taxi lane demand: `cartoboost_lag` ties seasonal-naive baselines on
  the January 2024 24-lane, 7-day holdout.
- Synthetic taxi-shaped suite: `cartoboost_lag` ranks first in the maintained
  committed artifact, with mean RMSE ratio 1.021 to each problem winner.
- M4 24-series-per-group sample: `cartoboost_lag` ranks first by mean RMSE
  ratio in the refreshed committed sample, but this is still a sample, not a
  full M4 corpus claim.
- M5 full-run protocol: Kaggle M5 Accuracy files are now a first-class source
  for the full 30,490 item-store daily unit-sales panel with the official
  28-day holdout shape. The June 19, 2026 run completed over all 30,490
  bottom-level series using the public M5 mirror, a 90-day recent-history
  window, and a CartoBoost-only fast roster.
- M6 full-run protocol: M6 assets are now a first-class source for a daily
  return point-forecast proxy over the public M6 asset panel. The June 19, 2026
  run completed over 100 symbols and 38,219 daily-return rows.

Forecasting claims should stay split-specific. A seasonal baseline tie is a
valid result; it means the short panel did not prove residual lift. M5 and M6
are real competition panels, but the maintained M5 result below is a
CartoBoost-only full-corpus fast run, not a full external-library bakeoff.

## Real NYC Taxi Lane Demand

| Field | Value |
| --- | --- |
| Source | NYC TLC yellow taxi trips |
| Period | January 2024 |
| Raw rows | 2,964,624 |
| Clean rows | 2,827,685 |
| Aggregated rows | 744 |
| Series | 24 pickup/dropoff lanes |
| Days | 31 |
| Horizon | 7 days |
| Static covariates | pickup zone, dropoff zone, distance, airport-lane flag, pickup borough code |

Reproduce:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source nyc-taxi \
  --year 2024 \
  --months 1 \
  --taxi-type yellow \
  --lanes 24 \
  --horizon 7 \
  --no-download \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_plots
```

### Real Taxi Result

| Model | Library | RMSE | MAE | WAPE | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `cartoboost_lag` | CartoBoost | 39.034 | 29.173 | 0.0937 | Ties best seasonal baseline. |
| `functime_snaive` | functime | 39.034 | 29.173 | 0.0937 | Tied best. |
| `statsforecast_seasonal_naive` | StatsForecast | 39.034 | 29.173 | 0.0937 | Tied best. |
| `statsforecast_autoarima` | StatsForecast | 56.986 | 42.836 | 0.1376 | Worse on this short panel. |
| `prophet_additive` | Prophet | 79.876 | 60.655 | 0.1950 | Worse on this short panel. |

The result says the short January panel is dominated by weekly persistence.
CartoBoost correctly falls back to the seasonal path rather than claiming an
unvalidated residual correction.

### Real Taxi Plots

![Forecasting metric comparison](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_tool_metric_comparison.png)

![Horizon RMSE by tool](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_horizon_rmse_by_tool.png)

![Forecast lines for top lanes](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_forecast_lines.png)

![Actual vs predicted demand](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_actual_vs_predicted.png)

## Synthetic Taxi-Shaped Suite

The synthetic suite uses four taxi-shaped forecasting problems: weekly lane
demand, airport calendar events, route-mix shifts, and borough monthly pulses.
It uses 12 series, 180 days, horizon 14, and 3 rolling-origin folds.

| Model | Mean RMSE ratio to problem best | Wins/ties | Top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_lag` | 1.021 | 2 | 3 |
| `statsforecast_autoarima` | 1.213 | 0 | 3 |
| `prophet_additive` | 1.220 | 2 | 4 |
| `statsforecast_autoets` | 1.538 | 0 | 1 |
| `statsforecast_autotheta` | 1.602 | 0 | 1 |
| `functime_snaive` | 1.774 | 0 | 0 |

This suite is a stress test and wiring check. It is not real TLC evidence.

## M4 Sample Suite

The maintained M4 artifact scores the first 24 series from each M4 group. It is
included to check non-taxi behavior and library interoperability.

| Model | Mean RMSE ratio to group best | Wins/ties | Top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_lag` | 1.027 | 5 | 6 |
| `xgboost_lag` | 1.185 | 3 | 1 |
| `lightgbm_lag` | 1.250 | 2 | 1 |
| `statsforecast_autoarima` | 1.291 | 2 | 1 |
| `statsforecast_autoets` | 1.328 | 2 | 2 |
| `statsforecast_dynamic_optimized_theta` | 1.344 | 2 | 1 |

This is not a full M4 claim. Use `--m4-series-limit 0` before making
full-corpus statements.

## M5 Full Competition Run

The M5 loader consumes Kaggle M5 Forecasting Accuracy files directly when they
exist locally. When downloads are allowed and those files are absent, it uses
Nixtla's public M5 mirror and records the mirror URL in the artifact. It
hard-fails under `--no-download` when the required files are missing.

| Field | Value |
| --- | --- |
| Source | Kaggle M5 Forecasting Accuracy files |
| Required files | `calendar.csv` and `sales_train_evaluation.csv` or `sales_train_validation.csv`; public mirror download is used when local files are absent and downloads are allowed |
| Series | Full bottom-level item-store corpus when `--m5-series-limit 0` is used |
| Split | Last 28 days from the supplied training/evaluation file |
| Horizon | 28 daily steps |
| Scoring | Shared harness: RMSE, MAE, MASE, WAPE, SMAPE, bias |
| Maintained artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |

Reproduce the maintained full-corpus fast run:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster cartoboost \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 0 \
  --m5-history-days 90 \
  --cartoboost-n-estimators 1 \
  --cartoboost-max-depth 3 \
  --cartoboost-min-samples-leaf 20 \
  --no-candidate-selection \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m5_plots
```

Use a positive `--m5-series-limit` only for smoke tests. Results from limited
runs must be labeled as samples, not as M5 full-corpus evidence.

### M5 Result

Command run on June 19, 2026:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster cartoboost \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 0 \
  --m5-history-days 90 \
  --cartoboost-n-estimators 1 \
  --cartoboost-max-depth 3 \
  --cartoboost-min-samples-leaf 20 \
  --no-candidate-selection \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m5_plots
```

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |
| Plots | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_plots/` |
| Source files | `data/forecasting_benchmarks/m5/datasets/sales_train_evaluation.csv`, `calendar.csv` |
| Mirror URL | `https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip` |
| Series | 30,490 item-store series |
| Rows | 2,744,100 daily unit-sales rows |
| Days materialized | 90 recent days from 1,941 available days |
| Horizon | 28 daily steps |
| Roster | `cartoboost` |
| Candidate selection | Disabled |
| CartoBoost settings | 1 estimator, learning rate 0.06, max depth 3, min samples leaf 20 |
| Total runtime | 21.941 seconds |
| CartoBoost RMSE | 2.634879 |
| CartoBoost MAE | 1.332997 |
| CartoBoost WAPE | 0.923884 |

The full external-library comparison was not used for M5 because recursive
external tree prediction and per-series heavyweight models are not practical on
30,490 series in the current benchmark harness. The maintained M5 artifact is
therefore a full-corpus CartoBoost fast-run check, not a claim that CartoBoost
beats the M5 leaderboard or external forecasting libraries.

## M6 Full Competition Proxy Run

The M6 loader consumes the public M6 methods `assets_m6.csv` file with
`symbol`, `date`, and `price` columns. If the file is missing, the runner fetches
it from the M6 methods repository unless `--no-download` is set.

| Field | Value |
| --- | --- |
| Source | M6 methods asset price panel |
| Required file | `assets_m6.csv` |
| Series | Every symbol in the assets file when `--m6-series-limit 0` is used |
| Transform | Calendar-daily adjusted-price panel with forward-filled non-trading days and daily returns as the target |
| Split | Last `--m6-horizon` calendar days from the daily return panel |
| Default horizon | 28 daily steps |
| Scoring | Shared library harness: RMSE, MAE, MASE, WAPE, SMAPE, bias |
| Maintained artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` |

Reproduce:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m6 \
  --m6-assets-path data/forecasting_benchmarks/m6/assets_m6.csv \
  --m6-series-limit 0 \
  --m6-horizon 28 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m6_plots
```

### M6 Result

Command run on June 19, 2026:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m6 \
  --m6-assets-path data/forecasting_benchmarks/m6/assets_m6.csv \
  --m6-series-limit 0 \
  --m6-horizon 28 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m6_plots
```

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` |
| Plots | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_plots/` |
| Source URL | `https://raw.githubusercontent.com/Mcompetitions/M6-methods/main/assets_m6.csv` |
| Series | 100 symbols |
| Rows | 38,219 daily-return rows |
| Days | 383 calendar days |
| Horizon | 28 calendar days |
| Total runtime | 192.666 seconds |
| Winner | `statsforecast_autoarima` |
| Best RMSE | 0.013402 |
| CartoBoost RMSE | 0.014348 |
| CartoBoost MAE | 0.009357 |
| CartoBoost WAPE | 1.271868 |
| CartoBoost RMSE ratio vs best | 1.070585 |

| Rank | Model | Read |
| ---: | --- | --- |
| 1 | `statsforecast_autoarima` | Best RMSE on this daily-return proxy. |
| 2 | `statsforecast_autoets` | Second by RMSE. |
| 3 | `functime_ridge` | Best non-StatsForecast model by RMSE. |
| 4 | `statsforecast_autotbats` | Strong but slower than simpler baselines. |
| 14 | `cartoboost_lag` | 7.1% higher RMSE than the best forecasting-library model. |

This is intentionally named a proxy run. The official M6 competition combined
rank-probability forecasts and investment decisions, with RPS and investment
return as official scoring dimensions. The CartoBoost forecasting-library
harness is a point-forecast comparison, so it is useful for interoperability and
daily return forecasting behavior, not for claiming official M6 leaderboard
equivalence.

## Limitations

- Interval coverage is only benchmark evidence when actually computed.
- Real taxi panel is short: 31 days and a 7-day holdout.
- Synthetic suite is diagnostic.
- M4 artifact is a 24-series-per-group sample.
- M5 full-run evidence is CartoBoost-only with a 90-day materialized history
  window and fast tree settings.
- M6 uses a daily point-forecast proxy, not the official RPS/investment-return
  competition scorer.
- External model availability depends on optional benchmark dependencies.
