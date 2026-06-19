# Forecasting Tool Benchmark

## Competition Results Snapshot

| Competition | Artifact | Scope | Result |
| --- | --- | --- | --- |
| M5 Forecasting Accuracy comparison sample | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` | 100 item-store series, 90 recent days, 28-day holdout, full 14-model roster | `statsforecast_autoets` won RMSE at 2.525734; CartoBoost RMSE was 2.678097. |
| M5 Forecasting Accuracy full-corpus check | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` | 30,490 item-store series, 90 recent days, 28-day holdout, CartoBoost-only fast roster | CartoBoost RMSE 2.634879, MAE 1.332997, WAPE 0.923884. |
| M6 financial assets | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` | 100 symbols, 38,219 daily-return rows, 28-day holdout, full 14-model roster | `statsforecast_autoarima` won RMSE at 0.013402; CartoBoost RMSE was 0.014348. |

## Bottom Line

The forecasting benchmark has five maintained views:

- Real NYC taxi lane demand: `cartoboost_lag` ties seasonal-naive baselines on
  the January 2024 24-lane, 7-day holdout.
- Synthetic taxi-shaped suite: `cartoboost_lag` ranks first in the maintained
  committed artifact, with mean RMSE ratio 1.021 to each problem winner.
- M4 24-series-per-group sample: `cartoboost_lag` ranks first by mean RMSE
  ratio in the refreshed committed sample, but this is still a sample, not a
  full M4 corpus claim.
- M5 comparison sample: Kaggle M5 Accuracy files are now a first-class source
  with a full 14-model roster sample over 100 item-store daily unit-sales
  series, the official 28-day holdout shape, and the same model-family table
  style as the M4 sample.
- M5 full-corpus check: The June 19, 2026 run also completed over all 30,490
  bottom-level series using the public M5 mirror, a 90-day recent-history
  window, and a CartoBoost-only fast roster. That artifact is coverage and
  throughput evidence, not a model bakeoff.
- M6 full-run protocol: M6 assets are now a first-class source for a daily
  return point-forecast proxy over the public M6 asset panel. The June 19, 2026
  run completed over 100 symbols and 38,219 daily-return rows.

Forecasting claims should stay split-specific. A seasonal baseline tie is a
valid result; it means the short panel did not prove residual lift. M5 and M6
are real competition panels, but the M5 model comparison below is a bounded
100-series sample, not a full external-library bakeoff across all 30,490
bottom-level series.

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

### M4 Model RMSE

M4 groups have different scales, so the ratio table above is the better
cross-group comparison. The table below reports the arithmetic mean of each
model's group RMSE from the same committed artifact.

| Model | Mean RMSE | Mean RMSE ratio to group best |
| --- | ---: | ---: |
| `cartoboost_lag` | 557.361552 | 1.027257 |
| `xgboost_lag` | 630.136135 | 1.185291 |
| `lightgbm_lag` | 713.181000 | 1.250493 |
| `statsforecast_autoarima` | 781.134754 | 1.291109 |
| `statsforecast_autoets` | 830.310579 | 1.328236 |
| `statsforecast_dynamic_optimized_theta` | 837.602838 | 1.343737 |
| `statsforecast_autotheta` | 841.111899 | 1.349802 |
| `statsforecast_autotbats` | 864.728804 | 1.395882 |
| `statsforecast_autoces` | 872.457851 | 1.399044 |
| `functime_snaive` | 868.033528 | 1.400568 |
| `functime_lightgbm` | 868.033528 | 1.400568 |
| `statsforecast_seasonal_naive` | 868.033528 | 1.400568 |
| `functime_ridge` | 874.441228 | 1.414405 |
| `prophet_additive` | 913.373419 | 1.454984 |

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
| Maintained comparison artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` |
| Maintained full-corpus artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |

Reproduce the maintained full-roster comparison sample:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster full \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 100 \
  --m5-history-days 90 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_plots
```

Reproduce the maintained full-corpus CartoBoost fast run:

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

### M5 Full-Roster Sample Result

Command run on June 19, 2026:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster full \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 100 \
  --m5-history-days 90 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_plots
```

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` |
| Plots | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_plots/` |
| Source files | `data/forecasting_benchmarks/m5/datasets/sales_train_evaluation.csv`, `calendar.csv` |
| Mirror URL | `https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip` |
| Series | 100 item-store series from 30,490 available |
| Rows | 9,000 daily unit-sales rows |
| Days materialized | 90 recent days from 1,941 available days |
| Horizon | 28 daily steps |
| Roster | `full`: CartoBoost, functime, StatsForecast, Prophet, XGBoost lag, LightGBM lag |
| Candidate selection | Enabled, shared one-origin calibration |
| Total runtime | 93.734 seconds |
| Winner | `statsforecast_autoets` |
| Best RMSE | 2.525734 |
| CartoBoost RMSE | 2.678097 |
| CartoBoost MAE | 1.187821 |
| CartoBoost WAPE | 0.958196 |

### M5 Model RMSE

The table below reports every model present in the committed M5 full-roster
sample artifact, ranked by RMSE. It is the M5 counterpart to the M4 sample
table: same shared harness, same full model family, and explicit sample scope.

| Model | RMSE | MAE | WAPE |
| --- | ---: | ---: | ---: |
| `statsforecast_autoets` | 2.525734 | 1.141999 | 0.921232 |
| `statsforecast_dynamic_optimized_theta` | 2.556517 | 1.163750 | 0.938779 |
| `statsforecast_autotbats` | 2.602055 | 1.156588 | 0.933001 |
| `functime_ridge` | 2.606775 | 1.207878 | 0.974376 |
| `statsforecast_autotheta` | 2.607077 | 1.196042 | 0.964828 |
| `statsforecast_autoarima` | 2.655754 | 1.194312 | 0.963433 |
| `cartoboost_lag` | 2.678097 | 1.187821 | 0.958196 |
| `functime_snaive` | 2.678097 | 1.187821 | 0.958196 |
| `functime_lightgbm` | 2.678097 | 1.187821 | 0.958196 |
| `statsforecast_seasonal_naive` | 2.678097 | 1.187821 | 0.958196 |
| `statsforecast_autoces` | 2.678097 | 1.187821 | 0.958196 |
| `prophet_additive` | 2.678097 | 1.187821 | 0.958196 |
| `xgboost_lag` | 2.793477 | 1.500446 | 1.210386 |
| `lightgbm_lag` | 2.825295 | 1.253991 | 1.011575 |

This sample does not support a claim that CartoBoost beats the M5 model
families; it ranks seventh by RMSE. The all-series M5 artifact remains useful
as a CartoBoost full-corpus check:

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |
| Scope | 30,490 item-store series, 2,744,100 rows, 90 recent days, 28-day holdout |
| Roster | `cartoboost` |
| Candidate selection | Disabled |
| CartoBoost settings | 1 estimator, learning rate 0.06, max depth 3, min samples leaf 20 |
| Total runtime | 21.941 seconds |
| CartoBoost RMSE | 2.634879 |
| CartoBoost MAE | 1.332997 |
| CartoBoost WAPE | 0.923884 |

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
| Roster | `full`: CartoBoost, functime, StatsForecast, Prophet, XGBoost lag, LightGBM lag |
| Candidate selection | Enabled, shared one-origin calibration |
| Total runtime | 173.591 seconds |
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

### M6 Model RMSE

The table below reports every model present in the committed M6 artifact,
ranked by RMSE.

| Model | RMSE | MAE | WAPE |
| --- | ---: | ---: | ---: |
| `statsforecast_autoarima` | 0.013402 | 0.007400 | 1.005844 |
| `statsforecast_autoets` | 0.013408 | 0.007456 | 1.013524 |
| `functime_ridge` | 0.013474 | 0.007670 | 1.042553 |
| `statsforecast_autotbats` | 0.013477 | 0.007663 | 1.041580 |
| `statsforecast_autoces` | 0.013522 | 0.007617 | 1.035289 |
| `statsforecast_dynamic_optimized_theta` | 0.013669 | 0.008204 | 1.115187 |
| `functime_snaive` | 0.013674 | 0.007500 | 1.019396 |
| `functime_lightgbm` | 0.013674 | 0.007500 | 1.019396 |
| `statsforecast_seasonal_naive` | 0.013674 | 0.007500 | 1.019396 |
| `prophet_additive` | 0.013674 | 0.007500 | 1.019396 |
| `lightgbm_lag` | 0.013674 | 0.007500 | 1.019396 |
| `statsforecast_autotheta` | 0.013683 | 0.008228 | 1.118462 |
| `xgboost_lag` | 0.014246 | 0.008896 | 1.209160 |
| `cartoboost_lag` | 0.014348 | 0.009357 | 1.271868 |

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
- M5 full-roster evidence is a 100-series sample with a 90-day materialized
  history window; the full 30,490-series artifact is CartoBoost-only with fast
  tree settings.
- M6 uses a daily point-forecast proxy, not the official RPS/investment-return
  competition scorer.
- External model availability depends on optional benchmark dependencies.
