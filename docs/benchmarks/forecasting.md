# Forecasting Tool Benchmark

## Bottom Line

The forecasting benchmark has three maintained views:

- Real NYC taxi lane demand: `cartoboost_lag` ties seasonal-naive baselines on
  the January 2024 24-lane, 7-day holdout.
- Synthetic taxi-shaped suite: `cartoboost_lag` ranks first in the maintained
  committed artifact, with mean RMSE ratio 1.021 to each problem winner.
- M4 24-series-per-group sample: `cartoboost_lag` ranks first by mean RMSE
  ratio in the refreshed committed sample, but this is still a sample, not a
  full M4 corpus claim.

Forecasting claims should stay split-specific. A seasonal baseline tie is a
valid result; it means the short panel did not prove residual lift.

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

## Limitations

- Interval coverage is only benchmark evidence when actually computed.
- Real taxi panel is short: 31 days and a 7-day holdout.
- Synthetic suite is diagnostic.
- M4 artifact is a 24-series-per-group sample.
- External model availability depends on optional benchmark dependencies.

