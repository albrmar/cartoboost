# Forecasting Tool Benchmark

## Decision Question

For pickup/dropoff lane demand forecasting, should a scientist choose
`cartoboost_lag`, a dedicated forecasting library, or a simple seasonal
baseline?

This page keeps the answer split-specific. CartoBoost is useful when its
train-only lag and residual features improve future demand forecasts. A
seasonal-naive method is the honest choice when a short panel contains too
little validated residual signal.

## Real NYC Taxi Panel

The maintained real-data artifact aggregates January 2024 yellow taxi trips
into daily pickup/dropoff lane demand for the 24 highest-volume lanes. Raw TLC
Parquet files stay under `data/nyc_taxi/` and are not committed. The committed
benchmark result contains only aggregate metrics and plots.

This differs from row-level taxi fare and duration modeling. Each forecast row
is one pickup/dropoff lane on one future date, and the target is the next daily
trip count for that lane.

| field | value |
| --- | ---: |
| raw TLC trip rows | 2,964,624 |
| cleaned trip rows | 2,827,685 |
| lane series | 24 |
| daily periods | 31 |
| held-out horizon | 7 days |
| aggregate forecast rows | 168 |

Source: [NYC TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Reproduce Real-Data Artifact

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source nyc-taxi \
  --year 2024 \
  --months 1 \
  --taxi-type yellow \
  --lanes 24 \
  --horizon 7 \
  --cartoboost-n-estimators 160 \
  --cartoboost-max-depth 6 \
  --no-download \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_plots
```

`--no-download` means missing real TLC inputs fail the run instead of silently
falling back to synthetic data.

## Models

| library | model names |
| --- | --- |
| `cartoboost` | `cartoboost_lag` |
| `functime` | `functime_snaive`, `functime_ridge`, `functime_lightgbm` |
| `statsforecast` | `statsforecast_seasonal_naive`, `statsforecast_autoets`, `statsforecast_autoarima`, `statsforecast_autotheta` |
| `prophet` | `prophet_additive` |

CartoBoost uses a guarded seasonal-residual strategy with 160 estimators, depth
6, and adaptive lag features for short histories. It starts from the 7-day
seasonal baseline, trains a CartoBoost residual model from lag and rolling
features plus pickup zone, dropoff zone, route distance, airport-lane flag, and
pickup borough code, then estimates residual shrinkage inside the training
window. The residual correction is applied only when the raw residual model
itself shows at least a 1% RMSE gain. Library baselines use their native panel
forecasting inputs.

## Real-Data Result

RMSE is the primary metric. MAE and WAPE are secondary. On this short real
holdout, CartoBoost ties the best seasonal-naive external baselines. The
training window is too short for the guarded residual and calendar-profile
selector, so CartoBoost falls back to the 7-day seasonal baseline instead of
applying an undervalidated correction.

| model | library | RMSE | MAE | WAPE |
| --- | --- | ---: | ---: | ---: |
| `cartoboost_lag` | `cartoboost` | 39.034 | 29.173 | 0.094 |
| `functime_snaive` | `functime` | 39.034 | 29.173 | 0.094 |
| `statsforecast_seasonal_naive` | `statsforecast` | 39.034 | 29.173 | 0.094 |
| `statsforecast_autoets` | `statsforecast` | 42.976 | 33.827 | 0.109 |
| `statsforecast_autoarima` | `statsforecast` | 56.986 | 42.836 | 0.138 |
| `functime_ridge` | `functime` | 75.876 | 54.380 | 0.175 |
| `prophet_additive` | `prophet` | 79.876 | 60.655 | 0.195 |
| `functime_lightgbm` | `functime` | 84.039 | 57.395 | 0.184 |
| `statsforecast_autotheta` | `statsforecast` | 115.395 | 73.488 | 0.236 |

![Forecasting tool metric comparison](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_tool_metric_comparison.png)

![Horizon RMSE by forecasting tool](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_horizon_rmse_by_tool.png)

![Forecast lines for top pickup/dropoff lanes](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_forecast_lines.png)

![Actual vs predicted demand by model](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_actual_vs_predicted.png)

## Real-Data Interpretation

The maintained NYC panel has strong weekly persistence and only 24 training
days before the 7-day holdout. The useful conclusion is that the honest best
model for this slice is mostly seasonal persistence. CartoBoost does not claim
a residual gain because train-only calibration rejects the residual correction.

For model choice, use `cartoboost_lag` when longer panels or richer known-future
features show a validated residual gain. Use seasonal naive when the future
holdout shows no validated improvement.

## Synthetic Problem Suite

The script also supports a multi-problem synthetic suite modeled after common
forecasting benchmark practice: multiple tasks, the same model roster,
rolling-origin splits, fixed horizons, and aggregate relative RMSE to each
problem's best model.

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source polars \
  --suite \
  --suite-folds 3 \
  --lanes 12 \
  --days 180 \
  --horizon 14 \
  --cartoboost-n-estimators 160 \
  --cartoboost-max-depth 6 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_library_suite_synthetic.json
```

The suite contains four taxi-shaped forecasting tasks: ordinary weekly lane
demand, airport calendar events, route-mix shifts, and borough monthly pulses.
On this suite, CartoBoost has the best average RMSE ratio to the problem winner
and wins or ties 2 of 4 tasks.

| model | mean RMSE ratio to problem best | wins or ties | top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_lag` | 1.021 | 2 | 3 |
| `statsforecast_autoarima` | 1.213 | 0 | 3 |
| `prophet_additive` | 1.220 | 2 | 4 |
| `statsforecast_autoets` | 1.538 | 0 | 1 |
| `statsforecast_autotheta` | 1.602 | 0 | 1 |
| `functime_snaive` | 1.774 | 0 | 0 |
| `statsforecast_seasonal_naive` | 1.774 | 0 | 0 |
| `functime_ridge` | 3.098 | 0 | 0 |
| `functime_lightgbm` | 3.402 | 0 | 0 |

Use this suite for deterministic library wiring and taxi-shaped stress tests.
Real-data claims should still point to TLC or another named dataset.

## M4 Forecasting Suite

M4 is not a taxi dataset. It is included because forecasting libraries commonly
use it to test whether a method generalizes across frequencies. The loader uses
`datasetsforecast.m4.M4.load`, which downloads the real M4 train/test files
locally when they are missing. This benchmark scores the last official horizon
inside the training panel so every model can be evaluated through the same
recursive CartoBoost-style interface.

For a practical local artifact, the maintained all-group run scores the first
24 series in each M4 group. Use `--m4-series-limit 0` to run every series in a
group or across the all-group suite; that is much slower because it also fits
per-series Prophet and AutoARIMA baselines.

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m4 \
  --m4-suite \
  --m4-series-limit 24 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m4_suite_sample.json
```

Full-series command:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m4 \
  --m4-suite \
  --m4-series-limit 0 \
  --output target/forecasting_m4_suite_full.json
```

On the 24-series-per-group artifact generated at
`2026-06-19T04:42:22.388275+00:00`, CartoBoost uses the native Rust
`CartoBoostLagForecaster` path with season-aware lag, rolling, delta, and trend
features capped to the shortest training series in the split. The benchmark
uses fixed CartoBoost defaults of 180 estimators, learning rate 0.06, max depth
4, and min leaf size 8, with fixed structural regularization for high-frequency
or quarterly seasonal regimes: those use at least max depth 5 and at most min
leaf size 6. Target mode is selected by a fixed horizon/seasonality heuristic:
`delta_from_last` when `season_length == 12`, `horizon == 13`, or
`horizon >= 24`; level targets otherwise.

Every model receives the same train-side candidate-selection pass: its raw
forecast is compared against the same fixed seasonal, calendar, drift,
half-drift, seasonal-drift, and seasonal-cycle drift candidates, and the choice
is made on an inner training holdout. Under that symmetric protocol, CartoBoost
wins or ties all 6 M4 groups, has 6 top-3 finishes, and has the best aggregate
mean RMSE ratio to the group winner on this maintained sample artifact.

| model | mean RMSE ratio to group best | wins or ties | top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_lag` | 1.000 | 6 | 6 |
| `statsforecast_autoarima` | 1.244 | 2 | 2 |
| `statsforecast_autoets` | 1.275 | 2 | 2 |
| `statsforecast_autotheta` | 1.296 | 2 | 1 |
| `functime_snaive` | 1.347 | 2 | 4 |
| `functime_lightgbm` | 1.347 | 2 | 0 |
| `statsforecast_seasonal_naive` | 1.347 | 2 | 0 |
| `prophet_additive` | 1.347 | 2 | 0 |
| `functime_ridge` | 1.368 | 2 | 3 |

| M4 group | winner | CartoBoost RMSE | best external model | best external RMSE | CartoBoost / best external RMSE | CartoBoost selected candidate |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Hourly | `cartoboost_lag` | 1285.476 | `statsforecast_autoarima` | 2220.190 | 0.579 | `cartoboost_lag` |
| Daily | `cartoboost_lag` | 39.603 | `functime_snaive` | 41.800 | 0.947 | `cartoboost_lag` |
| Weekly | `cartoboost_lag` | 230.234 | `statsforecast_autoets` | 234.643 | 0.981 | `cartoboost_lag` |
| Monthly | `tie` | 318.040 | `functime_snaive` | 318.040 | 1.000 | `shared_half_drift` |
| Quarterly | `cartoboost_lag` | 637.616 | `statsforecast_autoets` | 1021.905 | 0.624 | `cartoboost_lag` |
| Yearly | `tie` | 833.200 | `functime_snaive` | 833.200 | 1.000 | `shared_drift` |

The M4 result prevents a benchmark story that only contains taxi-shaped
problems. Applying the same fixed selector to every model also prevents
CartoBoost from receiving private post-processing. The remaining caveat is
scope: this is the maintained 24-series-per-group M4 sample artifact, not the
full M4 corpus. Use `--m4-series-limit 0` before making full-corpus claims.

Benchmarking references used for this suite:

- [`datasetsforecast` M4 loader and group definitions](https://nixtlaverse.nixtla.io/datasetsforecast/m4.html)
- [StatsForecast cross-validation and baseline workflow](https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_complete.html)
- [sktime `ForecastingBenchmark` design](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.benchmarking.forecasting.ForecastingBenchmark.html)
- [Darts historical forecasts and backtesting](https://unit8co.github.io/darts/)
- [GluonTS dataset/evaluation workflow](https://ts.gluon.ai/stable/tutorials/forecasting/extended_tutorial.html)

## Reporting Requirements

When refreshing this benchmark, capture RMSE, MAE, WAPE, horizon metrics,
training time, prediction time, model settings, sample size, task names, split
names, and whether the data is real TLC, synthetic taxi-shaped, M4 sample, or
full M4. Update this page in the same change as the artifact.
