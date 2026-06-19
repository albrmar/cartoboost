# Forecasting Tool Benchmark

## Question

On real NYC TLC yellow taxi data, how does CartoBoost lag forecasting compare
with dedicated forecasting tools on short pickup/dropoff lane demand panels?

## Data

The maintained real-data artifact aggregates January 2024 yellow taxi trips
into daily pickup/dropoff lane demand for the 24 highest-volume lanes. Raw TLC
Parquet files stay under `data/nyc_taxi/` and are not committed. The committed
benchmark result contains only aggregate metrics and plots.

This is a different modeling problem from the row-level NYC taxi benchmark:
each forecast row is one pickup/dropoff lane on one future date, and the target
is the next daily trip count for that lane. The model is asked to extrapolate a
short time series panel, not predict duration or fare for an already-observed
trip row.

| field | value |
| --- | ---: |
| raw TLC trip rows | 2,964,624 |
| cleaned trip rows | 2,827,685 |
| lane series | 24 |
| daily periods | 31 |
| held-out horizon | 7 days |
| aggregate forecast rows | 168 |

The source is the NYC TLC trip-record release:
<https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>.

## Command

```sh
uv run --group dev --group bench python scripts/forecasting_library_benchmark.py \
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

## Result

RMSE is the primary quality metric. MAE and WAPE are secondary quality metrics.
On this short real holdout, CartoBoost ties the best seasonal-naive external
baselines. The training window is too short for the guarded residual and
calendar-profile selector, so CartoBoost falls back to the 7-day seasonal
baseline instead of applying an undervalidated correction.

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

## Synthetic Problem Suite

The script also supports a multi-problem synthetic suite modeled after common
forecasting benchmark practice in libraries such as sktime, StatsForecast,
Darts, GluonTS, and datasetsforecast: multiple tasks, the same model roster,
rolling-origin splits, fixed horizons, and aggregate relative RMSE to each
problem's best model.

```sh
uv run --group dev --group bench python scripts/forecasting_library_benchmark.py \
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

## M4 Forecasting Suite

The benchmark can also run M4 through the same model roster. M4 is not a taxi
dataset; it is included because forecasting libraries commonly use it to test
whether a method generalizes across frequencies. The loader uses
`datasetsforecast.m4.M4.load`, which downloads the real M4 train/test files
locally when they are missing. This benchmark scores the last official horizon
inside the training panel so every model can be evaluated through the same
recursive CartoBoost-style interface.

For a practical local artifact, the maintained all-group run scores the first
24 series in each M4 group. Use `--m4-series-limit 0` to run every series in a
group or across the all-group suite; that is much slower because it also fits
per-series Prophet and AutoARIMA baselines.

```sh
uv run --group dev --group bench python scripts/forecasting_library_benchmark.py \
  --source m4 \
  --m4-suite \
  --m4-series-limit 24 \
  --cartoboost-n-estimators 80 \
  --cartoboost-max-depth 5 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m4_suite_sample.json
```

Full-series command:

```sh
uv run --group dev --group bench python scripts/forecasting_library_benchmark.py \
  --source m4 \
  --m4-suite \
  --m4-series-limit 0 \
  --cartoboost-n-estimators 80 \
  --cartoboost-max-depth 5 \
  --output target/forecasting_m4_suite_full.json
```

On the 24-series-per-group artifact, CartoBoost wins 4 of 6 M4 groups and has
the best mean RMSE ratio to the group winner. The remaining losses are Hourly,
where functime ridge has better RMSE, and Weekly, where StatsForecast AutoETS
has better RMSE.

| model | mean RMSE ratio to group best | wins or ties | top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_lag` | 1.070 | 4 | 4 |
| `statsforecast_autotheta` | 1.286 | 0 | 3 |
| `statsforecast_autoarima` | 1.371 | 0 | 3 |
| `statsforecast_autoets` | 1.452 | 1 | 4 |
| `functime_snaive` | 1.699 | 0 | 2 |
| `statsforecast_seasonal_naive` | 1.699 | 0 | 1 |
| `functime_ridge` | 2.286 | 1 | 1 |
| `prophet_additive` | 4.244 | 0 | 0 |
| `functime_lightgbm` | 4.344 | 0 | 0 |

| M4 group | winner | CartoBoost RMSE | best external model | best external RMSE | CartoBoost / best external RMSE | CartoBoost selected candidate |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Hourly | `functime_ridge` | 2532.274 | `functime_ridge` | 1951.180 | 1.298 | `cartoboost_seasonal_base` |
| Daily | `cartoboost_lag` | 40.069 | `statsforecast_autoets` | 40.543 | 0.988 | `cartoboost_residual_blend` |
| Weekly | `statsforecast_autoets` | 263.117 | `statsforecast_autoets` | 234.643 | 1.121 | `cartoboost_half_drift` |
| Monthly | `cartoboost_lag` | 318.040 | `statsforecast_autoarima` | 379.135 | 0.839 | `cartoboost_half_drift` |
| Quarterly | `cartoboost_lag` | 879.211 | `statsforecast_autoets` | 1021.905 | 0.860 | `cartoboost_residual_blend` |
| Yearly | `cartoboost_lag` | 311.987 | `functime_snaive` | 418.736 | 0.745 | `cartoboost_residual_blend` |

The M4 result is useful because it prevents a benchmark story that only
contains taxi-shaped problems. The train-calibrated trend candidates close the
large Monthly failure without holdout tuning: the selector chooses
`cartoboost_half_drift` for Monthly, improving CartoBoost from the previous
1323.636 RMSE seasonal-naive path to 318.040 RMSE. Hourly and Weekly remain the
next quality targets before claiming across-frequency dominance.

Benchmarking references used for this suite:

- `datasetsforecast` M4 loader and group definitions:
  <https://nixtlaverse.nixtla.io/datasetsforecast/m4.html>
- StatsForecast cross-validation and baseline workflow:
  <https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_complete.html>
- sktime `ForecastingBenchmark` design:
  <https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.benchmarking.forecasting.ForecastingBenchmark.html>
- Darts historical forecasts and backtesting:
  <https://unit8co.github.io/darts/>
- GluonTS dataset/evaluation workflow:
  <https://ts.gluon.ai/stable/tutorials/forecasting/extended_tutorial.html>

## Interpretation

This is a short panel with strong weekly demand structure and only 24 training
days before the 7-day holdout. The result is useful precisely because it shows
that the honest best model for this slice is mostly seasonal persistence. The
CartoBoost residual path is still part of the model, but the train-only
calibration rejects it when the residual signal is not strong enough. That
prevents the benchmark from claiming a holdout-tuned residual gain.

The synthetic fixture remains useful for deterministic library wiring checks,
but real-data claims should use the NYC TLC artifact above.
