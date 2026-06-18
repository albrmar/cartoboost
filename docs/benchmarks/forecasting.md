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
  --no-download \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_plots
```

## Models

| library | model names |
| --- | --- |
| `cartoboost` | `cartoboost_lag` |
| `functime` | `functime_snaive`, `functime_ridge`, `functime_lightgbm` |
| `statsforecast` | `statsforecast_seasonal_naive`, `statsforecast_autoets` |
| `prophet` | `prophet_additive` |

CartoBoost uses a guarded seasonal-residual strategy. It starts from the 7-day
seasonal baseline, trains a CartoBoost residual model from lag and rolling
features plus pickup zone, dropoff zone, route distance, airport-lane flag, and
pickup borough code, then estimates residual shrinkage from rolling origins
inside the training window. The residual correction is applied only when that
inner validation shows at least a 1% RMSE gain. Library baselines use their
native panel forecasting inputs.

## Result

RMSE is the primary quality metric. MAE and WAPE are secondary quality metrics.
On this short real holdout, CartoBoost ties the best seasonal-naive external
baselines. The train-only residual calibration used 6 inner rolling origins and
found only a 0.07% raw RMSE gain for the residual correction, below the fixed 1%
materiality threshold, so the residual weight was set to `0.0`.

| model | library | RMSE | MAE | WAPE |
| --- | --- | ---: | ---: | ---: |
| `cartoboost_lag` | `cartoboost` | 39.034 | 29.173 | 0.094 |
| `functime_snaive` | `functime` | 39.034 | 29.173 | 0.094 |
| `statsforecast_seasonal_naive` | `statsforecast` | 39.034 | 29.173 | 0.094 |
| `statsforecast_autoets` | `statsforecast` | 44.598 | 36.242 | 0.116 |
| `prophet_additive` | `prophet` | 79.876 | 60.655 | 0.195 |
| `functime_ridge` | `functime` | 90.961 | 63.403 | 0.204 |
| `functime_lightgbm` | `functime` | 99.754 | 66.426 | 0.213 |

![Forecasting tool metric comparison](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_tool_metric_comparison.png)

![Horizon RMSE by forecasting tool](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_horizon_rmse_by_tool.png)

![Forecast lines for top pickup/dropoff lanes](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_forecast_lines.png)

![Actual vs predicted demand by model](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_actual_vs_predicted.png)

## Interpretation

This is a short panel with strong weekly demand structure and only 24 training
days before the 7-day holdout. The result is useful precisely because it shows
that the honest best model for this slice is mostly seasonal persistence. The
CartoBoost residual path is still part of the model, but the train-only
calibration rejects it when the residual signal is not strong enough. That
prevents the benchmark from claiming a holdout-tuned residual gain.

The synthetic fixture remains useful for deterministic library wiring checks,
but real-data claims should use the NYC TLC artifact above.
