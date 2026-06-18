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

CartoBoost uses supervised lag and rolling-window features plus pickup zone,
dropoff zone, route distance, airport-lane flag, and pickup borough code. The
library baselines use their native panel forecasting inputs.

## Result

RMSE is the primary quality metric. MAE and WAPE are secondary quality metrics.
On this short real holdout, seasonal naive methods win. CartoBoost lag
forecasting is faster than Prophet but does not beat the best external
forecasting-library baseline.

| model | library | RMSE | MAE | WAPE |
| --- | --- | ---: | ---: | ---: |
| `functime_snaive` | `functime` | 39.034 | 29.173 | 0.094 |
| `statsforecast_seasonal_naive` | `statsforecast` | 39.034 | 29.173 | 0.094 |
| `statsforecast_autoets` | `statsforecast` | 44.598 | 36.242 | 0.116 |
| `cartoboost_lag` | `cartoboost` | 59.880 | 43.040 | 0.138 |
| `prophet_additive` | `prophet` | 79.876 | 60.655 | 0.195 |
| `functime_ridge` | `functime` | 90.961 | 63.403 | 0.204 |
| `functime_lightgbm` | `functime` | 99.754 | 66.426 | 0.213 |

![Forecasting tool metric comparison](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_tool_metric_comparison.png)

![Horizon RMSE by forecasting tool](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_horizon_rmse_by_tool.png)

![Forecast lines for top pickup/dropoff lanes](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_forecast_lines.png)

![Actual vs predicted demand by model](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_actual_vs_predicted.png)

## Interpretation

This is a short panel with strong weekly demand structure and only 24 training
days before the 7-day holdout. The result is useful precisely because it shows a
case where a simple seasonal baseline is hard to beat. The committed plots make
the failure mode visible: CartoBoost captures broad lane levels, but it
overstates parts of the final week relative to the seasonal naive and AutoETS
baselines.

The synthetic fixture remains useful for deterministic library wiring checks,
but real-data claims should use the NYC TLC artifact above.
