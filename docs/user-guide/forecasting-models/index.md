# Forecasting Model Guides

These pages document the main forecasting model classes with taxi-domain
examples. Each class is a thin Python wrapper over a Rust implementation exposed
through `cartoboost._native`.

Start with a baseline, then move to the model that matches the series structure:

| Model page | Use when |
| --- | --- |
| [Naive And Seasonal Naive](naive-seasonal.md) | You need transparent last-value or last-season baselines. |
| [Theta](theta.md) | A lightweight trend extrapolator is appropriate. |
| [ETS](ets.md) | Level, trend, and additive seasonality explain the series. |
| [ARIMA And AutoARIMA](arima.md) | Autocorrelation and differencing are important. |
| [Kalman](kalman.md) | A noisy local level and local trend should update over time. |
| [Kriging](kriging.md) | Nearby pickup zones or route midpoints should borrow spatial signal. |
| [CartoBoost Lag](cartoboost-lag.md) | Many related series should share one supervised lag model. |
| [Weighted Ensembles](ensembles.md) | Several native forecasters should be combined with explicit weights. |

## Shared Result Shape

Native forecasting models return a `ForecastResult` object. Use
`predictions()` for row tuples:

```python
forecast = model.predict(3)
rows = forecast.predictions()

for series_id, timestamp, horizon, model_name, mean in rows:
    print(series_id, timestamp, horizon, model_name, mean)
```

The tuple columns are also available from `forecast.columns()`. Use
`forecast.to_json()` and `cartoboost._native.ForecastResult.from_json(...)` for
native JSON roundtrips.

## Shared Input Shape

For quick examples, every forecaster can fit a plain numeric list:

```python
model.fit([18.0, 21.0, 23.0, 20.0])
```

For production taxi demand or fare-duration workflows, prefer a validated
`ForecastFrame`:

```python
from cartoboost.forecasting import ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="PULocationID",
    freq="h",
)
```

`ForecastFrame` validates timestamps, duplicate rows within each series, finite
targets, regular frequency, panel ids, and covariate role metadata.
