# ARIMA And AutoARIMA

ARIMA models are useful when differencing and autocorrelation explain the
series. CartoBoost exposes bounded non-seasonal Rust ARIMA wrappers.

## Models

| Model | Import | Use when |
| --- | --- | --- |
| `ArimaForecaster` | `from cartoboost.forecasting.local import ArimaForecaster` | You know the non-seasonal `(p, d, q)` order. |
| `AutoARIMAForecaster` | `from cartoboost.forecasting import AutoARIMAForecaster` | You want bounded candidate search over `(p, d, q)`. |

## Fixed-Order Example

```python
from cartoboost.forecasting.local import ArimaForecaster

hourly_pickups = [42, 38, 35, 31, 44, 67, 91, 105, 98, 86, 73, 69]

model = ArimaForecaster(p=2, d=1, q=1)
model.fit(hourly_pickups)
forecast = model.predict(6)

for row in forecast.predictions():
    print(row)
```

## AutoARIMA Example

```python
from cartoboost.forecasting import AutoARIMAForecaster

model = AutoARIMAForecaster(
    seasonal=False,
    max_p=3,
    max_d=1,
    max_q=2,
)
model.fit(hourly_pickups)
forecast = model.predict(12)
metadata = model.get_metadata()
```

## ForecastFrame Example

```python
from cartoboost.forecasting import AutoARIMAForecaster, ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand.query("PULocationID == 132"),
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    freq="h",
)

model = AutoARIMAForecaster(max_p=4, max_d=1, max_q=2)
model.fit(frame)
forecast = model.predict(24)
```

## Parameters

| Parameter | Notes |
| --- | --- |
| `p`, `d`, `q` | Non-negative order values for `ArimaForecaster`; `p <= 8`, `d <= 2`, and `q <= 8`. |
| `max_p`, `max_d`, `max_q` | Non-negative AutoARIMA search bounds with the same upper limits. |
| `seasonal` | Must be `False`; seasonal AutoARIMA is not supported by the current Rust binding. |
| `m` | Positive integer accepted by the wrapper, but seasonal search is not enabled. |
| `error_policy` | Must be `"raise"`. |

## Validation Notes

ARIMA is a single-series model family. For many pickup-zone series, either
backtest a representative set of zones independently or use
`CartoBoostLagForecaster` to learn one global model across panels.
