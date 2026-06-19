# ETS

`ETSForecaster` models level, additive trend, and optional additive seasonality
in Rust. It is a good fit when taxi demand has smooth state updates and a fixed
seasonal pattern.

## Example

```python
from cartoboost.forecasting import ETSForecaster

hourly_midtown_pickups = [
    51, 48, 45, 43, 59, 82, 110, 134, 128, 116, 103, 97,
    55, 50, 47, 44, 61, 86, 114, 139, 132, 119, 106, 99,
]

model = ETSForecaster(
    trend="additive",
    seasonal="additive",
    seasonal_periods=12,
    alpha=0.4,
    beta=0.1,
    gamma=0.2,
)
model.fit(hourly_midtown_pickups)
forecast = model.predict(6)

print(forecast.columns())
print(forecast.predictions())
```

## ForecastFrame Example

```python
from cartoboost.forecasting import ETSForecaster, ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand.query("PULocationID == 161"),
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    freq="h",
)

model = ETSForecaster(
    trend="additive",
    seasonal="additive",
    seasonal_periods=24,
    alpha=0.5,
    beta=0.1,
    gamma=0.2,
)
model.fit(frame)
forecast = model.predict(24)
```

## Parameters

| Parameter | Notes |
| --- | --- |
| `trend` | `None`, `"add"`, or `"additive"`. |
| `seasonal` | `None`, `"add"`, or `"additive"`. |
| `seasonal_periods` | Required and greater than `1` when `seasonal` is set. |
| `damped_trend` | Must be `False`; damped ETS is not currently supported. |
| `alpha` | Level smoothing in `(0, 1]`. |
| `beta` | Trend smoothing in `[0, 1]`. |
| `gamma` | Seasonal smoothing in `[0, 1]`; requires additive seasonality. |

## Unsupported Modes

The current Rust binding rejects damped trends, multiplicative trend, and
multiplicative seasonality. Use explicit failures as a signal to choose another
model rather than silently falling back to Python behavior.

## Validation Notes

ETS should be compared against seasonal naive at the same season length. For
hourly taxi demand, try daily (`24`) and weekly (`168`) season lengths when the
data window is long enough for stable validation.
