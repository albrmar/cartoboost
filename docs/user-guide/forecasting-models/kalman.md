# Kalman

`KalmanForecaster` is a Rust local-linear-trend state-space forecaster. It is
useful when the series has a noisy level and a slowly changing trend.

## When To Use

Use Kalman forecasting for taxi demand or fare aggregates when recent
observations should update the level and trend without requiring a fixed
seasonal cycle. It is often a good baseline for short horizons and noisy
single-zone series.

## Example

```python
from cartoboost.forecasting import KalmanForecaster

airport_pickups = [72, 75, 79, 82, 80, 86, 91, 96, 94, 99, 103, 108]

model = KalmanForecaster(
    level_process_variance=0.05,
    trend_process_variance=0.005,
    observation_variance=1.0,
)
model.fit(airport_pickups)
forecast = model.predict(6)

print(model.get_metadata())
print(forecast.predictions())
```

## ForecastFrame Example

```python
from cartoboost.forecasting import ForecastFrame, KalmanForecaster

frame = ForecastFrame.from_pandas(
    hourly_zone_demand.query("PULocationID == 132"),
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    freq="h",
)

model = KalmanForecaster(
    level_process_variance=0.10,
    trend_process_variance=0.01,
    observation_variance=2.0,
)
model.fit(frame)
forecast = model.predict(12)
```

## Parameters

| Parameter | Effect |
| --- | --- |
| `level_process_variance` | Larger values let the level change faster. |
| `trend_process_variance` | Larger values let the trend change faster. |
| `observation_variance` | Larger values make the model trust noisy observations less. |

## Tuning Pattern

Start with the defaults, then evaluate a small grid:

```python
candidates = [
    {"level_process_variance": 0.02, "trend_process_variance": 0.002, "observation_variance": 1.0},
    {"level_process_variance": 0.05, "trend_process_variance": 0.005, "observation_variance": 1.0},
    {"level_process_variance": 0.10, "trend_process_variance": 0.010, "observation_variance": 2.0},
]
```

Select settings with rolling-origin validation, not in-sample fit.

## Validation Notes

Kalman models do not encode hour-of-day wraparound or zone geography. If those
effects matter, compare Kalman against seasonal naive and a lag model with
calendar features.
