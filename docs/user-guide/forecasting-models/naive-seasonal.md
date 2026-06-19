# Naive And Seasonal Naive

Naive and seasonal naive models are the first baselines to run for forecasting.
They are intentionally simple and make leakage problems easier to spot.

## When To Use

Use `NaiveForecaster` when the next value should be compared against the last
observed value. Use `SeasonalNaiveForecaster` when the series has a stable
cycle, such as hourly pickup demand with `season_length=24` or daily demand
with `season_length=7`.

| Model | Behavior |
| --- | --- |
| `NaiveForecaster` | Repeats the most recent observed value for each future horizon. |
| `SeasonalNaiveForecaster(season_length)` | Repeats values from the most recent completed seasonal cycle. |

## Single-Series Example

```python
from cartoboost.forecasting import NaiveForecaster, SeasonalNaiveForecaster

hourly_pickups = [42, 38, 35, 31, 44, 67, 91, 105, 98, 86, 73, 69]

last_value = NaiveForecaster().fit(hourly_pickups)
last_cycle = SeasonalNaiveForecaster(season_length=6).fit(hourly_pickups)

print(last_value.predict(3).predictions())
print(last_cycle.predict(3).predictions())
```

## Pickup-Zone Panel Example

```python
from cartoboost.forecasting import ForecastFrame, SeasonalNaiveForecaster

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="PULocationID",
    freq="h",
)

model = SeasonalNaiveForecaster(season_length=24)
model.fit(frame)
forecast = model.predict(12)

for row in forecast.predictions()[:5]:
    print(row)
```

## Parameters

| Parameter | Applies to | Notes |
| --- | --- | --- |
| `season_length` | `SeasonalNaiveForecaster` | Required positive integer. Use `24` for hourly daily seasonality and `168` for hourly weekly seasonality. |
| `prediction_interval_levels` | Both wrappers | Validated as values between `0` and `1`; interval support depends on the native model output. |

## Validation Notes

Seasonal naive is the minimum meaningful baseline for strongly seasonal taxi
demand. If a more complex model cannot beat seasonal naive under rolling-origin
backtests, inspect feature leakage, horizon alignment, and whether the model is
overfitting repeated zones.
