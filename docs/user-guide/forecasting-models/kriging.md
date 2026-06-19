# Kriging

`KrigingForecaster` is a Rust ordinary-kriging panel forecaster. It borrows
signal across series using explicit coordinates keyed by series id.

## When To Use

Use kriging when nearby pickup zones, dropoff zones, or route midpoints should
have related future values. Every series id in the training panel should have a
coordinate.

## Example With Panel Dict

```python
from cartoboost.forecasting import KrigingForecaster

coordinates = {
    "132": (-73.7781, 40.6413),  # JFK area
    "161": (-73.9776, 40.7580),  # Midtown
    "236": (-73.9577, 40.7808),  # Upper East Side
}

series = {
    "132": [120, 118, 125, 140, 155, 160],
    "161": [80, 84, 91, 105, 118, 121],
    "236": [62, 65, 70, 78, 85, 88],
}

model = KrigingForecaster(
    coordinates=coordinates,
    range=2.0,
    nugget=1.0e-6,
)
model.fit(series)
forecast = model.predict(3)

for row in forecast.predictions():
    print(row)
```

## ForecastFrame Example

```python
from cartoboost.forecasting import ForecastFrame, KrigingForecaster

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="PULocationID",
    freq="h",
)

zone_centroids = {
    "132": (-73.7781, 40.6413),
    "161": (-73.9776, 40.7580),
    "236": (-73.9577, 40.7808),
}

model = KrigingForecaster(coordinates=zone_centroids, range=2.0, nugget=1e-6)
model.fit(frame)
forecast = model.predict(6)
```

## Parameters

| Parameter | Notes |
| --- | --- |
| `coordinates` | Mapping from series id to `(x, y)` or rows of `(series_id, x, y)`. |
| `range` | Spatial correlation range. Larger values make distant zones influence each other more. |
| `nugget` | Small positive stabilizer for the kriging system. |

## Data Requirements

Kriging is panel-oriented. Use stable series ids such as `PULocationID`,
`DOLocationID`, or route ids. Coordinate keys must match the string form of the
series ids used by the forecasting frame or panel dictionary.

## Validation Notes

Kriging uses spatial proximity, not future observations. Validate with
rolling-origin folds and consider a spatial holdout if the claim is about
generalizing to unseen zones.
