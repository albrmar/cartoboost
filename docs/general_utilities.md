# General Utilities

CartoBoost includes Rust-backed numerical utilities that can be used without
building a `CartoBoostRegressor` or a forecasting `ForecastFrame`. Use these
when you have a plain Python sequence, a small spatial interpolation problem, or
an intermittent-demand sequence and want a direct result.

```python
import cartoboost as cb
```

## Local-Level Kalman

Use local-level Kalman filtering when the signal is a noisy measurement of a
slowly moving level and there is no explicit trend term.

Toy problem: a warehouse inventory sensor reads a stable stock level with small
noise, and you want a smoothed level plus a short flat forecast.

```python
import cartoboost as cb

readings = [99.0, 101.0, 100.0, 102.0, 101.0, 103.0]

state = cb.local_level_kalman_filter(
    readings,
    level_process_variance=0.05,
    observation_variance=1.0,
    horizon=3,
)

print(state["final_state"]["level"])
print(state["forecast"])
```

For just the forecast means:

```python
forecast = cb.local_level_kalman_forecast(
    readings,
    horizon=3,
    level_process_variance=0.05,
    observation_variance=1.0,
)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `level_process_variance` | How much the latent level is allowed to move between observations. Larger values adapt faster. |
| `observation_variance` | Measurement noise. Larger values smooth the observations more strongly. |
| `horizon` | Number of future means to emit from the filtered final state. |

Return shape:

- `local_level_kalman_filter(...)` returns a dictionary with per-step estimates,
  final level state, and optional `forecast`.
- `local_level_kalman_forecast(...)` returns `list[float]`.

## Local-Linear-Trend Kalman

Use local-linear Kalman filtering when the series has a level and a slope. This
is usually a better toy model for load, traffic, or pickup counts that drift
over time.

Toy problem: daily pickup demand is increasing by about two trips per day, with
noise.

```python
import cartoboost as cb

pickups = [40.0, 42.0, 45.0, 47.0, 50.0, 51.0, 54.0]

state = cb.kalman_filter(
    pickups,
    level_process_variance=0.05,
    trend_process_variance=0.005,
    observation_variance=0.5,
    horizon=4,
)

print(state["final_state"]["level"], state["final_state"]["trend"])
print(state["forecast"])
```

For just the forecast means:

```python
forecast = cb.local_linear_trend_kalman_forecast(
    pickups,
    horizon=4,
    level_process_variance=0.05,
    trend_process_variance=0.005,
    observation_variance=0.5,
)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `level_process_variance` | Process noise for the latent level. |
| `trend_process_variance` | Process noise for the latent trend/slope. |
| `observation_variance` | Measurement noise in the observed values. |
| `horizon` | Number of future means to emit from the final level/trend state. |

Forecasting wrapper:

Use `KalmanForecaster` when you want the same local-linear model behind the
forecasting API.

```python
from cartoboost.forecasting.local import KalmanForecaster

model = KalmanForecaster(
    level_process_variance=0.05,
    trend_process_variance=0.005,
    observation_variance=0.5,
)
model.fit([40.0, 42.0, 45.0, 47.0, 50.0, 51.0, 54.0])
result = model.predict(4)

print(result.predictions())
```

Panel input is also supported:

```python
model.fit(
    {
        "PULocationID_142": [40.0, 42.0, 45.0, 47.0],
        "PULocationID_236": [70.0, 69.0, 68.0, 66.0],
    }
)
```

## Ordinary Kriging

Use ordinary kriging when you have observed values at known coordinates and want
spatial interpolation at new coordinates. The utility is independent of
forecasting. The forecasting wrapper uses the latest observed value for each
series and interpolates across panel coordinates.

Toy problem: three taxi zones have observed pickup pressure, and you want the
estimated pressure at a nearby zone centroid.

```python
import cartoboost as cb

observations = [
    (0.0, 0.0, 10.0),  # x, y, value
    (1.0, 0.0, 20.0),
    (0.0, 1.0, 14.0),
]
targets = [
    (0.25, 0.25),
    (0.75, 0.10),
]

predictions = cb.ordinary_kriging_predict(
    observations,
    targets,
    range=1.5,
    nugget=1.0e-6,
)

for row in predictions:
    print(row["x"], row["y"], row["mean"], row["weights"])
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `observations` | Sequence of `(x, y, value)` triples. |
| `targets` | Sequence of `(x, y)` coordinates to interpolate. |
| `range` | Distance scale for spatial covariance. Larger values spread influence farther. |
| `nugget` | Small diagonal variance for numerical stability and measurement noise. |

Return shape:

- `ordinary_kriging_predict(...)` returns a list of dictionaries:
  `{"x": ..., "y": ..., "mean": ..., "weights": [...]}`.

Forecasting wrapper:

Use `KrigingForecaster` when the data is a panel of series and each series has
a fixed coordinate. The wrapper forecasts each series by kriging the latest
known panel values at that series coordinate.

```python
from cartoboost.forecasting.local import KrigingForecaster

coordinates = {
    "PULocationID_142": (0.0, 0.0),
    "PULocationID_236": (1.0, 0.0),
    "PULocationID_239": (0.0, 1.0),
}

model = KrigingForecaster(coordinates=coordinates, range=1.5, nugget=1.0e-6)
model.fit(
    {
        "PULocationID_142": [10.0, 12.0],
        "PULocationID_236": [20.0, 21.0],
        "PULocationID_239": [14.0, 15.0],
    }
)
result = model.predict(1)

print(result.predictions())
```

Coordinates can also be passed as triples:

```python
KrigingForecaster(
    coordinates=[
        ("PULocationID_142", 0.0, 0.0),
        ("PULocationID_236", 1.0, 0.0),
    ],
)
```

## Intermittent Demand

Use Croston-family utilities when demand is non-negative and contains many
zeros. They are useful for sparse pickup requests, low-volume parts, or rare
lane/customer activity.

Toy problem: a low-volume lane has several zero-demand days and occasional
orders.

```python
import cartoboost as cb

demand = [0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 5.0, 0.0]

croston = cb.croston_forecast(demand, horizon=5, alpha=0.2)
sba = cb.sba_forecast(demand, horizon=5, alpha=0.2)
tsb = cb.tsb_forecast(demand, horizon=5, alpha=0.2, beta=0.1)

print(croston)
print(sba)
print(tsb)
```

Method guide:

| Method | Entry point | Use when |
| --- | --- | --- |
| Croston | `croston_forecast` or `intermittent_demand_forecast(method="croston")` | You need a simple baseline for sparse non-zero events. |
| SBA | `sba_forecast` or `intermittent_demand_forecast(method="sba")` | You want Croston with a standard bias adjustment. |
| TSB | `tsb_forecast` or `intermittent_demand_forecast(method="tsb")` | You want separate smoothing for event probability and non-zero demand size. |

Inputs must be finite and non-negative. The output is a `list[float]` of length
`horizon`.

## Single-Series Forecast Utilities

The same Rust-native local models used by forecasting wrappers can be called on
plain numeric sequences:

```python
import cartoboost as cb

values = [20.0, 21.0, 19.0, 22.0, 23.0, 24.0, 26.0]

print(cb.naive_forecast(values, horizon=3))
print(cb.seasonal_naive_forecast(values, horizon=3, season_length=7))
print(cb.theta_forecast(values, horizon=3))
print(cb.optimized_theta_forecast(values, horizon=3))
print(cb.ets_forecast(values, horizon=3, alpha=0.5, beta=0.1))
print(cb.arima_forecast(values, horizon=3, p=1, d=1, q=0))
print(cb.auto_arima_forecast(values, horizon=3, max_p=2, max_d=1, max_q=1))
```

Use `series_forecast` when the model name is dynamic:

```python
forecast = cb.series_forecast(
    "local_linear_trend_kalman",
    values,
    horizon=3,
    level_process_variance=0.05,
    trend_process_variance=0.005,
    observation_variance=1.0,
)
```

## Choosing A Utility

| Problem shape | Default utility |
| --- | --- |
| Noisy stable level | `local_level_kalman_filter` |
| Noisy level plus slope | `kalman_filter` or `local_linear_trend_kalman_forecast` |
| Spatial interpolation from coordinate samples | `ordinary_kriging_predict` |
| Sparse non-negative demand | `sba_forecast` or `tsb_forecast` |
| Quick local time-series baseline | `series_forecast` or the named forecast helpers |
