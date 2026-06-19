# Theta

Theta models provide lightweight trend extrapolation. They are useful when taxi
demand has a clear local trend but you do not need the full ARIMA or lag-model
surface.

## Models

| Model | Use when |
| --- | --- |
| `ThetaForecaster` | You want to set `theta` and smoothing `alpha` directly. |
| `OptimizedThetaForecaster` | You want deterministic grid search over theta and alpha values. |

## Example

```python
from cartoboost.forecasting import OptimizedThetaForecaster, ThetaForecaster

hourly_airport_pickups = [
    88, 84, 79, 72, 91, 118, 146, 162, 155, 141, 130, 126,
    92, 89, 83, 75, 96, 123, 151, 168, 160, 146, 133, 129,
]

manual = ThetaForecaster(theta=2.0, alpha=0.2)
manual.fit(hourly_airport_pickups)
manual_forecast = manual.predict(6)

optimized = OptimizedThetaForecaster(
    theta_grid=(1.0, 1.5, 2.0, 2.5, 3.0),
    alpha_grid=(0.1, 0.2, 0.4, 0.6),
)
optimized.fit(hourly_airport_pickups)
optimized_forecast = optimized.predict(6)
```

## Seasonal Theta

Seasonality requires both `seasonality` and `season_length`:

```python
from cartoboost.forecasting import ThetaForecaster

model = ThetaForecaster(
    theta=2.0,
    alpha=0.2,
    seasonality="additive",
    season_length=24,
)
model.fit(hourly_zone_demand_values)
forecast = model.predict(24)
```

`seasonality` accepts `None`, `"additive"`, or `"multiplicative"` at the Python
validation layer. Unsupported native combinations fail explicitly.

## Parameters

| Parameter | Notes |
| --- | --- |
| `theta` | Positive trend-control value for `ThetaForecaster`. |
| `alpha` | Smoothing value in `(0, 1]`. |
| `theta_grid` | Non-empty positive candidate values for `OptimizedThetaForecaster`. |
| `alpha_grid` | Non-empty smoothing candidates in `(0, 1]`. |
| `season_length` | Required when `seasonality` is set. |
| `prediction_interval_levels` | Validated interval levels between `0` and `1`. |

## Validation Notes

Compare theta against seasonal naive on the same rolling-origin folds. Theta can
look strong on trending demand, but it should not be used as evidence of
spatial generalization because it does not know pickup-zone geometry.
