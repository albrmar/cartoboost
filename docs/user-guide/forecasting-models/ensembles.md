# Weighted Ensembles

`WeightedEnsembleForecaster` combines native CartoBoost forecasting wrappers
with explicit fixed weights.

## When To Use

Use an ensemble when different models capture different parts of the taxi demand
pattern. For example, seasonal naive can preserve repeated hourly cycles while
theta or Kalman can adapt to a changing level.

## Example

```python
from cartoboost.forecasting import (
    KalmanForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
    WeightedEnsembleForecaster,
)

models = {
    "seasonal": SeasonalNaiveForecaster(season_length=24),
    "theta": ThetaForecaster(theta=2.0, alpha=0.2),
    "kalman": KalmanForecaster(),
}

ensemble = WeightedEnsembleForecaster(
    models=models,
    weights={
        "seasonal": 0.50,
        "theta": 0.30,
        "kalman": 0.20,
    },
    metadata={"purpose": "pickup-zone demand baseline"},
)

ensemble.fit(hourly_pickups)
forecast = ensemble.predict(12)
print(ensemble.get_metadata())
print(forecast.predictions())
```

## ForecastFrame Example

```python
from cartoboost.forecasting import ForecastFrame, SeasonalNaiveForecaster
from cartoboost.forecasting import ThetaForecaster, WeightedEnsembleForecaster

frame = ForecastFrame.from_pandas(
    hourly_zone_demand.query("PULocationID == 161"),
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    freq="h",
)

ensemble = WeightedEnsembleForecaster(
    models={
        "seasonal": SeasonalNaiveForecaster(season_length=24),
        "theta": ThetaForecaster(),
    },
    weights={"seasonal": 0.7, "theta": 0.3},
)
ensemble.fit(frame)
forecast = ensemble.predict(24)
```

## Rules

| Rule | Details |
| --- | --- |
| At least one model is required | Empty `models` raises `ValueError`. |
| Weights must match model names exactly | Missing or extra names raise `ValueError`. |
| Components must be native wrappers | Arbitrary Python estimators are not accepted. |
| Prediction intervals are not supported yet | Interval arguments raise `NotImplementedError`. |

Weights are passed as explicit numbers. They do not have to be learned by the
ensemble, so choose them from validation results or a fixed operating policy.

## Validation Notes

Report every component model and its weight. Compare the ensemble against the
best individual component, not only against the weakest baseline.
