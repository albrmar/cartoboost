# Forecasting Lag Features

CartoBoost forecasting builds supervised rows from panel time series and trains a
global `CartoBoostRegressor` across all panels. The Python forecasting layer handles
timestamp sorting, leakage-safe feature construction, recursive multi-step prediction,
and named feature matrices; model fitting and prediction still run through the native
CartoBoost regressor.

Target-derived features are panel isolated. A row for pickup zone `PULocationID=A` at
timestamp `t` only uses target values from the same pickup zone with timestamps strictly
before `t`.

```python
import pandas as pd

from cartoboost.forecasting.global_models import CartoBoostLagForecaster
from cartoboost.forecasting.lag_features import (
    CalendarFeatureConfig,
    LagFeatureConfig,
    RollingFeatureConfig,
)

forecaster = CartoBoostLagForecaster(
    time_col="pickup_hour",
    target_col="pickup_trips",
    panel_cols=["PULocationID"],
    lag_config=LagFeatureConfig(lags=[1, 24, 168]),
    rolling_config=RollingFeatureConfig(
        windows=[3, 24],
        aggregations=["mean", "max"],
        include_expanding=True,
        expanding_aggregations=["mean"],
    ),
    calendar_config=CalendarFeatureConfig(features=["hour", "dayofweek", "is_weekend"]),
    static_cols=["zone_capacity"],
    known_future_cols=["planned_drivers"],
    historical_covariate_cols=["observed_queue"],
    regressor_params={
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_samples_leaf": 20,
    },
)
forecaster.fit(training_pickup_demand)

result = forecaster.predict(future_pickup_hours)
```

`future_pickup_hours` must contain the timestamp, panel columns, and every
`known_future_cols` value. It must not contain `historical_covariate_cols`, because
those columns are observations available only after the future timestamp occurs.

## Feature Families

- `LagFeatureConfig`: creates direct target lags such as `pickup_trips_lag_1`.
- `RollingFeatureConfig`: creates shifted rolling summaries such as
  `pickup_trips_roll_24_mean`; expanding summaries are also shifted.
- `CalendarFeatureConfig`: derives numeric timestamp fields such as hour, day of week,
  month, weekend, and optional holiday indicators.
- `static_cols`: copies panel attributes such as taxi zone capacity.
- `known_future_cols`: copies planned or scheduled inputs known at prediction time,
  such as planned drivers or airport event flags.

Holiday features are opt-in. Pass `CalendarFeatureConfig(features=["is_holiday"])` and a
`holiday_fn` callable that accepts a pandas `Timestamp` and returns `True` for holidays.

## Recursive Forecasts

`CartoBoostLagForecaster.predict()` forecasts rows in timestamp order. For multi-step
requests, each prediction is appended to that panel's history before later horizons are
featurized. This supports recursive forecasts while preserving panel isolation.

The returned `ForecastResult` includes:

- `frame`: a copy of the future frame with a `forecast` column.
- `predictions`: NumPy array in the original future-row order.
- `feature_names`: exact names used to fit the regressor.
- `regressor_metadata`: backend, native metadata, and fitted feature-name metadata.
