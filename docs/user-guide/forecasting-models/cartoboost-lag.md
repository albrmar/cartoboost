# CartoBoost Lag

`CartoBoostLagForecaster` is the global supervised forecasting model. It builds
leakage-safe lag, rolling, calendar, and trend features in Rust, then fits a
CartoBoost regressor to forecast future horizons.

## When To Use

Use it when many related taxi time series should share one model:

- pickup-zone hourly demand;
- dropoff-zone hourly demand;
- airport lane demand;
- pickup-dropoff route counts;
- zone-level fare or duration aggregates.

For a single short series, start with seasonal naive, theta, ETS, ARIMA, or
Kalman instead.

## DataFrame Example

```python
from cartoboost.forecasting import CartoBoostLagForecaster

model = CartoBoostLagForecaster(
    time_col="pickup_hour",
    target_col="pickup_count",
    panel_cols=["PULocationID"],
    frequency="h",
    lags=[1, 2, 24, 168],
    rolling_windows=[24, 168],
    calendar_features=True,
    trend_features=True,
    recursive=True,
    target_mode="level",
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=20,
    min_gain=1e-8,
    splitters=["axis", "periodic:24"],
)

model.fit(hourly_zone_demand)
forecast = model.predict(24)
```

The dataframe must contain `time_col`, `target_col`, and all `panel_cols`.
Timestamps must be unique within each panel. The wrapper sorts by panel columns
and timestamp before building the native frame.

## Config Object Example

```python
from cartoboost.forecasting import (
    CalendarFeatureConfig,
    CartoBoostLagForecaster,
    LagFeatureConfig,
    RollingFeatureConfig,
)

model = CartoBoostLagForecaster(
    time_col="pickup_hour",
    target_col="pickup_count",
    panel_cols=["PULocationID"],
    frequency="h",
    lag_config=LagFeatureConfig(lags=[1, 24, 168]),
    rolling_config=RollingFeatureConfig(windows=[24, 168], aggregations=("mean",)),
    calendar_config=CalendarFeatureConfig(features=("dayofweek", "month", "day")),
    regressor_params={
        "n_estimators": 200,
        "learning_rate": 0.04,
        "max_depth": 5,
        "min_samples_leaf": 20,
        "splitters": ["axis", "periodic:24"],
    },
)
model.fit(hourly_zone_demand)
forecast = model.predict(24)
```

## Native Feature Support

| Option | Native support |
| --- | --- |
| `lags` | Positive historical lag offsets. |
| `rolling_windows` | Complete rolling mean windows. |
| `calendar_features=True` | Native calendar features. |
| `CalendarFeatureConfig` | Supports `dayofweek`, `month`, and `day`. |
| `trend_features=True` | Enables native lag-delta and rolling-trend features. |
| `target_mode` | Supports native target-mode handling. |
| `regressor_params` | Supports `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `min_gain`, and `splitters`. |

Unsupported feature-builder options fail explicitly rather than being silently
ignored.

## Splitter Choices

Use `["axis"]` for a basic supervised lag model. Add `periodic:24` when hour
features are present. Keep geographic features in the supervised frame or use
panel ids to let the global model learn repeated zone patterns.

## Validation Notes

Always use rolling-origin backtests. Lag and rolling features must be strictly
historical relative to the forecast row. Keep split boundaries stable when
comparing against seasonal naive, ARIMA, ETS, LightGBM, or XGBoost baselines.
