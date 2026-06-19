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
Kalman instead. The lag model is most useful when repeated structure across
zones or lanes gives the global regressor more examples than any one series can
provide.

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

## Runnable Visual Example

Run the committed example to create a forecast plot and a residual plot for a
synthetic taxi pickup-zone panel:

```bash
uv run python examples/forecasting/cartoboost_lag_visualization.py
```

It writes `target/examples/cartoboost_lag.png` and prints JSON with rows, zone
count, horizon, RMSE, and MAE. The example does not download data.

The core pattern is:

```python
import pandas as pd
from cartoboost.forecasting import CartoBoostLagForecaster

train = hourly_zone_demand.groupby("PULocationID", sort=False).head(72)

model = CartoBoostLagForecaster(
    time_col="pickup_hour",
    target_col="pickup_count",
    panel_cols=["PULocationID"],
    frequency="h",
    lags=[1, 2, 24],
    rolling_windows=[6, 24],
    calendar_features=True,
    trend_features=True,
    n_estimators=120,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=8,
    splitters=["axis", "periodic:24"],
)
model.fit(train)

forecast = pd.DataFrame(
    model.predict(12).predictions(),
    columns=["PULocationID", "pickup_hour", "horizon", "model", "prediction"],
)
forecast["pickup_hour"] = pd.to_datetime(forecast["pickup_hour"])
```

Plot one representative zone, then plot holdout residuals by horizon. Residuals
that drift in one direction usually mean the feature set is missing a level
shift or the recursive strategy is compounding short-horizon bias.

## Feature Families

| Feature family | Taxi interpretation | Common use |
| --- | --- | --- |
| `lags=[1, 2]` | Recent pickup pressure in the same zone or lane. | Short-horizon nowcasting. |
| `lags=[24, 168]` | Same hour yesterday or last week. | Daily and weekly demand cycles. |
| `rolling_windows=[6, 24]` | Recent moving average demand. | Smooth noisy pickup counts before tree splits. |
| `calendar_features=True` | Native day, month, and day-of-week features. | Let one global model separate weekday and weekend behavior. |
| `trend_features=True` | Lag deltas and rolling-trend features. | Capture ramps into commute peaks or airport surges. |
| `panel_cols=["PULocationID"]` | Series identity for each pickup zone. | Share tree structure while keeping panels isolated by history. |

Keep lags and rolling windows meaningful for the forecast frequency. For hourly
taxi data, `24` usually means same hour yesterday and `168` means same hour last
week. For daily aggregates, those same numbers mean very different behavior and
should be chosen deliberately.

## Splitter Choices

Use `["axis"]` for a basic supervised lag model. Add `periodic:24` when hourly
seasonality is present and the frame includes calendar or hour-cycle structure.
Keep geographic features in the supervised frame or use panel ids to let the
global model learn repeated zone patterns.

If a model with `periodic:24` wins on a taxi split, check that it wins on later
rolling origins too. A single split can overstate a periodic splitter when the
holdout happens to align with the same commute pattern.

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

## Interpreting Forecasts

Start by joining predictions back to the holdout on panel id and timestamp:

```python
joined = actual.merge(
    forecast,
    on=["PULocationID", "pickup_hour"],
    validate="one_to_one",
)
joined["error"] = joined["prediction"] - joined["pickup_count"]
joined["abs_error"] = joined["error"].abs()
```

Then inspect errors by horizon, hour of day, and zone:

```python
joined["hour"] = joined["pickup_hour"].dt.hour
print(joined.groupby("horizon")["abs_error"].mean())
print(joined.groupby("hour")["abs_error"].mean())
print(joined.groupby("PULocationID")["abs_error"].mean().sort_values(ascending=False))
```

Interpretation:

| Pattern | Meaning | Typical next step |
| --- | --- | --- |
| Error grows quickly with horizon. | Recursive forecasts are compounding bias. | Compare direct local models or reduce horizon for this surface. |
| Rush-hour residuals have one sign. | Calendar or trend features are not enough for commute peaks. | Add relevant lags, tune `periodic:24`, or include known future event features in the upstream frame. |
| One zone dominates MAE. | Shared model behavior is not matching that panel. | Check sparse history, missing zone metadata, or split that zone into its own validation slice. |
| Same-hour-yesterday beats the lag model. | The supervised model is overfitting or undertrained. | Increase data, simplify tree settings, and compare against seasonal naive before claiming improvement. |

## Validation Notes

Always use rolling-origin backtests. Lag and rolling features must be strictly
historical relative to the forecast row. Keep split boundaries stable when
comparing against seasonal naive, ARIMA, ETS, LightGBM, or XGBoost baselines.

For taxi-zone panels, report at least:

- split timestamps and horizon;
- number of zones or lanes;
- lag and rolling-window settings;
- `n_estimators`, `learning_rate`, `max_depth`, and `min_samples_leaf`;
- RMSE, MAE, and R2 when the comparison target is continuous;
- training and prediction time for benchmark claims.

Do not validate on randomly shuffled rows. A shuffled split leaks future pickup
patterns into training and makes lag features look stronger than they are.
