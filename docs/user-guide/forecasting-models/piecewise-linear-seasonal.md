# Piecewise Linear Seasonal

`PiecewiseLinearSeasonalForecaster` is the Rust-native local model for
interpretable trend, changepoints, seasonalities, events, extra regressors, and
component decomposition. It gives CartoBoost a Prophet-style modeling surface
without making Stan, CmdStan, or Python fitting code part of the forecasting
contract.

## When To Use

Use this model for one regular taxi demand, fare, duration, or trip-distance
series when the forecast should be explained as:

- a piecewise trend with optional changepoints;
- one or more repeating seasonal components;
- known event windows or known future regressors;
- additive or multiplicative component effects;
- residual intervals or quantile-style uncertainty summaries.

It is a good fit for airport pickup demand with event windows, Midtown demand
with recurring weekday and hour effects, or lane-level aggregate demand where a
scientist needs to inspect the trend and seasonal contribution separately.

Prefer seasonal naive, theta, ETS, ARIMA, or Kalman when the question is only a
small local baseline. Prefer `CartoBoostLagForecaster` or `AutoForecaster` when
many related zones or lanes should share supervised lag features.

## Scientific Role

Piecewise linear seasonal forecasting is a component model. It asks whether the
future can be explained by a trend path plus named seasonal, event, and
regressor effects. That makes it useful when the model output must support a
claim such as "the weekday seasonality explains the recurring lift, while the
trend changed after this cutoff."

The component rows are part of the evidence. If a benchmark or report uses this
model, preserve the forecast table, component decomposition, changepoints,
input cadence, regressors, and validation split with the metrics.

## Assumptions And Failure Modes

The model assumes the configured seasonal periods and event/regressor columns
match the forecast cadence and are known at prediction time. It can fail when
the strongest signal is cross-zone borrowing, sparse intermittent demand,
unmodeled disruptions, or a supervised panel effect that a local component model
cannot see.

Common failure modes in taxi data:

| Failure mode | Scientific interpretation | Comparison to run |
| --- | --- | --- |
| Trend changes too often. | Changepoint flexibility is fitting noise. | ETS, Kalman, or fewer changepoints. |
| Seasonal component is phase-shifted. | Season length or timestamp cadence is wrong. | Seasonal naive on the same cadence. |
| Event effect persists outside its window. | The event window is standing in for missing covariates. | Lag model with explicit calendar features. |
| One lane fits and another fails. | Local components do not transfer across panels. | CartoBoost lag or AutoForecaster. |

## Example

```python
from cartoboost.forecasting import ForecastFrame, PiecewiseLinearSeasonalForecaster

frame = ForecastFrame.from_pandas(
    hourly_airport_pickups,
    timestamp_col="pickup_hour",
    target_col="pickup_trips",
    series_id_col="PULocationID",
    freq="h",
    known_future_covariates=["hour", "day_of_week", "holiday_event"],
)

model = PiecewiseLinearSeasonalForecaster(
    growth="linear",
    season_length=24,
    changepoint_count=8,
    prediction_interval_levels=(0.8, 0.95),
)

forecast = model.fit(frame).predict(24)
components = model.components()
```

The Python class validates configuration and delegates fitting, prediction,
component extraction, fitted artifact serialization, and forecast rows to the
native binding. Python should not reimplement component math or fallback
prediction behavior.
