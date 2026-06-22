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
- external trend belief adjustments for forecast horizons;
- recent residual shock propagation when the fitted trend has under- or
  over-predicted for several consecutive observations;
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
    changepoints=8,
    weekly_fourier_order=3,
    daily_fourier_order=4,
    prediction_interval_levels=(0.8, 0.95),
)

forecast = model.fit(frame).predict(24)
components = model.components(24)
```

The Python class validates configuration and delegates fitting, prediction,
component extraction, fitted artifact serialization, and forecast rows to the
native binding. Python should not reimplement component math or fallback
prediction behavior.

## Prophet-Style Plotting

The local plotting layer provides full public plotting-utility parity with the
`prophet.plot` module from `prophet==1.2.2`, the package version resolved by
CartoBoost's benchmark dependency range `prophet>=1.1,<1.3`. Use
`cartoboost.plotting.plot`, `plot_components`, `plot_forecast_component`,
`plot_weekly`, `plot_yearly`, `plot_seasonality`,
`add_changepoints_to_plot`, `plot_cross_validation_metric`, the matching
Plotly helpers, and the helper-prop functions when a report expects Prophet's
plotting API names.

The proof is maintained in [Plotting](../../plotting.md): it lists every public
`prophet.plot` 1.2.2 utility beside the matching local implementation and
points to the parity tests in `tests/python/test_plotting.py`. This parity is
limited to plotting utilities over Prophet-shaped forecast/component data; the
reusable model API remains `PiecewiseLinearSeasonalForecaster`, not a
`prophet` alias.

## Trend Beliefs And Residual Shocks

Use `trend_adjustments` when forecast-time market beliefs should move the local
trend path without changing the fitted historical coefficients. The mapping is
keyed by forecast horizon, with values interpreted as trend multipliers. For
example, `{1: 1.01, 2: 1.02}` raises the horizon-1 trend by 1 percent and the
horizon-2 trend by 2 percent before the final forecast is assembled. Panel
models can use `trend_adjustments_by_series` for route- or zone-specific
beliefs; per-series values override global horizon values.

```python
model = PiecewiseLinearSeasonalForecaster(
    changepoints=8,
    weekly_fourier_order=3,
    trend_adjustments={1: 1.01, 2: 1.02, 3: 1.03},
    trend_adjustments_by_series={
        "132": {1: 1.04, 2: 1.05},
    },
)
```

Use residual shocks when recent same-sign residuals indicate that the local
trend is persistently under- or over-predicting a market. Set
`residual_shock_window` to the required run length, `residual_shock_scale` to
the fraction of the recent average residual to pass through, and
`residual_shock_decay` to control how quickly the shock fades across horizons.
The default scale is zero, so shock propagation is opt-in.

```python
model = PiecewiseLinearSeasonalForecaster(
    changepoints=8,
    weekly_fourier_order=3,
    residual_shock_window=3,
    residual_shock_scale=0.5,
    residual_shock_decay=0.8,
)
```

Component records include `trend`, `adjusted_trend`,
`trend_adjustment_multiplier`, `trend_adjustment`, and `residual_shock` so a
taxi demand report can separate fitted trend, external market belief, and
recent residual carry-forward.
