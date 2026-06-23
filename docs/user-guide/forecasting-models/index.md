# Forecasting Model Guides

These guides explain the native forecasting model classes. Use this section
when you need to pick, configure, or compare a model. Use
[Forecasting](../../forecasting.md) when you need `ForecastFrame`,
rolling-origin backtesting, artifacts, CLI workflows, or shared evidence rules.

Every class is a thin Python wrapper over Rust behavior exposed through
`cartoboost._native`. Python does not compute fallback forecasts for unsupported
model modes.

## Pick A Guide

| Model guide | Best first use | Notes |
| --- | --- | --- |
| [Naive And Seasonal Naive](naive-seasonal.md) | Establish transparent last-value and last-season baselines. | Start here for every forecast comparison. |
| [Theta](theta.md) | Extrapolate level and trend with a lightweight deterministic model. | Includes manual and optimized theta examples. |
| [ETS](ets.md) | Model additive level, trend, and seasonality. | Multiplicative and damped ETS modes are rejected. |
| [ARIMA And AutoARIMA](arima.md) | Use differencing and autocorrelation in a bounded non-seasonal search. | Seasonal AutoARIMA is not exposed. |
| [ARIMA Examples](arima-examples.md) | Run a visual ARIMA smoke check and benchmark-oriented example. | Uses deterministic taxi-lane fixtures. |
| [Kalman](kalman.md) | Track noisy local level and local trend over time. | Includes state diagnostics and visualization examples. |
| [Piecewise Linear Seasonal](/docs/user-guide/forecasting-models/piecewise-linear-seasonal) | Fit interpretable trend, changepoint, seasonality, event, and regressor components without Stan. | Rust-native API exposed through Python and WASM as `piecewise_linear_seasonal`. |
| [Prophet-Compatible Forecast Plot](prophet.md) | Render Prophet-shaped forecast outputs with the same plotting contract as `prophet.plot.plot`. | Plotting parity only; CartoBoost does not expose a reusable `Prophet` model class. |
| [Kriging](kriging.md) | Borrow signal across pickup-zone or route coordinates. | Useful for coordinate-aware panel forecasting. |
| [CartoBoost Lag](cartoboost-lag.md) | Learn one supervised lag model across many related series. | Use for pickup-zone, dropoff-zone, and lane-level panels. |
| [AutoForecaster](auto-forecaster.md) | Use the guarded Rust-native default selector over lag, direct, residual-corrected, intermittent, and classical candidates. | Includes diagrams for validation, gating, prediction, and metadata inspection. |
| [Weighted Ensembles](ensembles.md) | Combine fitted native forecasters with explicit weights. | Components and weights must be named explicitly. |

## Scientific Choice Criteria

Choose the model whose assumptions match the signal you can defend:

| Signal in the taxi series | First model to try | Scientific reason |
| --- | --- | --- |
| The latest observed level is the best short-horizon summary. | Naive | Tests whether any model adds information beyond persistence. |
| The same hour yesterday or same weekday last week dominates. | Seasonal naive | Tests repeatable seasonality without estimated parameters. |
| Level and trend are smooth, with optional simple seasonality. | Theta or ETS | Estimates a low-dimensional local structure that is easy to inspect. |
| Recent autocorrelation and differencing explain the series. | ARIMA or AutoARIMA | Models local serial dependence after bounded non-seasonal differencing. |
| The measured series is noisy and the latent level/trend should update gradually. | Kalman | Separates observation noise from latent state movement. |
| You need interpretable changepoints, Fourier seasonalities, event windows, known future regressors, quantiles, and component decomposition in one local model. | [Piecewise linear seasonal](/docs/user-guide/forecasting-models/piecewise-linear-seasonal) | Estimates the additive or multiplicative component path in Rust, keeps fitting deterministic and fast, and avoids Stan/CmdStan runtime costs. |
| You need a forecast figure that matches Prophet's plotting surface for a Prophet-shaped result. | [Prophet-compatible forecast plot](prophet.md) | Uses the same observed-point, forecast-line, capacity, floor, interval, axis, and legend behavior as `prophet.plot.plot`. |
| Nearby zones, route midpoints, or residual surfaces should be spatially related. | Kriging | Uses coordinate distance and a variogram to borrow cross-series signal. |
| Many related zones or lanes share lag, rolling, calendar, or trend structure. | CartoBoost lag | Learns one supervised model from many aligned panel examples. |
| A production taxi-demand panel needs a deterministic guarded default with auditable candidate weights. | AutoForecaster | Validates a fixed Rust-native roster, protects the lag baseline, and stores global, horizon, and series weights. |
| Validated models capture complementary errors. | Weighted ensemble | Averages explicit native components after each member proves useful. |

Do not choose a richer model only because it is available. A scientist should
be able to say which mechanism the model represents, what it ignores, and which
baseline threshold it must clear on a time-ordered holdout.

## Shared Input Patterns

For quick checks, local forecasters can fit a plain numeric sequence:

```python
from cartoboost.forecasting import SeasonalNaiveForecaster

model = SeasonalNaiveForecaster(season_length=24)
model.fit(zone_hourly_counts)
forecast = model.predict(12)
```

For production taxi demand or fare-duration workflows, prefer a validated
`ForecastFrame`:

```python
from cartoboost.forecasting import ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="PULocationID",
    freq="h",
)
```

`ForecastFrame` validates timestamps, duplicate rows within each series, finite
targets, regular frequency, panel ids, and covariate role metadata.

## Advanced Native Candidates

`AutoForecaster` can validate internal direct, rectified-recursive,
intermittent-demand, classical-expert, and decomposition-style candidates when
those candidates are exposed by the compiled Rust core. Treat those as roster
members of the guarded default selector unless a separate Python class is part
of the public API. Keep benchmark-specific names and competition scoring labels
in benchmark harnesses and reports.

Use [Piecewise Linear Seasonal](/docs/user-guide/forecasting-models/piecewise-linear-seasonal) when the forecast
claim depends on inspectable local structure: growth, changepoints, Fourier
seasonalities, event windows, known future regressors, uncertainty intervals,
quantiles, trend-belief adjustments, residual shock propagation, and component
contributions. The implementation is Rust-native and is also exposed to the
browser through the `piecewise_linear_seasonal` WASM model. There is no
`prophet` alias in reusable CartoBoost APIs.

## Shared Result Shape

Native forecasting models return a `ForecastResult` object. Use
`predictions()` for row tuples:

```python
forecast = model.predict(3)
rows = forecast.predictions()

for series_id, timestamp, horizon, model_name, mean in rows:
    print(series_id, timestamp, horizon, model_name, mean)
```

The tuple columns are also available from `forecast.columns()`. Use
`forecast.to_json()` and `cartoboost._native.ForecastResult.from_json(...)` for
native JSON roundtrips.

## Validation Order

For forecast claims, compare models under the same rolling-origin split:

1. Start with naive and seasonal naive baselines.
2. Add a local model that matches the series structure, such as theta, ETS,
   ARIMA, or Kalman.
3. Use `CartoBoostLagForecaster` when many related series should share lag,
   rolling, calendar, or trend features.
4. Use kriging when stable coordinates are part of the forecast signal.
5. Use weighted ensembles only after component models have been validated.

Report RMSE, MAE, horizon, split dates, training time, prediction time, model
settings, sample size, and whether the input data is real, generated acceptance
data, or synthetic.
