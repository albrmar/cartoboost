# Forecasting Models

CartoBoost forecasting models are native Rust implementations exposed through
thin Python wrappers. Use this page as the compact implementation map. For
model-selection guidance, scientific assumptions, failure modes, taxi examples,
and validation patterns, use the standalone
[forecasting model guides](user-guide/forecasting-models/index.md).

Python classes under `cartoboost.forecasting.local`,
`cartoboost.forecasting.global_models`, and `cartoboost.forecasting.ensemble`
validate parameters and delegate model execution to `cartoboost._native`.
Python does not provide fallback forecasting algorithms for unsupported modes.

The Forecasting V1 Python surface has native PyO3 training/prediction bindings
for these model names:

- `naive`
- `seasonal_naive`
- `theta`
- `optimized_theta`
- `ets`
- `arima`
- `auto_arima`
- `local_level_kalman`
- `kalman`
- `auto_local_level_kalman`
- `auto_kalman`
- `kriging`
- `cartoboost_lag`
- `cartoboost_auto_forecast`
- `weighted_ensemble`
- `stl_cartoboost`
- `mstl_cartoboost`
- `classical_expert_bank`
- `autostats_bank`

## Implemented Scope

| Native model | Guide | Implemented scope | Choose scientifically when |
| --- | --- | --- | --- |
| `naive` | [Naive And Seasonal Naive](user-guide/forecasting-models/naive-seasonal.md) | Repeats the latest observed value. | Persistence is the control hypothesis. |
| `seasonal_naive` | [Naive And Seasonal Naive](user-guide/forecasting-models/naive-seasonal.md) | Repeats values from the latest completed season. | Same-hour or same-weekday repetition is the control hypothesis. |
| `theta` | [Theta](user-guide/forecasting-models/theta.md) | Manual theta and smoothing parameters, with optional seasonal adjustment. | Level and trend extrapolation should explain the next horizons. |
| `optimized_theta` | [Theta](user-guide/forecasting-models/theta.md) | Deterministic native grid search over theta and alpha candidates. | A small reproducible trend grid is preferable to manual settings. |
| `ets` | [ETS](user-guide/forecasting-models/ets.md) | Additive ETS with optional additive seasonality. | Smoothed level, additive trend, and additive seasonal state are interpretable. |
| `arima` | [ARIMA And AutoARIMA](user-guide/forecasting-models/arima.md) | ARIMA(p,d,q) over bounded non-seasonal orders. | Recent autocorrelation and differencing explain one local series. |
| `auto_arima` | [ARIMA And AutoARIMA](user-guide/forecasting-models/arima.md) | Deterministic bounded search over non-seasonal ARIMA(p,d,q) candidates. | You need reproducible local order search before held-out validation. |
| `local_level_kalman` | [Kalman](user-guide/forecasting-models/kalman.md) | Local-level state-space model. | Observations are noisy measurements of one latent level. |
| `kalman` | [Kalman](user-guide/forecasting-models/kalman.md) | Local-linear-trend state-space model. | Observations are noisy measurements of latent level and trend. |
| `auto_local_level_kalman` | [Kalman](user-guide/forecasting-models/kalman.md) | Deterministic native grid search over local-level variance candidates. | A small reproducible variance grid should replace manual level/noise settings. |
| `auto_kalman` | [Kalman](user-guide/forecasting-models/kalman.md) | Deterministic native grid search over local-linear variance candidates. | A small reproducible variance grid should replace manual level/trend/noise settings. |
| `kriging` | [Kriging](user-guide/forecasting-models/kriging.md) | Ordinary-kriging panel forecaster over explicit series coordinates. | Nearby zones or route coordinates should borrow spatial signal. |
| `cartoboost_lag` | [CartoBoost Lag](user-guide/forecasting-models/cartoboost-lag.md) | Supervised native lag, rolling, calendar, trend, and CartoBoost regressor workflow. | Many related panels should share one leakage-safe lag model. |
| `cartoboost_auto_forecast` | [Forecasting Overhaul](forecasting_overhaul.md) | Deterministic benchmark alias for the hybrid AutoForecaster: AutoStats, direct CartoBoost, decomposition, intermittent, probabilistic, reconciliation, neural, and ensemble branches are rule-routed when inputs support them. | You want the default no-hyperopt CartoBoost forecasting contender for committed M4/M5/M6-style benchmarks. |
| `weighted_ensemble` | [Weighted Ensembles](user-guide/forecasting-models/ensembles.md) | Native PyO3 class requiring explicit native component models and weights. | Validated components make complementary errors under the same split. |
| `stl_cartoboost` | [Forecasting Decomposition](forecasting_decomposition.md) | Native additive STL decomposition with a native remainder forecaster and deterministic recomposition. | One dominant taxi seasonality should be separated before modeling residual autocorrelation. |
| `mstl_cartoboost` | [Forecasting Decomposition](forecasting_decomposition.md) | Native additive MSTL decomposition over multiple season lengths with a native remainder forecaster and deterministic recomposition. | Hour-of-day and day-of-week taxi patterns should be separated before residual modeling. |
| `classical_expert_bank` | This page | Native deterministic validation over a configured bank of classical local forecasters, then refit of the selected expert. | A small auditable roster should choose among naive, seasonal naive, theta, ETS, ARIMA, and Kalman assumptions. |
| `autostats_bank` | This page | Native default classical bank using naive, seasonal naive, theta, ETS, AutoARIMA, and Kalman variants. | You need a reproducible classical baseline selector before comparing a learned global taxi model. |

Unsupported multiplicative ETS, damped ETS, and seasonal AutoARIMA fail
explicitly.

`weighted_ensemble` is not a zero-argument CLI/default-registry model. Its
component models and weights must be named explicitly.

## Validation Expectations

For forecasting claims, use time-ordered or rolling-origin splits with the same
train/test rows across candidates. Start with naive and seasonal naive
baselines, add the local or global model whose assumptions match the signal,
and report RMSE, MAE, horizon, split dates, training time, prediction time,
model settings, sample size, and whether the data is real taxi data, generated
acceptance data, or synthetic.

Do not treat deterministic examples, generated visualization fixtures, or
Criterion speed benchmarks as real quality evidence. They are useful for API,
plotting, and implementation smoke checks.

General utilities also exposed outside `cartoboost.forecasting`:

- `local_level_kalman`: use `LocalLevelKalmanForecaster`,
  `cartoboost.local_level_kalman_filter`, or
  `cartoboost.local_level_kalman_forecast`.
- `local_linear_trend_kalman`: use `cartoboost.kalman_filter` or
  `cartoboost.local_linear_trend_kalman_forecast`.
- `croston`, `sba`, `tsb`: use `cartoboost.croston_forecast`,
  `cartoboost.sba_forecast`, or `cartoboost.tsb_forecast`.
- `ordinary_kriging`: use `cartoboost.ordinary_kriging_predict`.

For usage examples, see [General Utilities](general_utilities.md).

Native forecasting-class surface still required before frame-based Python
wrappers can be exposed:

- `unobserved_components`: no Rust/PyO3 forecasting class is exposed.
- `sarimax`: no Rust/PyO3 forecasting class is exposed.
- `dynamic_regression`: no Rust/PyO3 forecasting class is exposed.
- Specialized `mstl_ets` and `stl_arima` Python wrapper names are not exposed;
  use the native `stl_cartoboost` and `mstl_cartoboost` decomposition-hybrid
  forecasters where bindings are available.

Example:

```python
from cartoboost.forecasting.local import AutoKalmanForecaster, SeasonalNaiveForecaster

model = SeasonalNaiveForecaster(season_length=24)
model.fit([42.0, 37.0, 51.0])

kalman = AutoKalmanForecaster(validation_window=2)
kalman.fit([40.0, 42.0, 45.0, 47.0])
forecast = kalman.predict(2)
```

Those `fit` calls delegate to the corresponding `cartoboost._native`
forecasting classes.
