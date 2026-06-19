# Forecasting Models

CartoBoost forecasting models are owned by the Rust implementation. The Python
classes under `cartoboost.forecasting.local`,
`cartoboost.forecasting.global_models`, and `cartoboost.forecasting.ensemble`
validate parameters and delegate model execution to `cartoboost._native`.

The Forecasting V1 Python surface has native PyO3 training/prediction bindings
for these model names:

- `naive`
- `seasonal_naive`
- `theta`
- `optimized_theta`
- `ets`
- `arima`
- `auto_arima`
- `kalman`
- `kriging`
- `cartoboost_lag`
- `weighted_ensemble`

Implemented scope:

- `ets` is Rust additive ETS with optional additive seasonality.
- `arima` is Rust ARIMA(p,d,q) over bounded non-seasonal orders.
- `auto_arima` is Rust AutoARIMA over bounded ARIMA(p,d,q) candidates.
- `kalman` is a Rust local-linear-trend state-space model.
- `kriging` is a Rust ordinary-kriging panel forecaster over explicit series
  coordinates.
- `weighted_ensemble` is a native PyO3 class that requires explicit native
  component models; it is not a zero-argument CLI/default-registry model.

Python does not provide fallback forecasting algorithms for unsupported modes.
Unsupported multiplicative ETS, damped ETS, and seasonal AutoARIMA fail
explicitly.

General utilities now exposed outside `cartoboost.forecasting`:

- `local_level_kalman`: use `cartoboost.local_level_kalman_filter` or
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
- `mstl_ets`, `stl_arima`: no Rust/PyO3 decomposition-hybrid forecasting
  classes are exposed.
Example:

```python
from cartoboost.forecasting.local import KalmanForecaster, SeasonalNaiveForecaster

model = SeasonalNaiveForecaster(season_length=24)
model.fit([42.0, 37.0, 51.0])

kalman = KalmanForecaster()
kalman.fit([40.0, 42.0, 45.0, 47.0])
forecast = kalman.predict(2)
```

Those `fit` calls delegate to the corresponding `cartoboost._native`
forecasting classes.
