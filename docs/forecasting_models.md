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
- `auto_arima`
- `cartoboost_lag`
- `weighted_ensemble`

Implemented scope:

- `ets` is Rust additive ETS with optional additive seasonality.
- `auto_arima` is Rust AutoARIMA over bounded ARIMA(p,d,q) candidates.
- `weighted_ensemble` is a native PyO3 class that requires explicit native
  component models; it is not a zero-argument CLI/default-registry model.

Python does not provide fallback forecasting algorithms for unsupported modes.
Unsupported multiplicative ETS, damped ETS, and seasonal AutoARIMA fail
explicitly.

Example:

```python
from cartoboost.forecasting.local import SeasonalNaiveForecaster

model = SeasonalNaiveForecaster(season_length=24)
model.fit([42.0, 37.0, 51.0])
```

That `fit` call delegates to `cartoboost._native.SeasonalNaiveForecaster`.
