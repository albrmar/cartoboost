# Forecasting Models

CartoBoost forecasting models are owned by the Rust implementation. The Python
classes under `cartoboost.forecasting.local`, `cartoboost.forecasting.global_models`,
and `cartoboost.forecasting.ensemble` are thin wrappers over `cartoboost._native`.

The Forecasting V1 Python surface exposes these Rust-backed model names:

- `naive`
- `seasonal_naive`
- `theta`
- `optimized_theta`
- `cartoboost_lag`

These names are reserved for Rust implementations and do not have Python
fallback algorithms:

- `ets`
- `auto_arima`
- `weighted_ensemble`

When a Rust/PyO3 binding is not available yet, fitting or predicting with the
corresponding Python wrapper raises `NotImplementedError` with the missing native
class name. Python does not provide fallback forecasting algorithms for these
models.

Example:

```python
from cartoboost.forecasting.local import SeasonalNaiveForecaster

model = SeasonalNaiveForecaster(season_length=24)
model.fit([42.0, 37.0, 51.0])
```

That `fit` call delegates to `cartoboost._native.SeasonalNaiveForecaster`.
