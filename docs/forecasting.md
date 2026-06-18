# Forecasting

CartoBoost forecasting turns timestamped geographic series into deterministic
forecast tables, rolling-origin backtests, lag-supervised CartoBoost models,
ensembles, and portable artifacts. The default examples use NYC taxi-style
pickup/dropoff lanes, pickup zones, daily or hourly demand, fares, durations,
trip distance, and calendar features.

The public workflow is:

```python
from cartoboost.forecasting import ForecastFrame, ThetaForecaster

frame = ForecastFrame.from_pandas(
    df,
    timestamp_col="date",
    target_col="loads",
    series_id_col="lane_id",
    freq="D",
)

model = ThetaForecaster(season_length=7, prediction_interval_levels=[0.8, 0.95])
model.fit(frame)
forecast = model.predict(horizon=14)
```

`forecast.to_pandas()` uses stable columns:

```text
series_id, timestamp, horizon, model, mean, lower_80, upper_80, lower_95, upper_95
```

Single-series forecasts use `__single__` as the exported `series_id`.

## Method Selection

| Use case                          | Default method                      | Backup method                  |
| --------------------------------- | ----------------------------------- | ------------------------------ |
| Very short series                 | Naive / Seasonal Naive              | Theta if enough history        |
| Short zone or lane series         | Theta                               | ETS                            |
| Trended univariate series         | Optimized Theta                     | ETS damped trend               |
| Strong weekly seasonality         | Seasonal Naive + Theta seasonal     | ETS seasonal                   |
| Many related pickup/dropoff lanes | CartoBoostLagForecaster             | Weighted ensemble              |
| Rich known-future covariates      | CartoBoostLagForecaster             | ETS/ARIMA baseline comparison  |
| Need robust production baseline   | WeightedEnsembleForecaster          | Best backtested local model    |
| Need uncertainty intervals        | Theta/ETS with calibrated intervals | conformal residual calibration |

Forecast validation is rolling-origin only. Random cross-validation is not used
for forecasting because it leaks future target information into training folds.
