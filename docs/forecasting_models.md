# Forecasting Models

CartoBoost forecasting V1 includes local statistical baselines for taxi demand,
fare, duration, and zone-level time series. The models live under
`cartoboost.forecasting.local` and support a single series or an equal-length
panel where each column is a pickup zone, dropoff zone, or route-level series.

```python
from cartoboost.forecasting.local import SeasonalNaiveForecaster

pickup_demand_by_zone = {
    "PULocationID=161": [42, 37, 51, 45, 39, 55],
    "PULocationID=236": [31, 29, 35, 33, 30, 38],
}

model = SeasonalNaiveForecaster(season_length=3).fit(pickup_demand_by_zone)
forecast = model.predict(3, return_interval=True, level=0.9)
```

All local forecasters raise a fitted-before-predict error, return deterministic
output, expose `fitted_values_`, `residuals_`, and `metadata_` after fitting, and
can return residual-normal prediction intervals through `return_interval=True`.
When timestamps are passed to `fit`, forecast results include projected future
timestamps based on the last observed step.

## Baselines

`NaiveForecaster` repeats the last observed value for each series. It is useful
as the minimum acceptable baseline for pickup demand or fare forecasts.

`SeasonalNaiveForecaster` repeats the last observed seasonal cycle. Set
`season_length` to the taxi cadence being modeled, such as 24 for hourly
same-hour-of-day demand or 7 for daily same-day-of-week demand.

## Theta

`ThetaForecaster` fits a deterministic theta method per series. It supports:

- `theta`: positive theta coefficient.
- `alpha`: simple exponential smoothing level in `(0, 1]`.
- `seasonality="additive"` with `season_length`.
- `seasonality="multiplicative"` with `season_length`; all training values must
  be strictly positive.

Seasonal theta deseasonalizes each panel series, fits the theta trend and level,
then reseasonalizes fitted values and forecasts. `OptimizedThetaForecaster`
performs deterministic in-sample grid validation over `theta_grid` and
`alpha_grid`, records `validation_scores_`, and stores the selected parameters in
`metadata_`.

## Optional Models

`ETSForecaster` uses `statsmodels.tsa.holtwinters.ExponentialSmoothing` lazily.
If statsmodels is not installed, fitting raises:

```text
ETSForecaster requires statsmodels. Install it with `pip install statsmodels`.
```

ETS intervals use CartoBoost's residual-normal fallback around the statsmodels
point forecast.

`AutoARIMAForecaster` uses `pmdarima.auto_arima` lazily. By default,
`error_policy="raise"` raises a clear install or model-fitting error. With
`error_policy="fallback"`, missing pmdarima or backend fitting failures fall back
to `NaiveForecaster` and record `backend="naive_fallback"` plus
`fallback_reason` in `metadata_`.
