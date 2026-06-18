# Forecasting

CartoBoost forecasting is Rust-owned with thin Python wrappers. Python remains
responsible for ergonomic configuration, dataframe conversion, and documentation,
but forecasting model training and prediction come from `cartoboost._native`
bindings.

The public Python names with native training/prediction bindings are:

- `NaiveForecaster`
- `SeasonalNaiveForecaster`
- `ThetaForecaster`
- `OptimizedThetaForecaster`
- `CartoBoostLagForecaster`

Reserved names that are not exposed as Python fallback algorithms are:

- `ETSForecaster`
- `AutoARIMAForecaster`
- `WeightedEnsembleForecaster`

When the matching Rust/PyO3 class is unavailable, the wrapper raises
`NotImplementedError` with the missing native binding name. Python does not
compute fallback forecasts.

Forecast validation remains rolling-origin only. Random cross-validation is not
used for forecasting because it leaks future target information into training
folds.
