# Forecasting

CartoBoost forecasting is Rust-native with thin Python wrappers. Python remains
responsible for ergonomic configuration, dataframe conversion, CLI argument
validation, and documentation, but forecasting model training, prediction,
backtesting logic, metrics, artifacts, and leakage checks are owned by Rust
under `crates/`.

The public Python names with native PyO3 training/prediction bindings are:

- `NaiveForecaster`
- `SeasonalNaiveForecaster`
- `ThetaForecaster`
- `OptimizedThetaForecaster`
- `ETSForecaster`
- `AutoARIMAForecaster`
- `KalmanForecaster`
- `KrigingForecaster`
- `CartoBoostLagForecaster`
- `WeightedEnsembleForecaster`

Additional Rust-core forecasting behavior includes rolling-origin splitters,
forecast metrics, result serialization, artifact manifests, and leakage-safe
lag features. Python does not reimplement these behaviors.

Rust ETS is additive-only in this version. Rust AutoARIMA searches bounded
ARIMA(p,d,q) candidates with residual-lag moving-average terms; seasonal
AutoARIMA is rejected explicitly. Weighted ensembles require explicit native
component models. Python never computes fallback forecasts.

`KalmanForecaster` is an independent Rust local-linear state-space forecaster
for single-series or panel histories. `KrigingForecaster` is an independent
Rust ordinary-kriging forecaster over panel series and requires explicit
coordinates keyed by taxi-zone or lane series id.

Forecast validation remains rolling-origin only. Random cross-validation is not
used for forecasting because it leaks future target information into training
folds.
