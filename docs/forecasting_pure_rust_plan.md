# Pure Rust Forecasting Plan

This page supersedes the Python-first Forecasting V1 implementation plan when
pure Rust is required. Under this constraint, Rust is the forecasting engine.
Python may expose bindings, examples, and ergonomic wrappers, but Python must
not contain the core forecasting logic.

Model quality and implementation tests are specified separately in
[Forecasting Model Quality And Implementation Tests](forecasting_quality_tests.md).
That page is part of the acceptance contract for this plan.

This means:

- no `statsmodels`
- no `pmdarima`
- no Python-first model implementations
- no dependency on pandas for core forecasting
- no Python-only CLI logic
- Python wrappers call Rust through PyO3 and maturin

Python wrappers may convert pandas DataFrames into Rust-owned forecast frames,
but validation, fitting, prediction, backtesting, metrics, artifacts, intervals,
conformal calibration, reconciliation, and CLI execution must run in Rust.

## Required Rust Layout

```text
crates/cartoboost-forecasting/
  Cargo.toml
  src/
    lib.rs
    schema.rs
    frequency.rs
    traits.rs
    metrics.rs
    splitters.rs
    backtesting.rs
    artifacts.rs
    registry.rs
    ensemble.rs
    conformal.rs
    reconciliation.rs
    local/
      mod.rs
      naive.rs
      seasonal_naive.rs
      theta.rs
      ets.rs
      arima.rs
      kalman.rs
      state_space.rs
      intermittent.rs
      decomposition.rs
    global/
      mod.rs
      lag_features.rs
      cartoboost_lag.rs
      quantile_carto_boost_lag.rs
    cli.rs
```

Python wrappers should be limited to:

```text
python/cartoboost/forecasting/
  __init__.py
  bindings.py
  wrappers.py
```

## Rust API Target

```rust
pub struct ForecastFrame;
pub struct ForecastResult;
pub struct ForecastArtifact;
pub struct RollingOriginBacktester;

pub trait Forecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<(), ForecastError>;

    fn predict(
        &self,
        horizon: usize,
        future_frame: Option<&ForecastFrame>,
    ) -> Result<ForecastResult, ForecastError>;

    fn model_name(&self) -> &'static str;
    fn metadata(&self) -> serde_json::Value;
}
```

Required Rust model structs:

```rust
pub struct NaiveForecaster;
pub struct SeasonalNaiveForecaster;
pub struct ThetaForecaster;
pub struct OptimizedThetaForecaster;
pub struct EtsForecaster;
pub struct ArimaForecaster;
pub struct SarimaxForecaster;
pub struct LocalLevelKalmanForecaster;
pub struct LocalLinearTrendKalmanForecaster;
pub struct CrostonForecaster;
pub struct SbaForecaster;
pub struct TsbForecaster;
pub struct MstlEtsForecaster;
pub struct StlArimaForecaster;
pub struct CartoBoostLagForecaster;
pub struct QuantileCartoBoostLagForecaster;
pub struct ConformalForecaster;
pub struct WeightedEnsembleForecaster;
```

Only fully implemented models may be public or registered.

## Forbidden In Core Forecasting

```text
Python model logic
Python statistical libraries
FFI calls into statsmodels or pmdarima
shelling out to Python
placeholder wrappers
feature flags that hide unimplemented models
public registry entries for missing models
```

Allowed:

```text
Rust-native linear algebra
Rust-native optimization
Rust-native serialization
Rust-native CSV, Parquet, and Arrow IO
Rust-native CLI
Python bindings over Rust
optional feature-gated Python wrappers
```

## Revised Agent Split

Use six implementation agents, but make all core ownership Rust-first.

## Head Agent

Owns:

```text
crates/cartoboost-forecasting/Cargo.toml
crates/cartoboost-forecasting/src/lib.rs
scripts/forecasting_benchmark.rs or examples/forecasting_benchmark.rs
README.md
CHANGELOG.md
mkdocs.yml
```

Only the head agent may run:

```bash
cargo test
cargo clippy --all-targets --all-features
cargo fmt --check
cargo bench
uv run pytest
uv run maturin develop
uv run mkdocs build
```

The head agent must enforce:

- core forecasting is Rust
- Python is wrapper-only
- all registry entries map to real Rust implementations
- no Python-only forecasting dependency is introduced
- all public models have Rust tests
- all Python wrappers call Rust

## Agent 1: Schema, Frequency, Traits, Errors

Owns:

```text
crates/cartoboost-forecasting/src/schema.rs
crates/cartoboost-forecasting/src/frequency.rs
crates/cartoboost-forecasting/src/traits.rs
crates/cartoboost-forecasting/src/error.rs
crates/cartoboost-forecasting/src/lib.rs
crates/cartoboost-forecasting/tests/schema_tests.rs
crates/cartoboost-forecasting/tests/frequency_tests.rs
docs/forecasting_api.md
```

Implement:

```rust
ForecastFrame
ForecastResult
ForecastRow
ForecastInterval
ForecastQuantiles
ForecastMetadata
ForecastError
Forecaster
```

`ForecastFrame` must support single-series data, panel data, timestamp and
target columns, optional series id, static covariates, known-future covariates,
historical-only covariates, hierarchy columns, explicit or inferred frequency,
missing timestamp policies, duplicate detection, deterministic sorting, finite
target validation, and strict panel isolation.

Missing timestamp policies:

```text
Reject
FillZero
FillMissing
AsObserved
```

`ForecastResult` must support deterministic ordering, mean forecasts, intervals,
quantiles, metadata, JSON round trip, CSV export, and optional Parquet export
behind a feature flag.

## Agent 2: Backtesting, Splitters, Metrics

Owns:

```text
crates/cartoboost-forecasting/src/splitters.rs
crates/cartoboost-forecasting/src/metrics.rs
crates/cartoboost-forecasting/src/backtesting.rs
crates/cartoboost-forecasting/tests/splitter_tests.rs
crates/cartoboost-forecasting/tests/metric_tests.rs
crates/cartoboost-forecasting/tests/backtesting_tests.rs
docs/forecasting_backtesting.md
```

Implement rolling-origin, expanding-window, and sliding-window splitters;
rolling-origin backtesting; fold results; and metric sets.

Metrics must include MAE, RMSE, zero-safe MAPE, sMAPE, MASE, RMSSE, WAPE, bias,
pinball loss, interval coverage, mean interval width, MSIS, interval score,
hierarchical coherence error, and zero-demand precision/recall/F1.

Backtesting must align by:

```text
series_id
timestamp
horizon
```

It must never align by row order alone.

Hard leakage invariant:

```text
max(train_timestamp) < min(validation_timestamp)
```

No random cross-validation is allowed.

## Agent 3: Local Statistical Models

Owns:

```text
crates/cartoboost-forecasting/src/local/naive.rs
crates/cartoboost-forecasting/src/local/seasonal_naive.rs
crates/cartoboost-forecasting/src/local/theta.rs
crates/cartoboost-forecasting/src/local/ets.rs
crates/cartoboost-forecasting/src/local/arima.rs
crates/cartoboost-forecasting/src/local/kalman.rs
crates/cartoboost-forecasting/src/local/state_space.rs
crates/cartoboost-forecasting/src/local/intermittent.rs
crates/cartoboost-forecasting/src/local/decomposition.rs
crates/cartoboost-forecasting/src/local/mod.rs
crates/cartoboost-forecasting/tests/local_model_tests.rs
docs/forecasting_models.md
docs/forecasting_state_space.md
docs/forecasting_intermittent.md
docs/forecasting_decomposition.md
```

Implement Rust-native naive, seasonal naive, theta, optimized theta, ETS, ARIMA,
SARIMAX, local-level Kalman, local-linear-trend Kalman, Croston, SBA, TSB, STL
plus ETS, and STL plus ARIMA.

ETS must be Rust-native. Minimum supported forms:

```text
ANN
AAN
AAdN damped trend
ANA additive seasonality
AAA additive trend plus additive seasonality
```

ARIMA/SARIMAX must be Rust-native. If full MA optimization is too large, expose
only implemented order families.

Kalman models must expose filtered state, state covariance, forecast mean,
forecast variance, and prediction intervals. Smoothers may be exposed only when
implemented.

Intermittent demand models must reject negative demand, handle all-zero series
explicitly, support sparse freight-lane demand, and fit panel series
independently.

MSTL must not be public until fully implemented.

## Agent 4: Lag Features And CartoBoost Forecasting

Owns:

```text
crates/cartoboost-forecasting/src/global/lag_features.rs
crates/cartoboost-forecasting/src/global/carto_boost_lag.rs
crates/cartoboost-forecasting/src/global/quantile_carto_boost_lag.rs
crates/cartoboost-forecasting/src/global/mod.rs
crates/cartoboost-forecasting/tests/lag_feature_tests.rs
crates/cartoboost-forecasting/tests/carto_boost_lag_tests.rs
docs/forecasting_lag_features.md
docs/forecasting_quantile.md
```

Implement fixed lags, rolling mean, rolling median, rolling standard deviation,
rolling min/max, expanding mean, calendar features, static covariates,
known-future covariates, and panel-safe generation.

Leakage rule:

```text
For timestamp t, target-derived features may use only target values from timestamps < t.
```

`CartoBoostLagForecaster` must build the supervised matrix in Rust, train the
existing Rust CartoBoost regressor, forecast recursively, support panels, expose
feature names, and produce `ForecastResult`.

`QuantileCartoBoostLagForecaster` must train one Rust CartoBoost model per
quantile, enforce non-crossing quantiles, and derive intervals from quantiles.

## Agent 5: Registry, Ensemble, Conformal, Reconciliation, Artifacts, Config

Owns:

```text
crates/cartoboost-forecasting/src/registry.rs
crates/cartoboost-forecasting/src/ensemble.rs
crates/cartoboost-forecasting/src/conformal.rs
crates/cartoboost-forecasting/src/reconciliation.rs
crates/cartoboost-forecasting/src/artifacts.rs
crates/cartoboost-forecasting/src/config.rs
crates/cartoboost-forecasting/tests/registry_tests.rs
crates/cartoboost-forecasting/tests/ensemble_tests.rs
crates/cartoboost-forecasting/tests/conformal_tests.rs
crates/cartoboost-forecasting/tests/reconciliation_tests.rs
crates/cartoboost-forecasting/tests/artifact_tests.rs
crates/cartoboost-forecasting/tests/config_tests.rs
docs/forecasting_artifacts.md
docs/forecasting_conformal.md
docs/forecasting_hierarchical.md
```

Implement the registry, fixed and backtest-weighted ensembles, residual and
quantile conformal calibration, bottom-up reconciliation, top-down proportion
reconciliation, MinTrace only if fully implemented, artifacts, and strict TOML
config.

Artifacts must save and load:

```text
manifest.json
forecast.csv
metrics.json
model_metadata.json
config.toml
```

Optional:

```text
forecast.parquet
```

## Agent 6: CLI, Python Bindings, Examples, Docs

Owns:

```text
crates/cartoboost-forecasting/src/cli.rs
crates/cartoboost-forecasting/src/bin/cartoboost-forecast.rs
python/cartoboost/forecasting/__init__.py
python/cartoboost/forecasting/bindings.py
python/cartoboost/forecasting/wrappers.py
tests/forecasting/test_python_bindings.py
tests/forecasting/test_cli_forecast.py
examples/forecasting/*.rs
examples/forecasting/*.py
docs/forecasting_cli.md
docs/forecasting_examples.md
```

Implement Rust CLI:

```bash
cartoboost-forecast fit
cartoboost-forecast predict
cartoboost-forecast backtest
cartoboost-forecast compare
```

Python bindings must call Rust only, expose the public names from the Python API,
avoid `statsmodels` and `pmdarima`, and contain no forecasting logic beyond data
conversion and error wrapping.

Add Rust examples first, then Python wrapper examples.

## Acceptance Criteria

The implementation is complete only when:

- `ForecastFrame` is Rust-native
- `ForecastResult` is Rust-native
- backtesting is Rust-native
- metrics are Rust-native
- artifacts are Rust-native
- CLI is Rust-native
- Python is wrapper-only

Required validation:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
cargo test -p cartoboost-forecasting
uv run pytest tests/forecasting
uv run mkdocs build
```

Required benchmark:

```bash
cargo run -p cartoboost-forecasting --bin forecasting-benchmark -- \
  --output artifacts/forecasting_benchmark.json
```

Docs must clearly state:

- forecasting engine is Rust-native
- Python wrappers call Rust
- no Python forecasting dependencies are required
- model selection guide
- CLI usage
- Python wrapper usage
- artifact format
- backtesting methodology
- leakage prevention

The model quality page must also be satisfied before a public model is accepted.
