# Forecasting Model Quality And Implementation Tests

This page defines the quality bar for the pure-Rust forecasting engine. It is
not enough for a model to compile or return rows. A public forecasting model must
show that it is correctly implemented, leakage-safe, deterministic, and useful
on fixtures that exercise the behavior it claims to support.

Every public model must satisfy four gates:

1. Rust implementation tests prove the algorithmic contract.
2. Rust integration tests prove it works through `ForecastFrame`,
   `ForecastResult`, backtesting, registry, artifacts, and CLI.
3. Python binding tests prove Python calls Rust and does not reimplement model
   logic.
4. Benchmark fixtures record comparable metrics and runtime metadata.

If a model cannot pass these gates, it must not be public and must not appear in
`ForecastRegistry`.

## Required Test Layout

```text
crates/cartoboost-forecasting/tests/
  schema_tests.rs
  frequency_tests.rs
  splitter_tests.rs
  metric_tests.rs
  backtesting_tests.rs
  local_model_tests.rs
  state_space_tests.rs
  intermittent_tests.rs
  decomposition_tests.rs
  lag_feature_tests.rs
  cartoboost_lag_tests.rs
  quantile_tests.rs
  ensemble_tests.rs
  conformal_tests.rs
  reconciliation_tests.rs
  artifact_tests.rs
  config_tests.rs
  cli_tests.rs
  python_binding_contract_tests.rs
  full_stack_tests.rs
```

Synthetic benchmark fixtures should live under:

```text
crates/cartoboost-forecasting/tests/fixtures/
  trend_only.csv
  weekly_seasonal.csv
  intermittent_sparse.csv
  panel_lanes.csv
  known_future_covariates.csv
  noisy_geotemporal.csv
  hierarchy_bottom_up.csv
```

Golden outputs should be committed only when they represent deterministic
contracts. Large benchmark outputs belong under `artifacts/` only when they are
intentionally used as evidence in the PR.

## Fixture Design

Use deterministic fixtures with known structure:

| Fixture | Purpose | Required winner or invariant |
| --- | --- | --- |
| `trend_only` | linear trend without seasonality | trended models beat naive on MAE |
| `weekly_seasonal` | daily series with seven-day seasonality | seasonal naive/theta/ETS beat naive |
| `intermittent_sparse` | long zero runs and sporadic demand | Croston/SBA/TSB beat naive on WAPE or zero-demand F1 |
| `panel_lanes` | related lane-level series | panel models keep series isolated and preserve row alignment |
| `known_future_covariates` | future-known exogenous signal | SARIMAX or CartoBoost lag uses future exog without target leakage |
| `noisy_local_level` | level plus observation noise | local-level Kalman improves RMSE over naive |
| `noisy_local_linear_trend` | drifting state with noise | local-linear-trend Kalman improves RMSE over local-level |
| `hierarchy_bottom_up` | bottom series plus aggregate | reconciliation produces coherent totals |
| `quantile_calibration` | skewed residual distribution | quantiles are non-crossing and coverage is monotonic |

Fixtures must use taxi-domain vocabulary by default: `PULocationID`,
`DOLocationID`, pickup/dropoff lane ids, trip counts, fare, duration, trip
distance, hour, and day-of-week features.

## Algorithm-Specific Tests

### Naive And Seasonal Naive

Tests must verify:

- last-observation forecast for every horizon;
- seasonal cycle repetition with correct phase;
- insufficient-history errors;
- panel forecasts do not bleed values between series;
- deterministic timestamps and horizons;
- prediction intervals have the requested shape.

### Theta And Optimized Theta

Tests must verify:

- non-seasonal trend fixture improves over naive;
- additive seasonality removes and restores the seasonal pattern;
- multiplicative seasonality rejects non-positive targets;
- optimized theta selects parameters deterministically;
- metadata records selected theta, alpha, season length, and optimization score;
- panel fitting is one local model per series.

### ETS

Tests must verify each public ETS form:

```text
ANN
AAN
AAdN
ANA
AAA
```

Each form needs:

- fitted value length equals training length;
- residuals are finite;
- forecast mean and variance are finite;
- parameter optimization is deterministic for the same fixture;
- multiplicative modes are absent unless fully implemented and tested.

### ARIMA And SARIMAX

Tests must verify:

- differencing restores forecast timestamps correctly;
- unsupported order families are rejected at construction;
- MA/AR parameter estimation is deterministic;
- residuals are finite;
- known-future exogenous regressors are required at predict time;
- historical-only exogenous columns are rejected for future frames;
- seasonal differencing is exposed only when fully implemented.

### Kalman Models

Tests must verify:

- local-level filtered state follows the latent level fixture;
- local-linear-trend state tracks both level and slope;
- state covariance is positive semidefinite within tolerance;
- forecast variance increases with horizon;
- missing observations update the time step without using missing targets;
- intervals widen when process or observation noise increases.

### Intermittent Demand

Croston, SBA, and TSB tests must verify:

- negative demand is rejected;
- all-zero series returns explicit zero forecasts and metadata;
- sparse positive demand updates size and interval components correctly;
- long zero runs do not produce NaN or infinite forecasts;
- zero-demand precision, recall, and F1 are reported.

### Decomposition Models

STL-based tests must verify:

- seasonal component has the expected period;
- trend plus seasonal plus residual reconstructs the original series within
  tolerance;
- robust decomposition downweights an injected outlier;
- STL plus ETS and STL plus ARIMA reseasonalize forecasts correctly.

MSTL tests must not exist as public acceptance tests until MSTL is fully
implemented. Do not expose MSTL registry names before that point.

### CartoBoost Lag Forecasting

Tests must verify:

- fixed lag correctness for single-series and panel data;
- rolling and expanding features exclude the current target;
- panel series cannot share lag state;
- static covariates are repeated correctly;
- known-future covariates are available at prediction time;
- historical-only covariates are rejected for future frames;
- recursive forecasts append predicted targets only to the matching series;
- feature names are deterministic;
- training uses the Rust CartoBoost regressor, not Python.

### Quantile CartoBoost Lag

Tests must verify:

- one Rust model is trained per requested quantile;
- quantile outputs are non-crossing after correction;
- intervals derived from quantiles use the expected lower and upper levels;
- pinball loss is reported per quantile;
- coverage improves after conformal calibration on calibration fixtures.

## Leakage Tests

Leakage tests are required for every feature generator and backtester:

| Test | Required failure |
| --- | --- |
| validation timestamp included in train | splitter or backtester errors |
| current target used as lag | lag builder errors or test detects mismatched feature value |
| future target used in rolling window | lag builder errors or test detects mismatched feature value |
| panel A target appears in panel B feature | panel isolation assertion fails |
| future historical-only covariate provided | predictor rejects future frame |
| known-future covariate missing | predictor rejects future frame |
| forecast aligned by row order only | backtester alignment test fails |
| random CV requested | API does not expose random CV and config rejects it |

Every leakage test must be deterministic and must assert the specific error or
feature value. Tests that only check object construction are not sufficient.

## Metric Quality Tests

Metric tests must include hand-computed fixtures for:

- MAE
- RMSE
- MAPE with zeros
- sMAPE with zeros
- MASE
- RMSSE
- WAPE
- bias
- pinball loss at multiple quantiles
- interval coverage
- mean interval width
- MSIS
- interval score
- hierarchical coherence error
- zero-demand precision, recall, and F1

Frame-level metric tests must intentionally shuffle forecast rows and actual
rows. Correct code must still align by `series_id`, `timestamp`, and `horizon`.

## Artifact And Registry Tests

Tests must verify:

- every registered model constructs a real Rust implementation;
- unimplemented models are absent from the registry;
- model metadata round-trips through artifacts;
- forecast CSV round-trips without hidden process state;
- optional Parquet hard-fails clearly when the feature is not enabled;
- `config.toml` rejects unknown fields unless explicitly allowed;
- artifact manifests include model name, version, horizon, frequency, schema,
  feature config, interval metadata, calibration metadata, and backtest metrics.

## Python Binding Tests

Python binding tests must prove wrapper-only behavior:

- wrappers import without `statsmodels`, `pmdarima`, `sktime`, `prophet`,
  `neuralforecast`, or `darts`;
- `ForecastFrame.from_pandas` converts rows into Rust-owned data;
- `fit` and `predict` call Rust objects;
- Python model classes contain no forecasting algorithm code;
- Python forecast outputs match Rust CLI outputs on the same CSV fixture;
- Python errors preserve Rust error messages.

The Python package may use pandas only for input conversion and display. Pandas
must not be required by the Rust forecasting crate or Rust CLI.

## Benchmark Acceptance

The Rust benchmark must run:

```bash
cargo run -p cartoboost-forecasting --bin forecasting-benchmark -- \
  --output artifacts/forecasting_benchmark.json
```

The benchmark JSON must include:

```json
{
  "created_at": "...",
  "cartoboost_version": "...",
  "engine": "rust",
  "datasets": ["..."],
  "models": ["..."],
  "metrics": {
    "dataset": {
      "model": {
        "mae": 0.0,
        "rmse": 0.0,
        "mase": 0.0,
        "rmsse": 0.0,
        "wape": 0.0,
        "smape": 0.0,
        "bias": 0.0,
        "coverage_80": 0.0,
        "coverage_95": 0.0,
        "pinball_50": 0.0,
        "train_ms": 0.0,
        "predict_ms": 0.0
      }
    }
  }
}
```

Benchmarks must not require a model to win every fixture. They must show fair
settings, comparable splits, deterministic metrics, and leakage-safe
evaluation.

## CI Gates

Pure-Rust forecasting is not complete until these pass:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
cargo test -p cartoboost-forecasting
cargo run -p cartoboost-forecasting --bin forecasting-benchmark -- \
  --output artifacts/forecasting_benchmark.json
uv run maturin develop
uv run pytest tests/forecasting
uv run --group docs mkdocs build
```

A PR must report which tests are Rust unit tests, Rust integration tests, CLI
tests, Python binding tests, benchmark runs, and docs builds. If any public
model lacks one of those surfaces, the model must be removed from the public API
and registry until the gap is closed.
