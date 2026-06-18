# Benchmarks

These reports describe CartoBoost benchmarks as data-science studies. Each one
states the target, dataset, features, split design, comparison methods, metrics,
result, and interpretation. The goal is to explain what was measured and why it
matters.

| Report | Primary question |
| --- | --- |
| [Model Benchmark Suite](model-suite.md) | On controlled synthetic regression tasks, when do dense, ID, and graph features improve predictive quality against XGBoost and LightGBM? |
| [NYC Taxi Benchmarks](nyc-taxi.md) | On real NYC taxi fare, duration, and pickup-demand tasks, which feature families explain the quality gains under random and spatial holdout splits? |
| [Taxi Zone Acceptance](taxi-zone.md) | On a controlled taxi-lane fixture, can the model recover lane membership, route geometry, hour-of-day periodicity, and combined geotemporal structure? |
| [Neural Embedding Strategy Assessment](neural-embedding-strategy.md) | Do residual embeddings improve repeated-ID tasks without overstating cold-ID generalization? |
| [Neural Embedding Benchmark](neural-embedding-benchmark-latest.md) | Across random, temporal, geographic, tail, and cold-ID splits, where does neural augmentation reduce MAE? |

## Forecasting Benchmark

**Research question.** Can global lag features with geographic lane context
forecast short panels of taxi-like demand better than explicit forecasting
libraries on the same deterministic fixture?

**Dataset.** The fixture is synthetic but taxi-shaped. Rows represent daily
pickup/dropoff lane demand with pickup zone, dropoff zone, distance, airport
lane indicator, borough code, weekly seasonality, and deterministic event
spikes. The same data-generating process is loaded through both Polars and
DuckDB sources to verify that the result is not tied to a single dataframe
engine.

**Target.** Future daily lane demand over a fixed forecast horizon.

**Features.** CartoBoost receives lagged demand, rolling demand summaries,
calendar fields, pickup/dropoff zone identifiers, distance, airport-lane flag,
and borough code. Library baselines receive their native time-series inputs and
the comparable covariate structure supported by the benchmark script.

**Methods compared.**

| library | model names |
| --- | --- |
| `cartoboost` | `cartoboost_lag` |
| `functime` | `functime_snaive`, `functime_ridge`, `functime_lightgbm` |
| `statsforecast` | `statsforecast_seasonal_naive`, `statsforecast_autoets` |

**Command.**

```sh
uv run --group dev --group bench python scripts/forecasting_library_benchmark.py \
  --source polars \
  --output artifacts/forecasting_library_benchmark_polars.json

uv run --group dev --group bench python scripts/forecasting_library_benchmark.py \
  --source duckdb \
  --output artifacts/forecasting_library_benchmark_duckdb.json
```

**Metrics.** RMSE is the primary quality metric. MAE and WAPE are secondary
quality metrics. Model seconds are reported as context, not as the primary
claim.

**Current result, June 18, 2026.**

| model | library | RMSE | MAE | WAPE |
| --- | --- | ---: | ---: | ---: |
| `cartoboost_lag` | `cartoboost` | 0.379765 | 0.224178 | 0.010722 |
| `statsforecast_autoets` | `statsforecast` | 0.550793 | 0.397769 | 0.019025 |
| `functime_snaive` | `functime` | 0.841069 | 0.656984 | 0.031423 |
| `statsforecast_seasonal_naive` | `statsforecast` | 0.841069 | 0.656984 | 0.031423 |
| `functime_ridge` | `functime` | 2.673150 | 2.309550 | 0.110462 |
| `functime_lightgbm` | `functime` | 2.965697 | 2.807547 | 0.134281 |

The best external forecasting-library baseline in this run is
`statsforecast_autoets` from StatsForecast. CartoBoost lag forecasting reduces
RMSE by 31.05% relative to that baseline on this fixture. Polars and DuckDB
runs produced identical quality metrics; only loading and timing differed.

**Timing context from the Polars run.**

| model | library | model seconds |
| --- | --- | ---: |
| `cartoboost_lag` | `cartoboost` | 2.474875 |
| `statsforecast_autoets` | `statsforecast` | 0.500582 |
| `functime_snaive` | `functime` | 0.054331 |
| `statsforecast_seasonal_naive` | `statsforecast` | 0.005745 |
| `functime_ridge` | `functime` | 0.179506 |
| `functime_lightgbm` | `functime` | 77.810239 |

**Interpretation.** The target is a panel of related geographic lanes, not a
single long univariate series. The result supports the claim that supervised
lag features plus lane covariates are useful for this kind of short-panel
geotemporal demand problem. It should not be read as a claim that CartoBoost is
better than every forecasting method on long single-series workloads.

## Evaluation Rules

- Report the command, data source, target, feature set, split, and metrics.
- Name every comparison library and model.
- Separate quality findings from timing observations.
- Do not generalize synthetic or single-domain evidence beyond the measured
  target and split design.
- Refresh the written report whenever benchmark artifacts or benchmark behavior
  change.

## Commands

Run dependency-light NYC smoke validation:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --synthetic-smoke \
  --models mean
```

Run the synthetic model suite:

```sh
uv run --group dev --group bench python scripts/run_model_benchmark_suite.py
```

Run the maintained repeated NYC comparison:

```sh
just nyc-quality-benchmark-repeated
```

Refresh the full single-run NYC artifact:

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_nyc_taxi_quality_benchmarks.py \
  --no-download \
  --output-dir docs/assets/nyc_taxi_benchmarks
```
