# Benchmarks

CartoBoost benchmark docs are for reproducible model comparison. Use them to
answer data-science questions such as whether temporal-spatial splitters improve
random, spatial, or temporal holdouts against an axis-only CartoBoost model,
XGBoost, LightGBM, or a mean baseline.

| Page | Question |
| --- | --- |
| [Model Benchmark Suite](model-suite.md) | How do dense, neural, standalone graph, and graph-augmented CartoBoost variants compare with XGBoost/LightGBM on deterministic workloads? |
| [NYC Taxi Benchmarks](nyc-taxi.md) | Does CartoBoost beat LightGBM on real temporal-spatial taxi tasks, and which structural primitive explains the win? |
| [Neural Embedding Strategy Assessment](neural-embedding-strategy.md) | When do neural embeddings help, and how did OOF, fallback, multi-key, and shrinkage changes compare? |
| [Taxi Zone Acceptance](taxi-zone.md) | Does the model capture pickup/dropoff, temporal, spatial, and combined behavior on a controlled dataset? |
| [Neural Embedding Benchmark (latest)](neural-embedding-benchmark-latest.md) | How much does neural feature augmentation improve MAE under synthetic temporal-spatial holdouts? |

## Forecasting Benchmark

Forecasting V1 includes a deterministic synthetic benchmark harness:

```sh
python scripts/forecasting_benchmark.py --output artifacts/forecasting_benchmark.json
```

The harness compares naive, seasonal naive, theta, optimized theta,
CartoBoost lag forecasting, and weighted ensembles across trend-only, weekly
seasonal, intermittent sparse, panel lane, known-future covariate, and noisy
geotemporal fixtures. The output JSON records dataset names, model names, MAE,
RMSE, MASE, WAPE, sMAPE, bias, and interval coverage placeholders. These runs
are intended to prove repeatability, row alignment, and leakage-safe evaluation;
they are not presented as universal model superiority evidence.

## Forecasting Library Comparison

`scripts/forecasting_library_benchmark.py` compares CartoBoost global lag
forecasting with `functime` on a deterministic geographic-temporal
pickup/dropoff lane demand fixture. `functime` is the explicit known forecasting
library baseline; the benchmark runs Polars-native seasonal naive, ridge, and
LightGBM autoregressive forecasters and compares CartoBoost against the best
RMSE among those methods.

The fixture can be sourced through Polars or DuckDB:

```sh
uv run --group dev --group bench python scripts/forecasting_library_benchmark.py \
  --backend polars \
  --output artifacts/forecasting_library_benchmark_polars.json

uv run --group dev --group bench python scripts/forecasting_library_benchmark.py \
  --backend duckdb \
  --output artifacts/forecasting_library_benchmark_duckdb.json
```

Current quality run, June 18, 2026:

| backend | known library | best known method | CartoBoost RMSE | best known RMSE | RMSE ratio |
| --- | --- | --- | ---: | ---: | ---: |
| Polars | functime | seasonal naive | 0.379765 | 0.841069 | 0.451526 |
| DuckDB | functime | seasonal naive | 0.379765 | 0.841069 | 0.451526 |

Current speed context from the same artifact refresh:

| backend | CartoBoost model seconds | best known model seconds | slowest known model seconds | total seconds |
| --- | ---: | ---: | ---: | ---: |
| Polars | 0.605414 | 0.008985 | 0.256526 | 1.840797 |
| DuckDB | 0.493050 | 0.007441 | 0.216486 | 1.467652 |

The best known `functime` method by RMSE is seasonal naive, which is much faster
than CartoBoost but less accurate on this fixture. In this run,
`functime_lightgbm` is the slowest known-method row and is also less accurate
than seasonal naive on the deterministic lane-demand fixture.

This is targeted evidence for global geotemporal lag forecasting on many
related short lane series. It should not be generalized to every forecasting
task or to long single-series statistical forecasting workloads.

## Evaluation Helpers

Objective, calibration, spatial-diagnostic, and blocked-validation helpers are
available from the Python package:

```python
from cartoboost import (
    out_of_time_split,
    residual_morans_i,
    spatial_blocked_cv,
    temporal_blocked_cv,
)
```

Use `out_of_time_split` for the latest-period holdout, then compare that score
with random and spatial holdouts. Temporal-spatial models should improve where
the deployment split is hardest, not only on random validation rows.

Prediction speed should be reported only with the benchmark command, data size,
model settings, and comparison baseline.

## Quick Commands

Run dependency-light NYC smoke validation:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --synthetic-smoke \
  --models mean
```

Run the synthetic model suite with optional XGBoost and LightGBM baselines:

```sh
uv run --group dev --group bench python scripts/run_model_benchmark_suite.py
```

Run the maintained repeated NYC comparison target:

```sh
just nyc-quality-benchmark-repeated
```

Run the full single-run NYC benchmark artifact refresh:

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_nyc_taxi_quality_benchmarks.py \
  --no-download \
  --output-dir docs/assets/nyc_taxi_benchmarks
```

## Reporting Rules

- State the exact command, sample size, task set, model settings, and feature
  handling.
- Distinguish single-run diagnostics from repeated summaries.
- Do not present synthetic checks or NYC taxi outputs as universal model
  superiority claims.
- Treat synthetic checks as behavior evidence, not broad quality claims.
