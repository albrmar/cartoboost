# NYC Taxi Benchmarks

`scripts/run_nyc_taxi_quality_benchmarks.py` runs model-quality and speed
comparisons on NYC TLC yellow taxi trip-record Parquet files. It compares
CartoBoost with LightGBM and XGBoost when optional benchmark packages are
installed, and always supports a mean-prediction baseline.

The repeated wrapper, `scripts/run_repeated_nyc_taxi_benchmarks.py`, runs the
single-run benchmark several times and summarizes speed and quality.

## Dependency Boundary

CartoBoost runtime dependencies do not include the cross-package benchmark stack.
Install the `bench` group only for these comparisons.

| Package | Used for |
| --- | --- |
| `xgboost>=2.0` | XGBoost `XGBRegressor` baseline with configurable `tree_method`. |
| `lightgbm>=4.0` | LightGBM `LGBMRegressor` baseline. |
| `pandas>=2.0` | TLC Parquet loading and cleaning. |
| `pyarrow>=14.0` | Parquet engine for pandas. |
| `matplotlib>=3.7` | Plots. |

## Setup

Benchmark scripts are repository workflows. Install the package from PyPI for
normal model use:

```sh
uv add cartoboost
```

For reproducible benchmark development from a source checkout:

```sh
uv sync --group dev --group bench
```

The TLC data cache lives under `data/nyc_taxi/` and must not be committed.
The benchmark also caches `taxi_zone_lookup.csv` there so demand and row tasks
can use borough and service-zone context for spatial holdouts.

## Maintained Targets

Single run:

```sh
just nyc-quality-benchmark
```

Repeated 25k-row comparison:

```sh
just nyc-quality-benchmark-repeated
```

The repeated target writes per-run outputs under `target/nyc_taxi_repeated/`
and summary reports under `docs/assets/nyc_taxi_benchmarks/`.

The maintained repeated preset uses:

- CartoBoost candidate: `n_estimators=100`, `max_depth=5`,
  `splitters=axis_histogram:512,periodic:24,periodic:7,sparse_set`,
  `min_samples_leaf=1`.
- XGBoost baseline: `n_estimators=100`, `max_depth=4`,
  `tree_method=hist`, `subsample=1.0`, `colsample_bytree=1.0`.
- Zone IDs: `--zone-treatment target_mean`.

## Smaller Diagnostics

Run one 25k-row month:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --year 2024 \
  --months 1 \
  --sample-size 25000
```

Run without TLC files:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --synthetic-smoke \
  --models mean
```

## Tasks

| Task | Target |
| --- | --- |
| `duration` | Log trip duration from trip, zone, passenger, and time features. |
| `fare` | Log total fare amount from trip, zone, passenger, and time features. |
| `pickup_demand` | Log pickup count by pickup zone, hour, and weekday bucket. |

Each task reports a random split and a pickup-zone spatial holdout split.

## Comparable Feature Handling

The maintained comparison uses `--zone-treatment target_mean` by default. It
appends train-only smoothed pickup/dropoff zone target-mean features to every
model, including XGBoost and LightGBM. This keeps zone handling comparable
instead of making it CartoBoost-specific.

Neural and graph-augmented CartoBoost rows should be interpreted against the
same split boundary. Neural ID embeddings can help when pickup or dropoff zones
repeat between train and validation, but a cold-zone spatial holdout requires
fallback vectors for unseen zones. The maintained neural row uses out-of-fold
residual embeddings, support-aware shrinkage, multi-key zone embeddings, same
service-zone and same-borough fallback representatives, and adjacent-zone
neighbor fallback. Repeated-zone gains should not be reported as evidence of
cold-zone generalization.

To compare raw numeric zone IDs against target-mean treatment:

```sh
uv run --group dev --group bench python scripts/run_repeated_nyc_taxi_benchmarks.py \
  --runs 3 \
  --no-download \
  --tasks pickup_demand \
  --zone-treatment raw

uv run --group dev --group bench python scripts/run_repeated_nyc_taxi_benchmarks.py \
  --runs 3 \
  --no-download \
  --tasks pickup_demand \
  --zone-treatment target_mean
```

## Outputs

Single-run outputs under `docs/assets/nyc_taxi_benchmarks/`:

- `results.json`
- `results.md`
- `metric_summary.png`
- `speed_summary.png`
- `prediction_throughput.png`
- `plots/*_predicted_actual.png`
- `plots/*_zone_residuals.png`

Repeated-run summaries:

- `repeated_results.json`
- `repeated_results.md`

## Current Snapshot

The full single-run report was refreshed on June 18, 2026 with real January
2024 NYC TLC yellow taxi data, target-mean zone treatment, all maintained
CartoBoost, graph, neural, LightGBM, XGBoost, and mean rows, and the command:

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_nyc_taxi_quality_benchmarks.py \
  --no-download \
  --output-dir docs/assets/nyc_taxi_benchmarks
```

This command used the cached `data/nyc_taxi/yellow_tripdata_2024-01.parquet`
file and `--no-download`, so missing real data would fail instead of falling
back to a synthetic fixture.

On every runnable learned-model split, the best CartoBoost-family row beats
LightGBM on RMSE and R2:

| task/split | best CartoBoost-family row | CartoBoost RMSE | LightGBM RMSE | RMSE delta | R2 delta |
| --- | --- | ---: | ---: | ---: | ---: |
| duration/random | `cartoboost` | 0.278843 | 0.289829 | -0.010986 | 0.012637 |
| duration/spatial_holdout | `cartoboost` | 0.303525 | 0.316291 | -0.012766 | 0.018206 |
| fare/random | `cartoboost` | 0.139640 | 0.143692 | -0.004052 | 0.004239 |
| fare/spatial_holdout | `cartoboost` | 0.148375 | 0.152686 | -0.004311 | 0.007854 |
| pickup_demand/random | `cartoboost_graph_node2vec` | 0.403529 | 0.481552 | -0.078024 | 0.016572 |

The pickup-demand spatial holdout intentionally skips all learned models. That
split removes all zone demand history, so learned-model predictions collapse to
priors; reporting LightGBM or CartoBoost there would be a fallback comparison,
not a valid quality comparison.

## Why CartoBoost Is Better Here

Fare and duration are geotemporal row tasks. The winning row is base
CartoBoost, not a graph or neural variant, because the target is already well
explained by primitives LightGBM does not natively have: periodic hour/day
splitters, diagonal and radial spatial splitters, and sparse-set taxi-zone
membership. LightGBM receives comparable target-mean zone features, but it still
has to approximate pickup/dropoff geometry with axis-aligned tabular splits.

Pickup demand is different. It is a zone-time graph problem, and the best row is
`cartoboost_graph_node2vec`. The graph encoder learns pickup-zone topology from
observed zone relationships, then CartoBoost models that topology together with
hour, weekday, and zone effects. That is the benchmark case where graph
augmentation adds signal beyond the base geotemporal splitter set.

Neural and graph rows are therefore not treated as universal upgrades. If the
base geotemporal splitter set already captures the target, they can match the
base row while adding training cost. Their value is specific: neural rows expose
train-observed ID residual structure, and graph rows expose source-target or
zone-topology structure that ordinary dense columns do not encode.

The results are setup-specific evidence for this preset. They are not a general
claim about production accuracy or package superiority. Re-run the benchmark
when changing task definitions, feature handling, row sample, split strategy,
or estimator settings.
