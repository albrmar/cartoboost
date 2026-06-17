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
pip install cartoboost
```

For reproducible benchmark development from a source checkout, use a release
native extension:

```sh
uv sync --group dev --group bench
uv run --group dev maturin develop --release
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

The repeated 25k report was refreshed on June 4, 2026 with target-mean zone
treatment through `just nyc-quality-benchmark-repeated`. In that report,
CartoBoost beats XGBoost on RMSE and R2 for every task/split and has higher
median prediction throughput for every task/split, while training remains
slower than XGBoost `hist`.

The comparison checks:

- CartoBoost train time no slower than XGBoost.
- CartoBoost prediction throughput no slower than XGBoost.
- CartoBoost RMSE lower than XGBoost.
- CartoBoost R2 no worse than XGBoost.

Observed medians in the refreshed report:

| task/split | train ratio vs XGBoost | prediction throughput ratio vs XGBoost | RMSE delta vs XGBoost | R2 delta vs XGBoost |
| --- | ---: | ---: | ---: | ---: |
| duration/random | 1.76x | 1.161x | -0.004507 | 0.005453 |
| duration/spatial_holdout | 1.73x | 1.524x | -0.001197 | 0.001594 |
| fare/random | 1.70x | 1.920x | -0.000148 | 0.000163 |
| fare/spatial_holdout | 1.73x | 1.626x | -0.003096 | 0.003472 |
| pickup_demand/random | 1.91x | 1.515x | -0.012370 | 0.025518 |
| pickup_demand/spatial_holdout | 1.49x | 1.015x | -0.002626 | 0.008206 |

The results are setup-specific evidence for this preset. They are not a general
claim about production accuracy or package superiority.

For temporal-spatial modeling, interpret this report as evidence for the exact
NYC taxi preset: task definition, feature handling, row sample, split strategy,
and estimator settings. Re-run the benchmark when changing those assumptions.
