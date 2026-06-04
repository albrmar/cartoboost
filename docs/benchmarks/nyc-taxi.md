# NYC Taxi Benchmarks

`scripts/run_nyc_taxi_quality_benchmarks.py` runs optional model-quality and
speed comparisons on NYC TLC yellow taxi trip-record Parquet files. It compares
GeoBoost with LightGBM and XGBoost when optional benchmark packages are
installed, and always supports a mean-prediction baseline.

The repeated wrapper, `scripts/run_repeated_nyc_taxi_benchmarks.py`, runs the
single-run benchmark several times and writes aggregate speed and quality gates.

## Dependency Boundary

GeoBoost runtime dependencies do not include the cross-package benchmark stack.
Install the `bench` group only for these comparisons.

| Package | Used for |
| --- | --- |
| `xgboost>=2.0` | XGBoost `XGBRegressor` baseline with configurable `tree_method`. |
| `lightgbm>=4.0` | LightGBM `LGBMRegressor` baseline. |
| `pandas>=2.0` | TLC Parquet loading and cleaning. |
| `pyarrow>=14.0` | Parquet engine for pandas. |
| `matplotlib>=3.7` | Plots. |

## Setup

Benchmark timings should use a release native extension:

```sh
uv sync --group dev --group bench
uv run --group dev maturin develop --release
```

The TLC data cache lives under `data/nyc_taxi/` and must not be committed.

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
and committed summaries under `docs/assets/nyc_taxi_benchmarks/`.

The maintained repeated preset uses:

- GeoBoost candidate: `n_estimators=100`, `max_depth=5`,
  `splitters=axis_histogram:512`, `min_samples_leaf=1`.
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

Profile Rust fit timing inside repeated runs:

```sh
PYTHONPATH=python uv run --group dev --group bench python scripts/run_repeated_nyc_taxi_benchmarks.py \
  --runs 1 \
  --no-download \
  --profile-fit
```

Profile lines are emitted on stderr with `geoboost_fit_profile` and include
context, histogram accumulation/scoring, materialization, leaf, residual, and
prediction-update timings.

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
instead of making it GeoBoost-specific.

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

The committed repeated 25k report uses target-mean zone treatment. In the
current artifact, GeoBoost beats XGBoost on RMSE and R2 for every task/split but
misses the all-task speed gate against XGBoost `hist`. The repeated gate
requires:

- GeoBoost train time no slower than XGBoost.
- GeoBoost prediction throughput no slower than XGBoost.
- GeoBoost RMSE lower than XGBoost.
- GeoBoost R2 no worse than XGBoost.

The committed results are setup-specific evidence for this maintained preset.
They are not a general claim about production accuracy or package superiority.
