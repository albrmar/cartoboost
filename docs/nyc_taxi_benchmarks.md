# NYC Taxi Quality Benchmarks

`scripts/run_nyc_taxi_quality_benchmarks.py` runs model-quality and speed
benchmarks on NYC TLC yellow taxi trip-record Parquet files. It compares
GeoBoost with LightGBM and XGBoost when the optional benchmark packages are
installed, and always supports a mean-prediction baseline.

The benchmark writes results and plots to
`docs/assets/nyc_taxi_benchmarks/` by default.

## Comparison Dependencies

GeoBoost itself depends on NumPy and scikit-learn for the Python estimator API.
Cross-package benchmark comparisons are intentionally optional and live in the
`bench` dependency group:

| package | used for | required for normal GeoBoost use |
| --- | --- | --- |
| `xgboost>=2.0` | XGBoost `XGBRegressor` baseline with `tree_method=hist` | no |
| `lightgbm>=4.0` | LightGBM `LGBMRegressor` baseline | no |
| `pandas>=2.0` | NYC TLC Parquet loading and cleaning | no |
| `pyarrow>=14.0` | Parquet engine for pandas | no |
| `matplotlib>=3.7` | benchmark plots, from the dev dependency group | no |

The repeated speed report uses comparable tree settings by default:

- GeoBoost candidate: `n_estimators=100`, `max_depth=4`,
  `splitters=axis_histogram:64`.
- XGBoost baseline: `n_estimators=100`, `max_depth=4`, `tree_method=hist`,
  `subsample=1.0`, `colsample_bytree=1.0`.
- LightGBM baseline: `n_estimators=100`, `max_depth=4`, `num_leaves=16`.
- Zone IDs use `--zone-treatment target_mean` by default. The transform appends
  train-only smoothed pickup/dropoff zone target-mean features to every model,
  including XGBoost and LightGBM, so the comparison is feature-comparable rather
  than GeoBoost-specific.

## Problems

- Trip duration regression: predict log trip duration from trip, zone,
  passenger, and time features.
- Fare regression: predict log total fare amount from trip, zone, passenger, and
  time features.
- Pickup-zone demand regression: predict log pickup count for each pickup zone,
  hour, and weekday bucket.

Each task reports both a random split and a pickup-zone spatial holdout split.
The spatial split is intended to show geographic generalization quality rather
than only random-row interpolation.

## Running

Install the native GeoBoost extension in release mode first. Benchmark timings
should not use a debug Rust extension when comparing against optimized
LightGBM/XGBoost wheels:

```sh
uv run --group dev maturin develop --release
```

Sync the optional benchmark dependency group:

```sh
uv sync --group dev --group bench
```

Then run:

```sh
just nyc-quality-benchmark
```

For repeated speed-ratio validation on the 25k-row sample:

```sh
just nyc-quality-benchmark-repeated
```

That writes per-run artifacts under `target/nyc_taxi_repeated/` and commits only
aggregate summaries under `docs/assets/nyc_taxi_benchmarks/`.

For a smaller run:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --year 2024 \
  --months 1 \
  --sample-size 25000
```

To compare raw numeric zone IDs against the comparable target-mean zone
treatment:

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

For dependency-light smoke validation without TLC files:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --synthetic-smoke \
  --models mean
```

## Outputs

- `results.json`: machine-readable task, split, metric, parameter, skip, and
  timing metadata.
- `results.md`: human-readable quality and speed benchmark tables.
- `metric_summary.png`: RMSE summary across tasks, splits, and models.
- `speed_summary.png`: stacked train/predict time summary.
- `prediction_throughput.png`: prediction rows-per-second summary.
- `plots/*_predicted_actual.png`: predicted-vs-actual plots.
- `plots/*_zone_residuals.png`: pickup-zone residual summaries.

## Current 25k Comparison Snapshot

The committed `docs/assets/nyc_taxi_benchmarks/repeated_results.md` artifact was
regenerated with the comparable target-mean zone treatment. In that run,
GeoBoost is still slower than XGBoost on every NYC task/split and does not meet
the speed gate. Quality gaps are tracked explicitly instead of hidden:

- Duration: GeoBoost RMSE is `+0.018119` and `+0.015825` higher than XGBoost
  on random and spatial-holdout splits.
- Fare: GeoBoost RMSE is `+0.007884` and `+0.003004` higher than XGBoost.
- Pickup demand: GeoBoost RMSE is `+0.005680` and `+0.005781` higher than
  XGBoost after target-mean zone treatment.

The target-mean zone treatment materially improves the pickup-demand GeoBoost
quality compared with raw numeric zone IDs in paired local diagnostics:

| split | raw GeoBoost RMSE | target-mean GeoBoost RMSE | R2 delta |
| --- | ---: | ---: | ---: |
| pickup_demand/random | `0.476360` | `0.398324` | `+0.182178` |
| pickup_demand/spatial_holdout | `0.623172` | `0.603939` | `+0.061784` |

The same transform is not universally positive: it improved random duration and
fare quality, but regressed duration spatial holdout and slightly regressed fare
spatial holdout for GeoBoost in the one-run all-task diagnostic. Treat benchmark
tables as setup-specific evidence, not broad package superiority claims.

The TLC data cache lives under `data/nyc_taxi/` and should not be committed.
The TLC trip record page states that trip data is submitted by providers and
that TLC makes no representation about its accuracy, so benchmark reports should
be read as reproducible evidence for this setup rather than broad production
claims.
