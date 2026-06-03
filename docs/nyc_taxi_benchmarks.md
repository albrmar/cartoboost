# NYC Taxi Quality Benchmarks

`scripts/run_nyc_taxi_quality_benchmarks.py` runs model-quality benchmarks on
NYC TLC yellow taxi trip-record Parquet files. It compares GeoBoost with
LightGBM and XGBoost when those packages are installed, and always supports a
mean-prediction baseline.

The benchmark writes results and plots to
`docs/assets/nyc_taxi_benchmarks/` by default.

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

Install the native GeoBoost extension first:

```sh
uv run --group dev maturin develop
```

Sync the optional benchmark dependency group:

```sh
uv sync --group dev --group bench
```

Then run:

```sh
just nyc-quality-benchmark
```

For a smaller run:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --year 2024 \
  --months 1 \
  --sample-size 25000
```

For dependency-light smoke validation without TLC files:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --synthetic-smoke \
  --models mean
```

## Outputs

- `results.json`: machine-readable task, split, metric, parameter, skip, and
  dataset metadata.
- `results.md`: human-readable benchmark tables.
- `metric_summary.png`: RMSE summary across tasks, splits, and models.
- `plots/*_predicted_actual.png`: predicted-vs-actual plots.
- `plots/*_zone_residuals.png`: pickup-zone residual summaries.

The TLC data cache lives under `data/nyc_taxi/` and should not be committed.
The TLC trip record page states that trip data is submitted by providers and
that TLC makes no representation about its accuracy, so benchmark reports should
be read as reproducible evidence for this setup rather than broad production
claims.
