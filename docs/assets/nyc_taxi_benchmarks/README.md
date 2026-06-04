# NYC Taxi Benchmark Assets

This directory is the default output location for
`scripts/run_nyc_taxi_quality_benchmarks.py`.

Generated files include `results.json`, `results.md`, `metric_summary.png`,
`speed_summary.png`, `prediction_throughput.png`, and per-task plots under
`plots/`. Raw TLC Parquet files are cached under `data/nyc_taxi/` and are
intentionally not committed.

These artifacts are optional cross-package benchmark outputs. XGBoost,
LightGBM, pandas, and pyarrow come from the `bench` dependency group and are not
runtime dependencies of GeoBoost. The current committed comparison uses
`--zone-treatment target_mean`, which appends train-only smoothed zone
target-mean features to every boosting package before fitting.
