# NYC Taxi Benchmark Assets

This directory contains generated evidence for the NYC taxi benchmark. The
maintained narrative is
[NYC Taxi Benchmarks](../../benchmarks/nyc-taxi.md).

The study measures transformed trip duration, transformed fare amount, and
transformed pickup-zone demand. The feature sets include pickup/dropoff zone
context, trip descriptors, hour/day features, and graph features for the
zone-demand task.

Generated files include single-run `results.*`, repeated-run
`repeated_results.*`, top-level summary images, and per-task plots under
`plots/`.

Raw TLC Parquet files are cached under `data/nyc_taxi/` and must not be
committed.
