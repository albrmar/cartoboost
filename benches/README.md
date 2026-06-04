# GeoBoost Benchmark Scaffold

This directory contains Criterion microbenchmarks for `geoboost-core`.
Maintained benchmark runbooks and artifact policy live under
`docs/benchmarks/`.

Files:

- `training.rs`: `Booster::fit` on bounded synthetic matrices.
- `prediction.rs`: `Model::predict` and flat prediction paths on synthetic
  batches.
- `serialize.rs`: in-memory JSON serialization and deserialization.

Useful commands:

```sh
cargo bench --workspace --no-run
cargo bench --workspace
uv run --group dev python scripts/plot_benchmarks.py
```

Do not commit raw `target/criterion/` output. Refresh generated summary images
only when benchmark artifacts are explicitly part of the task.
