GeoBoost benchmark scaffold
===========================

These Criterion files describe the benchmark surface GeoBoost should grow into:

* `training.rs` covers histogram-style training work over synthetic feature
  matrices.
* `prediction.rs` covers batch prediction traversal work.
* `data_loading.rs` covers CSV parsing and matrix materialization work.

They currently use local placeholder kernels so the intended workloads are
documented before the public Rust API is finalized. Once `geoboost-core`
exports training, prediction, and data loading entry points, replace the local
kernels with calls into the crate and wire these files through the root
`Cargo.toml` bench configuration.

