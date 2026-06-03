GeoBoost benchmark scaffold
===========================

These Criterion benches cover the current public `geoboost-core` surfaces with
small deterministic fixtures:

* `training.rs` benchmarks `Booster::fit` on bounded synthetic matrices.
* `prediction.rs` benchmarks `Model::predict` on lightweight synthetic batches.
* `serialize.rs` benchmarks in-memory JSON serialization and deserialization.

They are wired through `crates/geoboost-core/Cargo.toml` with `harness = false`.
The fixture sizes intentionally stay small so the benches are suitable as
low-memory scaffolding while the implementation is still evolving.
