# Criterion Microbenchmarks

Criterion benchmarks cover Rust core surfaces with deterministic synthetic
fixtures. They are scaffolding for timing regressions, not a product benchmark
suite.

## Files

| File | Measures |
| --- | --- |
| `benches/training.rs` | `Booster::fit` on bounded synthetic matrices. |
| `benches/prediction.rs` | `Model::predict` and flat prediction paths on synthetic batches. |
| `benches/serialize.rs` | In-memory JSON serialization and deserialization. |

The benches are wired through `crates/geoboost-core/Cargo.toml` with
`harness = false`.

## Commands

Compile only:

```sh
cargo bench --workspace --no-run
```

Run the benches:

```sh
cargo bench --workspace
```

Generate a local summary plot when Criterion artifacts exist:

```sh
uv run --group dev python scripts/plot_benchmarks.py
```

The plot script writes `benches/benchmark_summary.png` by default. Treat that as
a generated artifact and refresh it only when benchmark artifacts are part of
the task.

## Interpretation

- Compare results only from the same machine class, build profile, and command.
- Prefer repeated local runs before claiming a regression.
- Do not compare Criterion synthetic fixtures directly with NYC taxi model
  quality benchmarks.
