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

`benches/prediction.rs` includes cached flat-axis prediction cases with
25,000-row batches and 100 trees. These cases are the primary speed smoke test
for the optimized dense prediction path. v2 prediction transforms, including
`log_l2` inverse transforms, are applied after the raw tree sum and should not
disable the cached flat-axis path for compatible models.

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

Run only the prediction speed smoke with a small sample count:

```sh
cargo bench --bench prediction -- --sample-size 10
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
