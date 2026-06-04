# Benchmarks

GeoBoost has three different evidence paths. Keep them separate when writing
runbooks, reports, and PR summaries.

| Path | Purpose | Default output | Commit policy |
| --- | --- | --- | --- |
| Criterion microbenchmarks | Rust training, prediction, and serialization timing scaffolds. | `target/criterion/` | Do not commit raw Criterion output. |
| Validation artifacts | Deterministic synthetic behavior checks and proof images. | `target/validation/` plus selected committed docs assets | Commit only intentionally refreshed docs assets. |
| NYC taxi benchmarks | Optional real-data model-quality and speed comparisons against benchmark-only packages. | `docs/assets/nyc_taxi_benchmarks/` and `target/nyc_taxi_repeated/` | Commit summary artifacts only when the benchmark was intentionally refreshed. |

## v2 Modeling Utility Checks

The v2 alpha helpers add objective, calibration, spatial-diagnostic, and
blocked-validation utility surfaces. Those utilities are validated by Python
tests, not by Criterion timing numbers:

```sh
uv run --group dev pytest \
  tests/python/test_objectives_v2.py \
  tests/python/test_modeling_metrics.py \
  tests/python/test_evaluation_protocol.py
```

Prediction speed remains covered by the Rust prediction benchmark. The cached
flat-axis benchmark exercises the fast dense path used by native Python
prediction after `maturin develop`.

## Quick Commands

Compile benchmarks without running them:

```sh
cargo bench --workspace --no-run
```

Run the full validation artifact path:

```sh
uv run --group dev maturin develop
uv run --group dev python scripts/run_full_validation.py
```

Run dependency-light NYC smoke validation:

```sh
uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --synthetic-smoke \
  --models mean
```

Run the maintained NYC comparison target:

```sh
just nyc-quality-benchmark-repeated
```

## Reporting Rules

- State the exact command, sample size, task set, model settings, and feature
  handling.
- Distinguish single-run diagnostics from repeated summaries.
- Do not present synthetic fixtures or NYC taxi outputs as universal model
  superiority claims.
- Explain why generated artifacts changed whenever committed markdown, JSON, or
  plots are refreshed.

## Pages

- [Artifact Policy](artifacts.md) defines which generated files belong in the
  repository.
- [Criterion Microbenchmarks](criterion.md) covers Rust benchmark scaffolding.
- [NYC Taxi Benchmarks](nyc-taxi.md) is the operational runbook for real-data
  comparisons.
- [Lane-Level Acceptance](lane-level.md) explains the committed lane diagnostic
  artifacts.
