# CartoBoost Agent Guide

## Dev environment tips
- This repository is a Rust workspace with Python bindings and a Python package for CartoBoost regression tooling.
- Use `uv sync --group dev` to prepare the Python development environment.
- Use `uv run --group dev maturin develop` after native binding changes so `cartoboost._native` is available to Python tests.
- Keep shared model behavior in `crates/cartoboost-core`; keep CLI-only behavior in `crates/cartoboost-cli`, PyO3 bindings in `crates/cartoboost-py`, and Python API ergonomics in `python/cartoboost`.
- Implement ALL algorithms and ALL behavior in Rust under `crates/`; Python classes should be thin configuration and ergonomics wrappers over native bindings.
- Do not edit build output, virtualenvs, caches, downloaded data, or benchmark output under `target/`, `.venv/`, `.pytest_cache/`, `data/`, or generated plot folders unless the task explicitly asks for generated artifacts.
- Keep examples, docs, and public benchmark narratives aligned with the NYC taxi domain: pickup/dropoff zones, `PULocationID`, `DOLocationID`, taxi trips, fare, duration, trip distance, and hour/day features. Avoid non-taxi-domain examples unless explicitly requested.
- Optional integrations must stay optional. Add dependencies under extras such as `polars`, `duckdb`, `h3`, or `s2`; do not make them core dependencies. Helpers that require an optional package should hard-fail with a clear install hint rather than silently degrading.
- Benchmark and validation code should hard-fail when required real inputs are missing or invalid. Avoid silent fallback behavior for geo assets, graph/neural inputs, or benchmark data when that would weaken the claim being tested.
- Quality claims must come from real runs with exact commands, fixed comparable settings, and recorded metrics. Do not p-hack, hyperopt one side, or tune benchmark settings until a preferred result appears. Compare against serious baselines such as LightGBM/XGBoost with the same train/test split and comparable estimator settings.
- Benchmark refreshes must include the writeup. When committing benchmark artifacts or changed benchmark behavior, update the maintained docs/report narrative in the same commit so readers understand the command, data source, metrics, winner, and why the result is structurally meaningful.

## Testing instructions
- Run `just validate` for full local validation.
- For targeted Rust checks, run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace`.
- For targeted Python checks, run `uv run --group dev ruff format --check python tests scripts`, `uv run --group dev ruff check python tests scripts`, and `uv run --group dev pytest`.
- For Python support matrix changes, do not commit claims for versions you have not validated end-to-end through CI/build logs.
  If an interpreter fails native build checks (for example PyO3 compatibility limits), revert the version claim and CI matrix entries until support is real.
- Add or update tests for behavioral changes, especially when changing splitters, serialization, CLI output, Python estimator behavior, or native bindings.
- For benchmark work, capture RMSE, MAE, R2, training time, prediction time, model settings, sample size, task names, and split names. Preserve output artifacts only when they are intentionally committed evidence; otherwise write generated runs under `target/` or `/tmp`.

## PR instructions
- Summarize which surface changed: core Rust, CLI, PyO3 bindings, Python API, docs, tests, fixtures, benchmarks, or scripts.
- Mention targeted validation commands that were run.
- If model outputs, golden files, fixtures, or benchmark artifacts changed, explain why the new outputs are expected.
- If a PR changes benchmark behavior or claims, include the exact command used and identify whether the data was synthetic, generated acceptance data, or real benchmark data.
