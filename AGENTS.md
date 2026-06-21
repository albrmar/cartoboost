# CartoBoost Agent Guide

## Dev environment tips
- This repository is a Rust workspace with Python bindings and a Python package for CartoBoost regression tooling.
- Use `uv sync --group dev` to prepare the Python development environment.
- Use `uv run --group dev maturin develop` after native binding changes so `cartoboost._native` is available to Python tests.
- Keep shared model behavior in `crates/cartoboost-core`; keep CLI-only behavior in `crates/cartoboost-cli`, PyO3 bindings in `crates/cartoboost-py`, and Python API ergonomics in `python/cartoboost`.
- Implement ALL algorithms and ALL behavior in Rust under `crates/`; Python classes should be thin configuration and ergonomics wrappers over native bindings.
- All modeling logic must be Rust-core first. Regression, forecasting, neural/graph modeling, metrics, validation, serialization, artifacts, and benchmark-critical scoring behavior belong in `crates/`; Python must not become the source of truth for model behavior.
- Forecasting core specifically must live in Rust. Python forecasting APIs may convert user data, expose ergonomic wrappers, or run benchmark orchestration, but fitting, prediction, backtesting, feature generation, metrics, intervals, artifacts, and leakage checks must be implemented in Rust and called through native bindings.
- Do not edit build output, virtualenvs, caches, downloaded data, or benchmark output under `target/`, `.venv/`, `.pytest_cache/`, `data/`, or generated plot folders unless the task explicitly asks for generated artifacts.
- Keep examples, docs, and public benchmark narratives aligned with the NYC taxi domain: pickup/dropoff zones, `PULocationID`, `DOLocationID`, taxi trips, fare, duration, trip distance, and hour/day features. Avoid non-taxi-domain examples unless explicitly requested.
- Optional integrations must stay optional. Add dependencies under extras such as `polars`, `duckdb`, `h3`, or `s2`; do not make them core dependencies. Helpers that require an optional package should hard-fail with a clear install hint rather than silently degrading.
- Benchmark and validation code should hard-fail when required real inputs are missing or invalid. Avoid silent fallback behavior for geo assets, graph/neural inputs, or benchmark data when that would weaken the claim being tested.
- Quality claims must come from real runs with exact commands, fixed comparable settings, and recorded metrics. Compare against serious baselines such as LightGBM/XGBoost with the same train/test split and comparable estimator settings.
- Benchmark-driven model improvements belong in reusable implementation: shared feature generation, validation, training behavior, and native model code. Keep benchmark datasets, split boundaries, model lists, metrics, and acceptance gates stable across reruns.
- Benchmark refreshes must include a real writeup, not just protocol notes. When committing benchmark artifacts or changed benchmark behavior, update the maintained docs/report narrative in the same commit with the exact command, data source, sample size, task/split definitions, model roster, comparable settings, metric breakdown tables, timing breakdown, artifact paths, relevant plots/images, winner or tie interpretation, limitations, and why the result is structurally meaningful.
- Keep evaluation protocol instructions in this agent guide, not public docs. Public docs should show benchmark evidence, commands, data, plots, metric breakdowns, and interpretations; agent-only protocol details belong here.
- Public docs should read for package users: lead with results, ranked model comparisons, commands, and limits. Do not frame pages or commits around docs process labels such as "new", "simplified", "cleanup", or "provenance".
- When asked to commit and push, work directly on `main` and push directly to `origin/main` unless the user explicitly asks for a branch or PR. Do not create `codex/*` or feature branches for routine requested commits.
- Forecast benchmark docs should keep one ranked model-comparison table per dataset family when possible. If a family includes multiple artifacts, use a `Run` column instead of scattering several small tables.
- Avoid vague table headers such as `Scope` in public benchmark docs. Prefer concrete labels such as `Model`, `RMSE`, `MAE`, `WAPE`, `Read`, `Artifact`, `Details`, and `Result`.
- If benchmark code affecting reported metrics changes, rerun the affected maintained forecast benchmarks before updating public benchmark claims. Do not rely on stale artifact values when the user expects current benchmark evidence.

## Testing instructions
- Run `just validate` for full local validation.
- Always run lint/format checks before pushing. All linting and format checks must go through `uv run --group dev pre-commit run --all-files`; report that command in the handoff.
- For targeted Rust behavior checks, run `cargo test --workspace` or the narrower relevant `cargo test` command.
- For targeted Python behavior checks, run `uv run --group dev pytest` or the narrower relevant pytest command.
- For Python support matrix changes, do not commit claims for versions you have not validated end-to-end through CI/build logs.
  If an interpreter fails native build checks (for example PyO3 compatibility limits), revert the version claim and CI matrix entries until support is real.
- Add or update tests for behavioral changes, especially when changing splitters, serialization, CLI output, Python estimator behavior, or native bindings.
- For benchmark work, capture RMSE, MAE, R2, training time, prediction time, model settings, sample size, task names, and split names. Preserve output artifacts only when they are intentionally committed evidence; otherwise write generated runs under `target/` or `/tmp`.
- Be sure to update documentation and llms.txt for navigation

## PR instructions
- Summarize which surface changed: core Rust, CLI, PyO3 bindings, Python API, docs, tests, fixtures, benchmarks, or scripts.
- Mention targeted validation commands that were run.
- If model outputs, golden files, fixtures, or benchmark artifacts changed, explain why the updated outputs are expected.
- If a PR changes benchmark behavior or claims, include the exact command used and identify whether the data was synthetic, generated acceptance data, or real benchmark data.
