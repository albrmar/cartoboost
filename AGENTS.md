# CartoBoost Agent Guide

## Dev environment tips
- This repository is a Rust workspace with Python bindings and a Python package for CartoBoost regression tooling.
- Use `uv sync --group dev` to prepare the Python development environment.
- Use `uv run --group dev maturin develop` after native binding changes so `cartoboost._native` is available to Python tests.
- Keep shared model behavior in `crates/cartoboost-core`; keep CLI-only behavior in `crates/cartoboost-cli`, PyO3 bindings in `crates/cartoboost-py`, and Python API ergonomics in `python/cartoboost`.
- Do not edit build output, virtualenvs, caches, downloaded data, or benchmark output under `target/`, `.venv/`, `.pytest_cache/`, `data/`, or generated plot folders unless the task explicitly asks for generated artifacts.

## Testing instructions
- Run `just validate` for full local validation.
- For targeted Rust checks, run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace`.
- For targeted Python checks, run `uv run --group dev ruff format --check python tests scripts`, `uv run --group dev ruff check python tests scripts`, and `uv run --group dev pytest`.
- For Python support matrix changes, do not commit claims for versions you have not validated end-to-end through CI/build logs.
  If an interpreter fails native build checks (for example PyO3 compatibility limits), revert the version claim and CI matrix entries until support is real.
- Add or update tests for behavioral changes, especially when changing splitters, serialization, CLI output, Python estimator behavior, or native bindings.

## PR instructions
- Summarize which surface changed: core Rust, CLI, PyO3 bindings, Python API, docs, tests, fixtures, benchmarks, or scripts.
- Mention targeted validation commands that were run.
- If model outputs, golden files, fixtures, or benchmark artifacts changed, explain why the new outputs are expected.
