# Developer Guide

The developer workflow is designed around a small number of explicit contracts:
core behavior lives in Rust, Python provides estimator ergonomics, the CLI stays
dense-CSV focused, and docs describe only behavior that is implemented and
tested.

## First Commands

```sh
uv sync --group dev
uv run --group dev maturin develop
just validate
```

For docs:

```sh
uv sync --group docs --no-install-project
uv run --group docs --no-sync mkdocs serve
uv run --group docs --no-sync mkdocs build --strict
```

## Development Principles

- Keep shared model behavior in `crates/geoboost-core`.
- Keep command-line behavior in `crates/geoboost-cli`.
- Keep PyO3 binding mechanics in `crates/geoboost-py`.
- Keep Python API ergonomics in `python/geoboost`.
- Add tests for behavioral changes, especially splitters, serialization, CLI
  output, estimator behavior, and native bindings.
- Do not edit generated output under `target/`, `.venv/`, `.pytest_cache/`,
  `data/`, or generated plot folders unless the task explicitly requires
  generated artifacts.

## High-Value Pages

- [Architecture](architecture.md) explains ownership boundaries.
- [Build And Test](build-test.md) lists validation commands and when to run
  them.
- [Extending GeoBoost](extending.md) shows where to add new splitters, losses,
  artifact fields, bindings, and CLI behavior.
- [Documentation](documentation.md) explains the GitHub Pages setup and docs
  quality bar.
- [Release Process](release.md) summarizes release readiness checks.
