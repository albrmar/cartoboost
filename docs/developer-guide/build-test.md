# Build And Test

## Full Validation

```sh
just validate
```

This is the local source of truth. It runs Rust formatting, clippy, Rust tests,
Python formatting/linting, native extension installation, pytest, validation
reports, and benchmark compilation.

## Targeted Rust Loop

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Run this loop after changing `crates/geoboost-core`, `crates/geoboost-cli`, or
`crates/geoboost-py`.

## Targeted Python Loop

```sh
uv sync --group dev
uv run --group dev maturin develop
uv run --group dev ruff format --check python tests scripts
uv run --group dev ruff check python tests scripts
uv run --group dev pytest
```

Run `maturin develop` after native binding changes so `geoboost._native` is
available to Python tests.

## Wheel Builds

```sh
uv run --group dev maturin build --release --locked --out dist
```

CI builds CPython 3.10, 3.11, 3.12, and 3.13 wheels for these targets:

- Linux x86_64: `x86_64-unknown-linux-gnu`
- Linux arm64: `aarch64-unknown-linux-gnu`
- macOS x86_64: `x86_64-apple-darwin`
- macOS arm64: `aarch64-apple-darwin`
- Windows x86_64: `x86_64-pc-windows-msvc`
- Windows arm64: `aarch64-pc-windows-msvc`

Linux wheels use manylinux2014 compatibility. The workflow also builds a source
distribution so unsupported platforms can fall back to a local Rust build.

## Docs Loop

```sh
uv sync --group docs --no-install-project
uv run --group docs --no-sync mkdocs serve
uv run --group docs --no-sync mkdocs build --strict
```

`mkdocs build --strict` is the same docs validation used by the Pages workflow.

## Validation Scripts

```sh
uv run --group dev python scripts/run_full_validation.py
uv run --group dev python scripts/run_v1_validation.py
```

These scripts write deterministic artifacts under `target/validation/`. Do not
commit generated target output unless a task explicitly asks for it.

## Benchmarks

```sh
cargo bench --workspace --no-run
```

Benchmark compilation is part of validation. Numeric benchmark claims require a
documented reproducible setup; compile success alone is not a performance
claim.

Benchmark runbooks and generated artifact rules live in
[Benchmarks](../benchmarks/index.md).
