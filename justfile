set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    just --list

fmt:
    cargo fmt --all
    uv run --group dev ruff format python tests scripts

lint:
    cargo clippy --workspace --all-targets -- -D warnings
    uv run --group dev ruff format --check python tests scripts
    uv run --group dev ruff check python tests scripts

test:
    cargo test --workspace
    uv run --group dev pytest

build:
    uv run --group dev maturin build --release

develop:
    uv run --group dev maturin develop

sdist:
    uv run --group dev maturin sdist

wheel:
    uv run --group dev maturin build --release

validate:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
    uv run --group dev ruff format --check python tests scripts
    uv run --group dev ruff check python tests scripts
    uv run --group dev pytest
    uv run --group dev python scripts/run_full_validation.py
    cargo bench --workspace --no-run

clean:
    cargo clean
    rm -rf build dist target wheels *.egg-info .pytest_cache .ruff_cache .venv
