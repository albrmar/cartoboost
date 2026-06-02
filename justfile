set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    just --list

fmt:
    cargo fmt --all
    python3 -m ruff format python tests scripts

lint:
    cargo clippy --workspace --all-targets -- -D warnings
    python3 -m ruff format --check python tests scripts
    python3 -m ruff check python tests scripts

test:
    cargo test --workspace
    python3 -m pytest

build:
    python3 -m maturin build --release

develop:
    python3 -m maturin develop

sdist:
    python3 -m maturin sdist

wheel:
    python3 -m maturin build --release

validate:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
    python3 -m ruff format --check python tests scripts
    python3 -m ruff check python tests scripts
    python3 -m pytest
    cargo bench --workspace --no-run

clean:
    cargo clean
    rm -rf build dist target wheels *.egg-info .pytest_cache .ruff_cache
