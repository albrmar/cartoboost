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
    uv run --group dev maturin build --release --locked --out dist

develop:
    uv run --group dev maturin develop

sdist:
    uv run --group dev maturin sdist --out dist

wheel:
    uv run --group dev maturin build --release --locked --out dist

validate:
    uv sync --group dev
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
    uv run --group dev ruff format --check python tests scripts
    uv run --group dev ruff check python tests scripts
    uv run --group dev maturin develop
    uv run --group dev pytest
    uv run --group dev python scripts/run_full_validation.py
    uv run --group dev python scripts/run_v1_validation.py
    cargo bench --workspace --no-run

nyc-quality-benchmark:
    uv run --group dev maturin develop --release
    PYTHONPATH=python uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py

nyc-quality-benchmark-smoke:
    PYTHONPATH=python uv run --group dev --group bench python scripts/run_nyc_taxi_quality_benchmarks.py --synthetic-smoke --models mean

nyc-quality-benchmark-repeated:
    uv run --group dev maturin develop --release
    PYTHONPATH=python uv run --group dev --group bench python scripts/run_repeated_nyc_taxi_benchmarks.py --no-download

clean:
    cargo clean
    rm -rf build dist target wheels *.egg-info .pytest_cache .ruff_cache .venv
