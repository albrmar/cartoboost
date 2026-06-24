set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    just --list

fmt:
    cargo fmt --all
    uv run ruff format python tests scripts

lint:
    uv run pre-commit run --all-files

test:
    cargo test --workspace
    uv run pytest

build:
    uv run maturin build --release --locked --out dist

develop:
    uv run maturin develop

pre-commit-install:
    uv run pre-commit install

pre-commit:
    uv run pre-commit run --all-files

sdist:
    uv run maturin sdist --out dist

wheel:
    uv run maturin build --release --locked --out dist

validate:
    uv sync
    uv run pre-commit run --all-files
    cargo test --workspace
    uv run maturin develop
    uv run pytest
    uv run python scripts/run_full_validation.py
    uv run python scripts/run_v1_validation.py
    cargo bench --workspace --no-run

nyc-quality-benchmark:
    uv run maturin develop --release
    PYTHONPATH=python uv run --group bench python scripts/run_nyc_taxi_quality_benchmarks.py

nyc-quality-benchmark-smoke:
    PYTHONPATH=python uv run --group bench python scripts/run_nyc_taxi_quality_benchmarks.py --synthetic-smoke --models mean

nyc-quality-benchmark-repeated:
    uv run maturin develop --release
    PYTHONPATH=python uv run --group bench python scripts/run_repeated_nyc_taxi_benchmarks.py --no-download

model-benchmark-suite:
    PYTHONPATH=python uv run --group bench python scripts/run_model_benchmark_suite.py

clean:
    cargo clean
    rm -rf build dist target wheels *.egg-info .pytest_cache .ruff_cache .venv
