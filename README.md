# geoboost

Clean-room GeoBoost-inspired regression tools exposed through Rust, Python, and
a small command-line interface.

## Project Layout

- `crates/geoboost-core`: Rust library for core regression logic.
- `crates/geoboost-cli`: Command-line interface for training, prediction, evaluation, and inspection workflows.
- `crates/geoboost-py`: PyO3/maturin extension crate that publishes `geoboost._native`.
- `python/geoboost`: Python package surface.
- `configs`: Example training configuration files.
- `docs`: Project and contract documentation.
- `tests`: Python tests.

The full clean-room repository plan is tracked in
[`docs/repo_plan.md`](docs/repo_plan.md).
Generated proof images for spatial segmentation are committed under
[`docs/assets`](docs/assets).
Additional phase-style proof images for axis, diagonal, gaussian, periodic,
fuzzy, linear-leaf, sparse-set, and learning-rate fixtures are committed under
[`docs/assets/splitter_tests`](docs/assets/splitter_tests).
Generated acceptance metrics and smoke-test reports are written under
`target/validation/` by `uv run --group dev python scripts/run_full_validation.py`.
They cover axis, diagonal, gaussian, periodic, fuzzy, linear-leaf, sparse-set,
and learning-rate shrinkage fixtures.

## Python API

The Python package exposes `GeoBoostRegressor`:

```python
from geoboost import GeoBoostRegressor

model = GeoBoostRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X, y)
predictions = model.predict(X)
```

The estimator is compatible with sklearn-style workflows such as `Pipeline`,
`GridSearchCV`, `clone`, and NumPy-array predictions.

## Requirements

- Stable Rust, installed automatically by `rust-toolchain.toml` when using rustup.
- Python 3.10 or newer.
- uv 0.7 or newer.

## Development

Create a virtual environment, install development tooling, and build the extension in editable mode:

```sh
uv sync --group dev
uv run --group dev maturin develop
```

Common commands are available through either `make` or `just`:

```sh
make fmt
make lint
make test
make build
```

The `test` target runs both the Rust workspace tests and the Python test suite.

## Continuous Integration

The GitHub Actions workflow checks:

- Rust formatting, clippy, and workspace tests.
- Python linting and tests across Python 3.10 through 3.13.
- Release wheel builds with maturin.

## License

MIT
