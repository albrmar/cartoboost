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

## Python API

The Python package exposes `GeoBoostRegressor`:

```python
from geoboost import GeoBoostRegressor

model = GeoBoostRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X, y)
predictions = model.predict(X)
```

## Requirements

- Stable Rust, installed automatically by `rust-toolchain.toml` when using rustup.
- Python 3.10 or newer.
- maturin 1.7 or newer.

## Development

Create a virtual environment, install development tooling, and build the extension in editable mode:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
maturin develop
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
