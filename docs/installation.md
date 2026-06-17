# Installation

CartoBoost is published on PyPI as `cartoboost`. The PyPI package includes the
Rust native extension used by `CartoBoostRegressor`, so most users do not need a
Rust toolchain.

## Install From PyPI

```sh
uv add cartoboost
```

The published wheels target CPython 3.10, 3.11, 3.12, and 3.13 on:

- Linux x86_64 and aarch64 with manylinux2014 compatibility.
- macOS x86_64 and arm64.
- Windows x86_64 and arm64.

If a matching wheel is available, `uv` installs the compiled Rust extension
directly. If no compatible wheel exists, `uv` may try to build from source,
which requires Rust and the Python build toolchain.

## Optional Extras

Install optional integrations with extras:

```sh
uv add "cartoboost[explain]"
uv add "cartoboost[optuna]"
uv add "cartoboost[polars]"
uv add "cartoboost[onnx]"
```

| Extra | Adds |
| --- | --- |
| `explain` | SHAP explanations. |
| `optuna` | Hyperparameter tuning examples and workflows. |
| `polars` | Polars input support. |
| `onnx` | ONNX export for the supported dense axis-tree subset. |

## Verify The Install

```sh
python -c "import cartoboost; print(cartoboost.__version__)"
cartoboost --help
```

Python usage should import without any separate native build step:

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(n_estimators=10, max_depth=2)
```

## Source Checkout

Use a source checkout for development, local validation, or benchmark scripts
that rely on repository files:

```sh
git clone https://github.com/TheCulliganMan/CartoBoost.git
cd CartoBoost
uv sync --group dev
uv run --group dev maturin develop
```

For a release-mode local extension, useful for benchmarks:

```sh
uv run --group dev maturin develop --release
```

Run local validation with:

```sh
just validate
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError: cartoboost._native` | Reinstall from PyPI or run `uv run --group dev maturin develop` in a source checkout. |
| `uv` tries to compile from source | Use CPython 3.10-3.13 on a supported platform, or install Rust before building. |
| `cartoboost` command not found | Make sure the Python environment where `cartoboost` was installed is active. |
