# Installation

CartoBoost is published on PyPI as `cartoboost`.

## Install From PyPI

```sh
uv add cartoboost
```

The published wheels target CPython 3.10, 3.11, 3.12, and 3.13 on:

- Linux x86_64 and aarch64 with manylinux2014 compatibility.
- macOS x86_64 and arm64.
- Windows x86_64 and arm64.

If no compatible wheel exists, `uv` may try to build from source, which requires
the project build toolchain.

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

Python usage should work immediately after install:

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(n_estimators=10, max_depth=2)
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError` during import | Reinstall CartoBoost in a clean Python environment. |
| `uv` tries to compile from source | Use CPython 3.10-3.13 on a supported platform, or install the project build toolchain before building. |
| `cartoboost` command not found | Make sure the Python environment where `cartoboost` was installed is active. |
