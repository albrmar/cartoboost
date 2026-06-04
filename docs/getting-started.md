# Getting Started

This guide builds the native extension, trains a small regression model, and
checks prediction from Python and the CLI.

## Requirements

- Rust stable toolchain.
- Python 3.10 or newer.
- `uv` 0.7 or newer.
- A checkout of the GeoBoost repository.

## Development Install

```sh
uv sync --group dev
uv run --group dev maturin develop
```

`maturin develop` builds `geoboost._native` and installs it into the `uv`
environment. Use `backend="rust"` while developing native behavior so missing
bindings fail clearly.

## Train From Python

```python
from geoboost import GeoBoostRegressor

X = [[0.0], [1.0], [2.0], [3.0]]
y = [0.0, 1.0, 2.0, 3.0]

model = GeoBoostRegressor(
    n_estimators=20,
    learning_rate=0.1,
    max_depth=2,
    min_samples_leaf=1,
    splitters=["axis"],
    backend="rust",
)
model.fit(X, y)

predictions = model.predict([[1.5], [2.5]])
```

## Train From Dense CSV

Create a small file named `train.csv`:

```csv
distance,hour,target
1.0,8,10.0
2.0,9,12.0
3.0,17,18.0
4.0,18,20.0
```

Create `config.toml`:

```toml
target = "target"
n_estimators = 20
learning_rate = 0.1
max_depth = 2
min_samples_leaf = 1
splitter = "axis"
```

Run:

```sh
cargo run -p geoboost-cli -- train --data train.csv --config config.toml --model-out model.json
cargo run -p geoboost-cli -- predict --model model.json --input train.csv --predictions-out predictions.csv
```

The CLI v1 path expects dense numeric CSV input. Sparse-set route-cell features
are a Python API feature.

## Save And Reload

```python
model.save("model.geoboost.json")
loaded = GeoBoostRegressor.load("model.geoboost.json")

model.save_weights("model.weights.json")
weights_loaded = GeoBoostRegressor.load_weights("model.weights.json")
```

Use `save` for native model artifacts and `save_weights` for the explicit
weights-artifact contract. ONNX export is available only for dense axis-tree
constant-leaf models when the optional `onnx` dependency is installed.

## Verify The Checkout

```sh
just validate
```

For a faster Python-focused loop:

```sh
uv run --group dev ruff format --check python tests scripts
uv run --group dev ruff check python tests scripts
uv run --group dev pytest
```
