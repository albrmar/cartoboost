# Getting Started

This guide installs the local package, trains a small regression model, and
checks prediction from Python and the CLI.

## Requirements

- Rust stable toolchain.
- Python 3.10 or newer.
- `uv` 0.7 or newer.
- A checkout of the GeoBoost repository.

## Install

```sh
uv sync --group dev
uv run --group dev maturin develop
```

`maturin develop` builds `geoboost._native` and installs it into the `uv`
environment. The Python estimator requires this native extension.

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
)
model.fit(X, y)

predictions = model.predict([[1.5], [2.5]])
```

For a temporal-spatial model, add splitters that match the feature structure:

```python
model = GeoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24"],
)
```

Use `periodic:24` for hour-of-day, diagonal or Gaussian 2D splitters for
coordinate pairs, and `sparse_set` when rows have route-cell or zone
memberships.

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

The CLI expects dense numeric CSV input. Use the Python estimator for sparse-set
route-cell features.

## Save And Reload

```python
model.save("model.geoboost.json")
loaded = GeoBoostRegressor.load("model.geoboost.json")

model.save_weights("model.weights.json")
weights_loaded = GeoBoostRegressor.load_weights("model.weights.json")
```

Use `save` for GeoBoost JSON model artifacts and `save_weights` for portable
prediction artifacts. ONNX export is available only for dense axis-tree
constant-leaf models when the optional `onnx` dependency is installed.

## Run Local Checks

```sh
just validate
```

For a faster Python-focused loop:

```sh
uv run --group dev ruff format --check python tests scripts
uv run --group dev ruff check python tests scripts
uv run --group dev pytest
```

## Out-Of-Time Validation

For temporal-spatial problems, hold out the latest rows before trusting model
quality:

```python
from geoboost import out_of_time_split

train_idx, validation_idx = out_of_time_split(
    pickup_times,
    validation_fraction=0.2,
)

model.fit(X_train_all[train_idx], y_all[train_idx])
prediction = model.predict(X_train_all[validation_idx])
```

See [Evaluation Protocol](evaluation_protocol.md#out-of-time-validation) for
cutoff dates, exact validation sizes, gaps, pandas indexing, and sparse-set
examples.
