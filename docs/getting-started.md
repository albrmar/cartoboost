# Getting Started

This guide installs CartoBoost from PyPI, trains a small regression model, and
checks prediction from Python and the CLI.

## Requirements

- Python 3.10 or newer.
- `uv`.

## Install

```sh
uv add cartoboost
```

The PyPI package includes `cartoboost._native`, the Rust native extension used
for training and prediction.

Verify the install:

```sh
python -c "import cartoboost; print(cartoboost.__version__)"
cartoboost --help
```

## Train From Python

```python
from cartoboost import CartoBoostRegressor

X = [[0.0], [1.0], [2.0], [3.0]]
y = [0.0, 1.0, 2.0, 3.0]

model = CartoBoostRegressor(
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
model = CartoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24"],
)
```

Use `periodic:24` for hour-of-day, diagonal or Gaussian 2D splitters for
coordinate pairs, and `sparse_set` when rows have route-cell or zone
memberships.

## Add Neural Features (Hybrid)

Use `NeuralEmbeddingRegressor` for neural features: learn a small ID-based
embedding table from residuals via the Rust-backed trainer, then append those
dense vectors to your model input.

```python
import numpy as np
from cartoboost import NeuralEmbeddingRegressor

X_train = np.array(
    [
        [0.2, 8.0],
        [1.4, 9.0],
        [2.1, 17.0],
        [3.3, 18.0],
    ]
)
y_train = np.array([10.5, 11.0, 12.4, 13.2], dtype=float)
ids_train = np.array([101, 102, 101, 103], dtype=np.uint64)

X_test = np.array([[1.0, 16.0], [3.0, 19.0]])
ids_test = np.array([101, 104], dtype=np.uint64)

neural_model = NeuralEmbeddingRegressor(
    dim=8,
    base_model_kwargs=dict(
        n_estimators=40,
        learning_rate=0.08,
        max_depth=3,
        min_samples_leaf=2,
        splitters=["axis"],
    ),
    final_model_kwargs=dict(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=4,
        splitters=["axis", "periodic:24"],
    ),
)

neural_model.fit(X_train, y_train, ids=ids_train)
predictions = neural_model.predict(X_test, ids=ids_test)
```

Pass `ids` directly as shown above, or pass `id_column="cell_id"` with a pandas
dataframe input. Neural features are added as additional numeric columns before
training and inference.

See [Neural Features](neural-features.md) for full pipeline details, artifact
format, and benchmarks.

## Add Graph Features

Use graph features when rows depend on relationships between entities, places,
or source-target pairs. Graph encoders are implemented in Rust and exposed
through Python wrappers.

```python
from cartoboost.graph import (
    DirectionalFeature,
    DirectionalityConfig,
    GraphEmbeddingsConfig,
    GraphEncoderConfig,
    GraphEncoderFamily,
    GraphFeatureTransformer,
)

config = GraphEmbeddingsConfig(
    encoder=GraphEncoderConfig(
        family=GraphEncoderFamily.HINSAGE,
        input_dim=2,
        node_type_count=2,
        edge_type_triples=((0, 0, 1), (1, 1, 0)),
        neighbor_samples=(10, 10),
    ),
    directionality=DirectionalityConfig(
        preserve_source_target_roles=True,
        compute_asymmetry_features=True,
        directional_features=(DirectionalFeature.SOURCE_TARGET_EMBEDDING,),
    ),
)

transformer = GraphFeatureTransformer.from_config(config)
bundle = transformer.fit_transform(
    node_features,
    edges=typed_edges,
    node_types=node_types,
)
```

The resulting `GraphFeatureBundle` contains dense graph columns and provenance
metadata that can be appended to your model input. See
[Graph Features](graph-features.md) for node2vec, GraphSAGE,
HeteroGraphSAGE, HinSAGE, directional features, and directed metapaths.

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
cartoboost train --data train.csv --config config.toml --model-out model.json
cartoboost predict --model model.json --input train.csv --predictions-out predictions.csv
```

The CLI expects dense numeric CSV input. Use the Python estimator for sparse-set
route-cell features.

## Save And Reload

```python
model.save("model.cartoboost.json")
loaded = CartoBoostRegressor.load("model.cartoboost.json")

model.save_weights("model.weights.json")
weights_loaded = CartoBoostRegressor.load_weights("model.weights.json")
```

Use `save` for CartoBoost JSON model artifacts and `save_weights` for portable
prediction artifacts. ONNX export is available only for dense axis-tree
constant-leaf models when the optional `onnx` dependency is installed.

## Run Local Checks

Local checks are for source checkouts. Clone the repository and build the native
extension into the development environment first:

```sh
git clone https://github.com/TheCulliganMan/CartoBoost.git
cd CartoBoost
uv sync --group dev
uv run --group dev maturin develop
```

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
from cartoboost import out_of_time_split

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
