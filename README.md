# CartoBoost

[![PyPI](https://img.shields.io/pypi/v/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![Python](https://img.shields.io/pypi/pyversions/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![CI](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml)
[![Docs](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml)
[![Publish](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/publish-pypi.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CartoBoost is a Python regression toolkit for temporal, spatial, geotemporal,
and graph-derived prediction problems. It keeps the estimator workflow familiar
to scikit-learn users while adding modeling primitives for place, time, sparse
taxi-zone membership, pickup-dropoff directionality, and learned graph context.

Use CartoBoost when a standard tabular booster is a strong baseline, but your
problem still requires hand-built features to represent:

- wraparound time such as hour-of-day, weekday, or seasonal cycles;
- 2D spatial boundaries, corridors, depots, hotspots, and service regions;
- list-valued memberships such as pickup zones, dropoff zones, or H3 cells;
- directed movement such as pickup-to-dropoff taxi flows;
- high-cardinality IDs that benefit from learned embeddings.

## Core Capabilities

CartoBoost supports:

- L2 and quantile regression objectives.
- Constant and linear residual leaves.
- Axis, histogram-axis, diagonal 2D, Gaussian/radial 2D, periodic, sparse-set,
  and fuzzy split behavior.
- Dense numeric arrays plus list-valued sparse-set features.
- Feature schemas for numeric, periodic, sparse-set, and model-contract
  validation.
- JSON model artifacts and portable weights artifacts.
- Optional SHAP explanations, Optuna tuning, Polars input support, and ONNX
  export for the supported dense axis-tree subset.
- Standalone neural embedding regressors for high-cardinality IDs, plus optional
  neural feature-generation workflows.
- Standalone node2vec, GraphSAGE, heterogeneous GraphSAGE, and typed-schema
  HinSAGE graph regressors and link predictors, plus optional graph feature
  encoders.
- General Rust-backed utilities outside the forecasting API, including
  single-series forecast helpers, local-level/local-linear Kalman filters,
  Croston/SBA/TSB intermittent demand, and ordinary kriging.
- Rust-native forecasting APIs for geographic and temporal single-series or
  panel taxi demand, including rolling-origin backtests, naive/seasonal
  naive/theta/optimized-theta/ETS/AutoARIMA models, supervised CartoBoost lag
  forecasting, Rust-core weighted ensembles, CLI runs, and portable forecast
  artifacts.

## Forecasting

```python
from cartoboost.forecasting import ForecastFrame, ThetaForecaster

frame = ForecastFrame.from_pandas(
    taxi_lane_demand,
    timestamp_col="pickup_date",
    target_col="pickup_trips",
    series_id_col="pickup_dropoff_lane",
    freq="D",
)

model = ThetaForecaster(season_length=7)
model.fit(frame)
forecast = model.predict(horizon=14)
```

Forecast outputs use deterministic columns:
`series_id`, `timestamp`, `horizon`, `model`, and `mean`. The Python
forecasting surface is a wrapper over `cartoboost._native`; it does not compute
fallback forecasts when a native binding is missing. Use the forecasting docs
for geographic-temporal CLI usage, lag-feature forecasting, backtesting,
artifacts, and examples built around pickup/dropoff lanes and taxi-zone demand.

## Benchmarks

The benchmark reports focus on the data-science question behind each run:
dataset, target, feature set, split design, comparison models, metrics, and the
meaning of the result.

- NYC taxi benchmarks measure transformed trip duration, fare amount, and
  pickup-zone demand from pickup/dropoff zones, trip attributes, and hour/day
  features.
- Forecasting benchmarks measure daily pickup/dropoff lane demand with lagged
  demand, rolling summaries, calendar fields, zone IDs, airport-lane flags, and
  borough context. External forecasting comparisons name the exact libraries:
  StatsForecast and functime.
- Synthetic model-suite benchmarks isolate dense numeric signal, repeated-ID
  residual signal, and source-target graph signal against LightGBM and XGBoost.
- Taxi-zone acceptance benchmarks check whether lane membership, route
  geometry, and periodic hour behavior are recoverable before making broader
  quality claims.

## Install

Install the released package from PyPI:

```sh
uv add cartoboost
```

Optional integrations:

```sh
uv add "cartoboost[explain]"  # SHAP support
uv add "cartoboost[h3]"       # H3 lat/lon encoder
uv add "cartoboost[s2]"       # S2 lat/lon encoder
uv add "cartoboost[duckdb]"   # DuckDB relation inputs
uv add "cartoboost[optuna]"   # Optuna tuning
uv add "cartoboost[polars]"   # Polars inputs
uv add "cartoboost[onnx]"     # ONNX export subset
```

Verify the install:

```sh
python -c "import cartoboost; print(cartoboost.__version__)"
cartoboost --help
```

## Basic Regression

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    splitters=["axis"],
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The estimator supports sklearn-style `get_params`, `set_params`, `clone`,
`Pipeline`, `GridSearchCV`, and NumPy-array predictions.

## Temporal-Spatial Modeling

Use dense columns for numeric location and time features, and sparse-set columns
for memberships such as pickup zones, dropoff zones, or encoded H3 cells.

```python
from cartoboost import CartoBoostRegressor

schema = {
    "dense": [
        {"name": "pickup_x", "kind": "numeric"},
        {"name": "pickup_y", "kind": "numeric"},
        {"name": "hour_of_day", "kind": "periodic", "period": 24},
        {"name": "trip_distance", "kind": "numeric"},
    ],
    "sparse_sets": [
        {"name": "taxi_zones", "kind": "sparse_set"},
    ],
}

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=30,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24", "sparse_set"],
    fuzzy=True,
    fuzzy_bandwidth=0.05,
)

model.fit(
    X_train_dense,
    y_train,
    sparse_sets={"taxi_zones": taxi_zones_train},
    feature_schema=schema,
)

predictions = model.predict(
    X_test_dense,
    sparse_sets={"taxi_zones": taxi_zones_test},
)
```

Why this helps:

- `periodic:24` treats midnight-adjacent hours as neighbors.
- `diagonal_2d` learns oblique spatial boundaries more directly than axis-only
  trees.
- `gaussian_2d` isolates radial neighborhoods around local hotspots.
- `sparse_set` splits on list-valued route or cell membership without a wide
  one-hot matrix.
- `fuzzy=True` reduces hard jumps near spatial or temporal boundaries.

## Graph Models

Graph models run independently through `Node2VecStandaloneRegressor`,
`GraphSageStandaloneRegressor`, `HeteroGraphSageStandaloneRegressor`,
`HinSageStandaloneRegressor`, and the matching standalone link predictors.
Supported graph families are node2vec, GraphSAGE, HeteroGraphSAGE, and HinSAGE.
Direction is a first-class contract: `A -> B` and `B -> A` can be separate
facts, features, and embeddings.

Graph encoders can also emit graph-derived columns for another estimator when
you explicitly want a feature-generation workflow.

See [Graph Models And Features](docs/graph-features.md) for standalone graph
regressors, standalone link predictors, encoder configs, directional features,
OD-pair nodes, metapaths, artifacts, and benchmark guidance.

## Neural Embedding Models

Use `NeuralEmbeddingStandaloneRegressor` for direct supervised ID modeling
without a boosted wrapper.

```python
from cartoboost import NeuralEmbeddingStandaloneRegressor

model = NeuralEmbeddingStandaloneRegressor(dim=16, random_state=7)
model.fit(pickup_zone_ids_train, y_train, dense=X_train)
predictions = model.predict(pickup_zone_ids_test, dense=X_test)
```

Use `NeuralEmbeddingRegressor` when high-cardinality IDs carry stable signal and
you explicitly want learned dense embeddings appended to a tabular model input.

```python
from cartoboost import NeuralEmbeddingRegressor

model = NeuralEmbeddingRegressor(
    dim=16,
    use_residual=True,
    base_model_kwargs={"n_estimators": 80, "splitters": ["axis"]},
    final_model_kwargs={"n_estimators": 120, "splitters": ["axis", "periodic:24"]},
)

model.fit(X_train, y_train, ids=ids_train)
predictions = model.predict(X_test, ids=ids_test)
```

For a quick head-to-head comparison on one split:

```python
from cartoboost import benchmark_neural_vs_cartoboost

results = benchmark_neural_vs_cartoboost(X, y, ids, split_ratio=0.8)
```

Use this helper as an initial signal check, then validate with your real
temporal, spatial, grouped, or out-of-time split.

See [Neural Embedding Models And Features](docs/neural-features.md) for the
standalone neural API, artifacts, fallback behavior, and optional feature
generation workflow.

## Save, Load, And Explain

```python
model.save("model.cartoboost.json")
loaded = CartoBoostRegressor.load("model.cartoboost.json")

explanation = loaded.explain_shap(
    X_test_dense,
    background=X_train_dense,
    sparse_sets={"taxi_zones": taxi_zones_test},
    background_sparse_sets={"taxi_zones": taxi_zones_train},
)
```

Model artifacts are versioned JSON and include optional metadata, feature
schema, and training configuration fields. Graph and neural standalone artifacts
are complete model artifacts. Feature-generation artifacts should be persisted
with whichever downstream model consumes their generated columns.

## CLI

The CLI supports dense numeric CSV train, predict, eval, and inspect workflows.
Use the Python API for list-valued sparse taxi-zone features and graph-derived
feature pipelines.

```sh
cartoboost train --data train.csv --config configs/regression.toml --model-out model.json
cartoboost predict --model model.json --input test.csv --predictions-out predictions.csv
cartoboost eval --model model.json --data test_with_target.csv
```

## Documentation

- [Documentation Home](docs/index.md)
- [Installation](docs/installation.md)
- [Getting Started](docs/getting-started.md)
- [Python Estimator](docs/user-guide/python-estimator.md)
- [Parameters](docs/user-guide/parameters.md)
- [Spatial Modeling](docs/spatial_modeling.md)
- [Graph Models And Features](docs/graph-features.md)
- [Neural Embedding Models And Features](docs/neural-features.md)
- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Feature Schema](docs/feature_schema.md)
- [Sparse Features](docs/sparse_features.md)
- [Model Artifacts](docs/model_artifact.md)
- [Python API Reference](docs/reference/python-api.md)
- [CLI Reference](docs/reference/cli.md)
- [Benchmarks](docs/benchmarks/index.md)
