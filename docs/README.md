# CartoBoost Documentation

This directory contains the scientist-facing documentation for deciding when
CartoBoost is the right model family, fitting it responsibly, evaluating it
against serious baselines, explaining outputs, and saving reproducible
artifacts for temporal, spatial, geotemporal, forecasting, graph, and
neural-embedding workflows.

## Contents

- [Getting Started](getting-started.md)
- [Feature Catalog](feature_catalog.md)
- [Choose A Model](user-guide/model-types.md)
- [Python API Reference](reference/python-api.md)
- [Parameters](user-guide/parameters.md)
- [Forecasting Wrapper](forecasting.md)
- [Model Guides](user-guide/forecasting-models/index.md)
- [Objectives](objectives.md)
- [Constraints](constraints.md)
- [Spatial Modeling](spatial_modeling.md)
- [Evaluation Protocol](evaluation_protocol.md)
- [Feature Schema](feature_schema.md)
- [Sparse Features](sparse_features.md)
- [Graph Features](graph-features.md)
- [Neural Features](neural-features.md)
- [SHAP Support](shap.md)
- [CLI](user-guide/cli.md)
- [Model Artifacts](model_artifact.md)
- [Python API Reference](reference/python-api.md)
- [CLI Reference](reference/cli.md)
- [Benchmarks](benchmarks/index.md)
  - [Fair Benchmarking Program](benchmarks/fair-benchmarking.md)
  - [Neural Embedding Strategy Assessment](benchmarks/neural-embedding-strategy.md)

CartoBoost is most useful when time, place, route membership, and directed
relationships are part of the scientific question instead of incidental columns.
Examples include pickup-zone demand that wraps across midnight, fare or
duration residuals that vary by airport lane, spatially blocked validation
where nearby taxi zones should not leak into the holdout, and source-target
flows where `PULocationID -> DOLocationID` is not interchangeable with the
reverse trip.

Choose CartoBoost when you want to test whether explicit temporal-spatial,
sparse-zone, graph, or native forecasting structure improves a measured outcome
under the same train/test split as strong baselines such as LightGBM or XGBoost.
The docs emphasize those modeling decisions first; API details and code snippets
exist to make the chosen workflow reproducible.

## Quick Start

```sh
uv add cartoboost
```

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    splitters=["axis"],
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The CLI is available for dense numeric CSV workflows:

```sh
cartoboost train --data train.csv --config configs/regression.toml --model-out model.json
cartoboost predict --model model.json --input test.csv --predictions-out predictions.csv
```

Use the Python API for sparse-set, neural embedding, and graph-derived feature
pipelines.

## Site Presentation

The documentation site is built with Docusaurus. The custom homepage lives in
`src/pages/index.tsx`, the docs sidebar lives in `sidebars.ts`, and the branded
theme layer lives in `src/css/custom.css`. Keep React components focused on
navigation, presentation, and docs-specific visualizations; modeling behavior
and examples still belong in the Rust, Python, and Markdown sources described
above.
