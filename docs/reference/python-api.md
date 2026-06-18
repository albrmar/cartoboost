# Python API Reference

This page lists the public Python entry points used to fit, evaluate, explain,
and save CartoBoost regression models.

## `cartoboost.CartoBoostRegressor`

```python
CartoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    min_gain=1e-8,
    loss="l2",
    quantile_alpha=0.5,
    huber_delta=1.0,
    log_offset=1.0,
    loss_params=None,
    splitters=None,
    leaf_predictor="constant",
    linear_leaf_features=None,
    fuzzy=False,
    fuzzy_bandwidth=0.0,
    fuzzy_kernel="linear",
    l2_regularization=1.0,
    constant_l2_regularization=0.0,
    random_state=None,
    n_threads=None,
    monotonic_constraints=None,
)
```

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, sample_weight=None, feature_schema=None, sparse_sets=None, eval_set=None)` | `self` | `eval_set` is accepted but currently ignored. |
| `predict(X, sparse_sets=None)` | `numpy.ndarray` | Requires matching dense width and sparse columns. |
| `predict_additive_values(X, sparse_sets=None)` | `numpy.ndarray` | Row sums equal `predict(X)`. |
| `make_shap_explainer(background, **kwargs)` | SHAP explainer | Requires optional SHAP dependency. |
| `explain_shap(X, background=..., **kwargs)` | `shap.Explanation` | Convenience SHAP entry point. |
| `save(path)` | `None` | Writes a model artifact. |
| `save_weights(path, format="auto")` | `None` | Writes JSON weights or supported ONNX. |
| `CartoBoostRegressor.load(path)` | estimator | Loads native model artifacts. |
| `CartoBoostRegressor.load_weights(path)` | estimator | Loads weights artifacts. |
| `get_params(deep=True)` | `dict` | sklearn-compatible parameter inspection. |
| `set_params(**params)` | `self` | Validates known parameter names. |

## `cartoboost.NeuralEmbeddingRegressor`

```python
regressor = NeuralEmbeddingRegressor(
    dim=16,
    fallback="global_mean_vector",
    random_state=None,
    neural_transformer=None,
    use_residual=True,
    oof_folds=1,
    drop_id_column=True,
    id_column=None,
    support_prior_strength=1.0,
    base_model_kwargs=None,
    final_model_kwargs=None,
)
```

Rust-native neural-augmented boosted estimator that appends offline
`NeuralEmbeddingFeatures` to dense features and trains `CartoBoostRegressor` on
the expanded matrix.

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, sample_weight=None, sparse_sets=None, id_column=None, ids=None, fallback_ids=None, neighbor_ids=None, **fit_kwargs)` | `self` | Pass 1D IDs for one key or 2D IDs for multi-key embeddings. `fallback_ids` provides hierarchical fallback chains; `neighbor_ids` provides graph-aware fallback. |
| `predict(X, sparse_sets=None, id_column=None, ids=None, fallback_ids=None, neighbor_ids=None)` | `numpy.ndarray` | Requires the same ID key count used during fit. |
| `transform(X, id_column=None, ids=None, fallback_ids=None, neighbor_ids=None)` | `numpy.ndarray` | Returns dense matrix with neural columns appended. |
| `score(X, y, sparse_sets=None, id_column=None, ids=None, fallback_ids=None, neighbor_ids=None)` | `float` | Computes mean absolute error on predictions. |
| `timings` | `dict[str, float]` | Fit timing components in milliseconds: `base_fit_ms`, `neural_fit_ms`, `final_fit_ms`. |

Set `oof_folds > 1` to train final-model embedding columns out of fold. Use
`support_prior_strength` to shrink rare IDs more strongly toward their prior.

## Graph Encoders

Rust-native neighborhood-based encoders for downstream boosting workflows.

`Node2VecEncoder` is for transductive directed/weighted random-walk embeddings;
`GraphSageEncoder` is for homogeneous graphs; `HeteroGraphSageEncoder` is for
typed relations; `HinSageEncoder` is the typed-schema HinSAGE surface with
native Rust validation, relation-aware sampling, and link feature construction.

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(node_count, edges, edge_weights=None)` | `list[list[float]]` | Trains `Node2VecEncoder` on directed edges with optional non-negative weights. |
| `fit(node_count, edges, node_features)` | `list[list[float]]` | Trains GraphSAGE encoder weights on an edge list and returns node embeddings. |
| `fit(node_types, edges, node_features)` | `list[list[float]]` | Trains `HinSageEncoder` on typed nodes and `(source, target, relation)` edges validated against `edge_type_triples`. |
| `encode(node_features)` | `list[list[float]]` | Encodes features with learned weights for inference. |
| `link_embeddings(embeddings, pairs)` | `list[list[float]]` | Builds HinSAGE link-prediction features as `[source, target, abs_delta, product]`. |
| `loss_curve()` | `list[float]` | Per-epoch training loss history. |
| `save_artifact_json(path)` | `None` | Persists deterministic encoder artifact. |
| `to_artifact_json()` | `str` | Emits JSON artifact payload. |
| `load_artifact_json(path)` | `Node2VecEncoder` / `GraphSageEncoder` / `HeteroGraphSageEncoder` / `HinSageEncoder` | Loads serialized encoder state. |

## `cartoboost.graph` feature helpers

The `cartoboost.graph` package contains the Python graph-feature layer used to
precompute dense and sparse graph inputs before fitting `CartoBoostRegressor`.

| Entry point | Purpose |
| --- | --- |
| `GraphFeatureConfig.from_config(cfg)` | Validates YAML-style graph config blocks with schema, directionality, metapaths, encoder settings, and outputs. |
| `GraphSchema`, `EdgeType`, `DirectionalityConfig` | Describe directed heterogeneous graph schemas and source-target requirements. |
| `DirectedMetaPath` | Validates typed node/relation/node metapaths against a `GraphSchema`. |
| `GraphFeatureTransformer.from_config(cfg)` | Fits native node2vec, GraphSAGE, HeteroGraphSAGE, or typed-schema HinSAGE encoders and emits a `GraphFeatureBundle`. |
| `Node2VecFeatureEncoder.from_config(cfg)` | Thin Python wrapper around Rust-native node2vec with `dim`, `walk_length`, `walks_per_node`, `window_size`, `p`, `q`, and optional edge weights. |
| `HinSageFeatureEncoder.from_config(cfg)` | Thin Python wrapper around Rust-native HinSAGE with `node_type_count`, `edge_type_triples`, and optional per-relation `neighbor_samples`. |
| `GraphFeatureBundle` | Carries dense graph columns, optional sparse sets, feature names, node IDs, and provenance metadata. |
| `MetaPathWalkGenerator`, `TemporalWalkGenerator` | Generate constrained directed metapath walks and monotonic temporal walks. |
| `materialize_source_target_pair_nodes(edges)` | Creates stable `("od_pair", source, target)` nodes so `A -> B` and `B -> A` stay distinct. |
| `link_prediction_report(labels, scores, query_ids=None, k=10)` | Reports AUC/AP and optional top-k/MRR ranking metrics. |

Directional source-target features are opt-in through
`directionality.compute_asymmetry_features`. Supported outputs include
`graph_source_target_embedding`, `graph_target_source_embedding`,
`graph_forward_reverse_similarity_delta`, `graph_source_outbound_strength`,
`graph_target_inbound_strength`, `graph_flow_imbalance_ratio`,
`graph_directed_temporal_drift`, and generic flow metrics such as
`graph_source_target_affinity` and `graph_flow_asymmetry`.

### Function: `cartoboost.benchmark_neural_vs_cartoboost`

```python
benchmark_neural_vs_cartoboost(
    X,
    y,
    ids,
    split_ratio=0.8,
    neural_kwargs=None,
    cartoboost_kwargs=None,
)
```

Returns:

- `structured_mae`
- `neural_mae` (reported as `hybrid_mae` in the current helper payload)
- `improvement`
- `cartoboost_fit_ms`
- `cartoboost_predict_ms`
- `neural_fit_ms` (reported as `hybrid_fit_ms` in the current helper payload)
- `neural_predict_ms` (reported as `hybrid_predict_ms` in the current helper payload)

Use this helper for quick, deterministic smoke comparisons on a held-out split.

## `cartoboost.FeatureSchema`

```python
FeatureSchema(dense, sparse_sets=None)
```

Helper for declaring numeric, periodic, and sparse-set feature roles.

| Method | Returns | Notes |
| --- | --- | --- |
| `to_dict()` | `dict` | Compact Python representation. |
| `to_rust_payload(dense_width, sparse_names)` | `dict` | Validated `names` and `kinds` payload. |
| `to_json(dense_width, sparse_names)` | `str` | JSON-encoded Rust payload. |

## SHAP Helpers

```python
cartoboost.make_shap_explainer(model, background, **kwargs)
cartoboost.explain_shap(model, X, background=..., **kwargs)
```

These functions are also available as estimator methods. See
[SHAP Support](../shap.md).

## Evaluation Helpers

```python
cartoboost.out_of_time_split(times, validation_fraction=0.2, gap=0)
cartoboost.spatial_blocked_cv(coordinates, n_splits=5)
cartoboost.temporal_blocked_cv(times, n_splits=5, gap=0)
cartoboost.grouped_blocked_cv(groups, n_splits=5)
```

`out_of_time_split` returns one `(train_idx, validation_idx)` pair for a future
holdout window. Use `validation_size=...` for an exact tail count or
`cutoff=...` for rows strictly after a time boundary. Use `gap=...` to remove
recent training rows immediately before the validation window.

Use these helpers to evaluate temporal-spatial generalization: future periods,
withheld locations, or held-out route groups often reveal failure modes that a
random split hides.

## I/O Helpers

```python
cartoboost.io.read_geojson(path)
```

Reads a GeoJSON file into a Python dictionary.
