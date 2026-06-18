from __future__ import annotations

import tempfile

import cartoboost.graph as cb_graph
import cartoboost.neural as cb_neural
import numpy as np
from cartoboost import (
    GraphSageLinkPredictor,
    GraphSageStandaloneRegressor,
    HeteroGraphSageLinkPredictor,
    HeteroGraphSageStandaloneRegressor,
    HinSageLinkPredictor,
    HinSageStandaloneRegressor,
    NeuralEmbeddingStandaloneRegressor,
    Node2VecEncoder,
    Node2VecLinkPredictor,
    Node2VecStandaloneRegressor,
)


def test_graph_and_neural_exposure_surfaces() -> None:
    assert cb_graph.Node2VecStandaloneRegressor is Node2VecStandaloneRegressor
    assert cb_graph.GraphSageStandaloneRegressor is GraphSageStandaloneRegressor
    assert cb_graph.HeteroGraphSageStandaloneRegressor is HeteroGraphSageStandaloneRegressor
    assert cb_graph.HinSageStandaloneRegressor is HinSageStandaloneRegressor
    assert cb_graph.Node2VecLinkPredictor is Node2VecLinkPredictor
    assert cb_graph.GraphSageLinkPredictor is GraphSageLinkPredictor
    assert cb_graph.HeteroGraphSageLinkPredictor is HeteroGraphSageLinkPredictor
    assert cb_graph.HinSageLinkPredictor is HinSageLinkPredictor
    assert cb_neural.NeuralEmbeddingStandaloneRegressor is NeuralEmbeddingStandaloneRegressor
    assert Node2VecEncoder(dim=2).output_dim == 2


def test_neural_embedding_standalone_regressor_predicts_and_reloads() -> None:
    ids = np.array([1, 2, 1, 3, 2, 3], dtype=np.uint64)
    dense = np.column_stack([ids.astype(float) / 3.0])
    y = np.array([1.0, 2.0, 1.1, 3.0, 2.1, 3.2])
    model = NeuralEmbeddingStandaloneRegressor(
        dim=3,
        n_estimators=12,
        max_depth=2,
        min_samples_leaf=1,
        random_state=7,
    ).fit(ids, y, dense=dense)

    pred = model.predict(ids, dense=dense)
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/neural.json"
        model.save(path)
        loaded = NeuralEmbeddingStandaloneRegressor.load(path)
        np.testing.assert_allclose(loaded.predict(ids, dense=dense), pred)


def test_node2vec_standalone_pair_regressor_predicts() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    row_nodes = np.array([0, 1, 2, 3])
    row_targets = np.array([1, 2, 3, 0])
    dense = np.column_stack([row_nodes.astype(float)])
    y = np.array([1.0, 1.5, 2.0, 0.7])

    model = Node2VecStandaloneRegressor(
        dim=4,
        walk_length=5,
        walks_per_node=2,
        window_size=2,
        epochs=1,
        negative_samples=1,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
        seed=11,
    ).fit(
        node_count=4,
        edges=edges,
        row_nodes=row_nodes,
        row_targets=row_targets,
        dense=dense,
        y=y,
    )

    pred = model.predict(row_nodes, row_targets=row_targets, dense=dense)
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))


def test_graphsage_standalone_pair_regressor_predicts() -> None:
    node_features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.25],
        ],
        dtype=np.float32,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    row_nodes = np.array([0, 1, 2, 3])
    row_targets = np.array([1, 2, 3, 0])
    y = np.array([1.2, 1.7, 2.1, 0.5])

    model = GraphSageStandaloneRegressor(
        input_dim=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
        seed=13,
    ).fit(
        node_features=node_features,
        edges=edges,
        row_nodes=row_nodes,
        row_targets=row_targets,
        y=y,
    )

    pred = model.predict(
        node_features=node_features,
        row_nodes=row_nodes,
        row_targets=row_targets,
    )
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))


def test_hetero_and_hinsage_standalone_regressors_predict_and_reload() -> None:
    node_features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.25],
        ],
        dtype=np.float32,
    )
    typed_edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1)]
    row_nodes = np.array([0, 1, 2, 3])
    row_targets = np.array([1, 2, 3, 0])
    y = np.array([1.2, 1.7, 2.1, 0.5])

    hetero = HeteroGraphSageStandaloneRegressor(
        input_dim=2,
        relation_count=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
        seed=23,
    ).fit(
        node_features=node_features,
        edges=typed_edges,
        row_nodes=row_nodes,
        row_targets=row_targets,
        y=y,
    )
    hetero_pred = hetero.predict(
        node_features=node_features,
        row_nodes=row_nodes,
        row_targets=row_targets,
    )
    assert hetero_pred.shape == y.shape
    assert np.all(np.isfinite(hetero_pred))

    hinsage = HinSageStandaloneRegressor(
        input_dim=2,
        node_type_count=2,
        edge_type_triples=[(0, 0, 1), (1, 1, 0)],
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
        seed=29,
    ).fit(
        node_features=node_features,
        node_types=[0, 1, 0, 1],
        edges=typed_edges,
        row_nodes=row_nodes,
        row_targets=row_targets,
        y=y,
    )
    pred = hinsage.predict(
        node_features=node_features,
        row_nodes=row_nodes,
        row_targets=row_targets,
    )
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/hinsage.json"
        hinsage.save(path)
        loaded = HinSageStandaloneRegressor.load(path)
        np.testing.assert_allclose(
            loaded.predict(
                node_features=node_features,
                row_nodes=row_nodes,
                row_targets=row_targets,
            ),
            pred,
        )


def test_standalone_link_predictors_report_metrics() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    candidates = [(0, 1), (0, 2), (1, 2), (1, 3)]
    labels = [1, 0, 1, 0]

    node2vec = Node2VecLinkPredictor(
        dim=4,
        walk_length=5,
        walks_per_node=2,
        window_size=2,
        epochs=1,
        negative_samples=1,
        seed=17,
    ).fit(node_count=4, edges=edges)
    node2vec_report = node2vec.report(candidates, labels, query_ids=[0, 0, 1, 1], k=1)
    assert {"auc", "average_precision", "mean_reciprocal_rank"}.issubset(node2vec_report)

    node_features = np.eye(4, 2, dtype=np.float32)
    sage = GraphSageLinkPredictor(
        input_dim=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        seed=19,
    ).fit(node_features=node_features, edges=edges)
    sage_report = sage.report(
        node_features=node_features,
        pairs=candidates,
        labels=labels,
        query_ids=[0, 0, 1, 1],
        k=1,
    )
    assert {"auc", "average_precision", "mean_reciprocal_rank"}.issubset(sage_report)

    typed_edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1)]
    hetero = HeteroGraphSageLinkPredictor(
        input_dim=2,
        relation_count=2,
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        seed=31,
    ).fit(node_features=node_features, edges=typed_edges)
    hetero_report = hetero.report(
        node_features=node_features,
        pairs=candidates,
        labels=labels,
        query_ids=[0, 0, 1, 1],
        k=1,
    )
    assert {"auc", "average_precision", "mean_reciprocal_rank"}.issubset(hetero_report)

    hinsage = HinSageLinkPredictor(
        input_dim=2,
        node_type_count=2,
        edge_type_triples=[(0, 0, 1), (1, 1, 0)],
        hidden_dims=(4,),
        epochs=2,
        negative_samples=1,
        seed=37,
    ).fit(
        node_features=node_features,
        node_types=[0, 1, 0, 1],
        edges=typed_edges,
    )
    hinsage_report = hinsage.report(
        node_features=node_features,
        pairs=candidates,
        labels=labels,
        query_ids=[0, 0, 1, 1],
        k=1,
    )
    assert {"auc", "average_precision", "mean_reciprocal_rank"}.issubset(hinsage_report)


def test_hinsage_native_encoder_exposes_graph_aware_encode() -> None:
    from cartoboost import HinSageEncoder

    node_features = np.eye(4, 2, dtype=np.float32)
    node_types = [0, 1, 0, 1]
    edge_type_triples = [(0, 0, 1), (1, 1, 0)]
    edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1)]

    encoder = HinSageEncoder(
        input_dim=2,
        node_type_count=2,
        edge_type_triples=edge_type_triples,
        hidden_dims=[4],
        epochs=2,
        negative_samples=1,
    )
    fitted = np.asarray(encoder.fit(node_types, edges, node_features))
    encoded = np.asarray(encoder.encode_graph(node_types, edges, node_features))
    np.testing.assert_allclose(encoded, fitted)
