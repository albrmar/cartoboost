from __future__ import annotations

from cartoboost.graph import (
    GraphFeatureTransformer,
    MetaPathWalkGenerator,
    SignedEdgeSampler,
    TemporalWalkGenerator,
    binary_auc,
    binary_average_precision,
    mean_reciprocal_rank,
    top_k_metrics,
    normalize_heterogeneous_graph,
)


def test_meta_path_generator_respects_relation_constraints() -> None:
    edges = [
        (0, 1, "rep-lane"),
        (1, 2, "lane-carrier"),
        (2, 3, "carrier-rep"),
    ]
    generator = MetaPathWalkGenerator(
        metapath=("rep-lane", "lane-carrier", "carrier-rep"),
        walk_length=4,
        walks_per_node=2,
        seed=7,
    )
    walks = generator.generate(edges=edges, start_nodes=[0, 1])
    assert len(walks) == 4
    for walk in walks:
        assert walk[0] in {0, 1}


def test_temporal_walk_generator_enforces_time_monotonicity() -> None:
    edges = [
        (0, 1, 1.0),
        (0, 2, 2.0),
        (1, 3, 3.0),
        (2, 4, 4.0),
    ]
    generator = TemporalWalkGenerator(walk_length=4, walks_per_node=1, seed=13)
    walks = generator.generate(temporal_edges=edges, start_nodes=[0])
    assert walks
    for walk in walks:
        assert walk[0] == 0
        assert walk[-1] in {1, 2, 3, 4}


def test_signed_sampler_avoids_false_negatives() -> None:
    edges = [(0, 1), (0, 2)]
    sampler = SignedEdgeSampler(negative_samples=2, seed=3)
    sampled = sampler.sample(node_count=4, edges=edges)
    sampled_edge_set = {edge for values in sampled.values() for edge in values}
    assert not sampled_edge_set.intersection(set(edges))


def test_graph_metrics_match_simple_reference() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.2]
    assert binary_auc(labels, scores) > 0.8
    assert binary_average_precision(labels, scores) > 0.9

    top_k = top_k_metrics(labels, scores, query_ids=["a", "a", "b", "b"], k=1)
    assert top_k["recall_at_k"] == 1.0
    assert top_k["hit_rate_at_k"] == 1.0

    mrr = mean_reciprocal_rank(labels, scores, query_ids=["a", "a", "b", "b"])
    assert mrr > 0.8


def test_graph_feature_transformer_homogeneous_fit_transform_smoke() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [(0, 1), (1, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {"graph_sage": {"hidden_dims": [2], "epochs": 2, "input_dim": 2}},
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
    )
    assert bundle.embeddings.shape == (3, 2)


def test_graph_feature_transformer_rejects_mismatched_input_dim() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [(0, 1), (1, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {"graph_sage": {"input_dim": 3, "hidden_dims": [2], "epochs": 2}},
    )
    try:
        transformer.fit_transform(node_features=features, edges=edges, node_count=3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected input_dim mismatch to raise")


def test_graph_transformer_accepts_graph_embeddings_config_namespace() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [(0, 1), (1, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "hinsage",
                    "hetero": False,
                    "input_dim": 2,
                    "hidden_dims": [2],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
    )
    assert bundle.embeddings.shape == (3, 2)


def test_normalize_heterogeneous_graph_supports_reverse_relation_materialization() -> None:
    graph = normalize_heterogeneous_graph(
        edges=[(0, 1, "origin_to_dest"), (1, 2, "origin_to_dest")],
        directed=True,
        materialize_reverse_edges=True,
        reverse_relation_suffix="_reverse",
    )
    assert len(graph.edges) == 4
    assert "origin_to_dest" in graph.relation_ids
    assert "origin_to_dest_reverse" in graph.relation_ids

    forward_relation = graph.relation_to_index["origin_to_dest"]
    reverse_relation = graph.relation_to_index["origin_to_dest_reverse"]
    assert (0, 1, forward_relation) in graph.edges
    assert (1, 0, reverse_relation) in graph.edges
    assert (2, 1, reverse_relation) in graph.edges
