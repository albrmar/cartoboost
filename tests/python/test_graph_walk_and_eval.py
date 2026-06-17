from __future__ import annotations

from cartoboost.graph import (
    DirectedMetaPath,
    DirectionalityConfig,
    EdgeType,
    GraphFeatureConfig,
    GraphFeatureTransformer,
    GraphSchema,
    HinSageFeatureEncoder,
    MetaPathWalkGenerator,
    SignedEdgeSampler,
    TemporalWalkGenerator,
    binary_auc,
    binary_average_precision,
    link_prediction_report,
    materialize_source_target_pair_nodes,
    mean_reciprocal_rank,
    normalize_heterogeneous_graph,
    top_k_metrics,
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


def test_directed_metapath_validates_schema_and_drives_relation_walk() -> None:
    schema = GraphSchema(
        node_types=["source_h3", "target_h3", "carrier"],
        edge_types=[
            EdgeType("carrier", "serves_outbound_from", "source_h3"),
            EdgeType("source_h3", "flows_to", "target_h3"),
        ],
        directionality=DirectionalityConfig(preserve_source_target_roles=True),
    ).validate()
    path = DirectedMetaPath(
        (
            "carrier",
            "serves_outbound_from",
            "source_h3",
            "flows_to",
            "target_h3",
        ),
    ).validate(schema)

    generator = MetaPathWalkGenerator(metapath=path, walk_length=3, walks_per_node=1)
    walks = generator.generate(
        start_nodes=[0],
        edges=[(0, 1, "serves_outbound_from"), (1, 2, "flows_to")],
    )
    assert walks == [[0, 1, 2]]


def test_directionality_config_rejects_undirected_source_target_roles() -> None:
    schema = GraphSchema(
        node_types=["market"],
        edge_types=[EdgeType("market", "connected_to", "market")],
        directed=False,
        directionality=DirectionalityConfig(preserve_source_target_roles=True),
    )
    try:
        schema.validate()
    except ValueError as error:
        assert "requires a directed graph schema" in str(error)
    else:
        raise AssertionError("expected source-target role validation to fail")


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


def test_signed_sampler_returns_fewer_negatives_when_source_is_dense() -> None:
    edges = [(0, 0), (0, 1), (0, 2)]
    sampler = SignedEdgeSampler(negative_samples=2, seed=3)

    sampled = sampler.sample(node_count=3, edges=edges)

    assert sampled == {edge: [] for edge in edges}


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

    report = link_prediction_report(labels, scores, query_ids=["a", "a", "b", "b"], k=1)
    assert report["auc"] > 0.8
    assert report["average_precision"] > 0.9
    assert report["mean_reciprocal_rank"] > 0.8


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


def test_graph_feature_transformer_directional_features_disabled_by_default() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [(0, 1), (1, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
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
    assert len(bundle.feature_names) == 2
    assert "dir_source_target_affinity" not in bundle.feature_names


def test_graph_feature_transformer_directional_features_enabled() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7], [0.4, 0.6]]
    edges = [(0, 1), (1, 2), (2, 1)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {
                    "compute_asymmetry_features": True,
                    "directional_feature_prefix": "dir",
                    "directional_features": [
                        "dir_source_target_affinity",
                        "dir_target_source_affinity",
                        "dir_flow_imbalance_ratio",
                    ],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=4,
        directed=True,
    )
    assert bundle.embeddings.shape == (4, 5)
    assert len(bundle.feature_names) == 5
    assert bundle.feature_names == [
        "graph_sage_homo_00",
        "graph_sage_homo_01",
        "dir_source_target_affinity",
        "dir_target_source_affinity",
        "dir_flow_imbalance_ratio",
    ]


def test_graph_feature_transformer_directional_features_use_weights_and_time() -> None:
    features = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    edges = [(0, 1), (1, 0), (0, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {
                    "compute_asymmetry_features": True,
                    "directional_features": [
                        "graph_forward_flow_weight",
                        "graph_reverse_flow_weight",
                        "graph_directed_temporal_drift",
                    ],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
        edge_weights=[2.0, 5.0, 3.0],
        edge_timestamps=[10.0, 20.0, 40.0],
    )
    forward_idx = bundle.feature_names.index("graph_forward_flow_weight")
    reverse_idx = bundle.feature_names.index("graph_reverse_flow_weight")
    drift_idx = bundle.feature_names.index("graph_directed_temporal_drift")

    assert bundle.embeddings[0, forward_idx] == 5.0
    assert bundle.embeddings[0, reverse_idx] == 5.0
    assert bundle.embeddings[0, drift_idx] == 8.0


def test_graph_feature_transformer_directional_features_reject_invalid_names() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [(0, 1), (1, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_sage": {
                "input_dim": 2,
                "hidden_dims": [2],
                "epochs": 2,
                "directionality": {
                    "compute_asymmetry_features": True,
                    "directional_features": ["unknown_feature"],
                },
            },
        },
    )
    try:
        transformer.fit_transform(node_features=features, edges=edges, node_count=3)
    except ValueError as error:
        assert "no recognized feature names" in str(error)
    else:
        raise AssertionError("expected invalid directional feature selection to raise")


def test_graph_feature_transformer_outputs_directional_features_contract() -> None:
    features = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    edges = [(0, 1), (1, 0), (0, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {"directional_feature_prefix": "graph"},
                "outputs": {
                    "directional_features": [
                        "source_target_embedding",
                        "target_source_embedding",
                        "forward_reverse_similarity_delta",
                        "source_outbound_strength",
                        "target_inbound_strength",
                        "flow_imbalance_ratio",
                        "directed_temporal_drift",
                    ],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
        edge_weights=[2.0, 5.0, 3.0],
        edge_timestamps=[10.0, 20.0, 40.0],
    )
    assert bundle.feature_names[-7:] == [
        "graph_source_target_embedding",
        "graph_target_source_embedding",
        "graph_forward_reverse_similarity_delta",
        "graph_source_outbound_strength",
        "graph_target_inbound_strength",
        "graph_flow_imbalance_ratio",
        "graph_directed_temporal_drift",
    ]


def test_graph_feature_transformer_emits_freight_directional_aliases() -> None:
    features = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    edges = [(0, 1), (1, 0), (0, 2)]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "graphsage",
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {
                    "compute_asymmetry_features": True,
                    "directional_features": [
                        "graph_od_forward_similarity",
                        "graph_od_reverse_similarity",
                        "graph_origin_outbound_strength",
                        "graph_destination_inbound_strength",
                        "graph_forward_flow_volume_30d",
                        "graph_reverse_flow_volume_30d",
                        "graph_directional_market_drift",
                        "graph_directional_acceptance_rate",
                        "graph_directional_price_pressure",
                    ],
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
        edge_weights=[2.0, 5.0, 3.0],
        edge_timestamps=[10.0, 20.0, 40.0],
    )
    assert bundle.feature_names[-9:] == [
        "graph_od_forward_similarity",
        "graph_od_reverse_similarity",
        "graph_origin_outbound_strength",
        "graph_destination_inbound_strength",
        "graph_forward_flow_volume_30d",
        "graph_reverse_flow_volume_30d",
        "graph_directional_market_drift",
        "graph_directional_acceptance_rate",
        "graph_directional_price_pressure",
    ]


def test_graph_feature_transformer_hetero_directional_features_enabled() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    edges = [
        (0, 1, "rep_lane"),
        (1, 2, "lane_carrier"),
    ]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "hinsage",
                    "hetero": True,
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {"compute_asymmetry_features": True},
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_count=3,
        directed=True,
    )
    assert bundle.embeddings.shape[0] == 3
    assert bundle.embeddings.shape[1] == len(bundle.feature_names)
    assert bundle.feature_names[0] == "graph_sage_hetero_00"
    assert "graph_source_target_embedding" in bundle.feature_names
    assert "graph_target_source_embedding" in bundle.feature_names
    assert "graph_forward_reverse_similarity_delta" in bundle.feature_names
    assert sum(name.startswith("graph_") for name in bundle.feature_names) >= 14


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


def test_source_target_pair_nodes_keep_forward_and_reverse_distinct() -> None:
    materialized = materialize_source_target_pair_nodes(
        [
            ("Chicago", "Dallas", "flows_to"),
            ("Dallas", "Chicago", "flows_to"),
        ],
    )
    assert ("od_pair", "Chicago", "Dallas") in materialized.pair_node_ids
    assert ("od_pair", "Dallas", "Chicago") in materialized.pair_node_ids
    assert (
        "Chicago",
        ("od_pair", "Chicago", "Dallas"),
        "source_to_pair",
    ) in materialized.edges
    assert (
        ("od_pair", "Dallas", "Chicago"),
        "Chicago",
        "pair_to_target",
    ) in materialized.edges


def test_graph_feature_transformer_can_create_od_pair_nodes_with_zero_features() -> None:
    features = [[0.1, 0.2], [0.0, 0.3], [0.2, 0.7]]
    node_ids = ["carrier_a", "origin", "destination"]
    edges = [
        ("carrier_a", "origin", "serves_outbound_from"),
        ("origin", "destination", "flows_to"),
    ]
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "hinsage",
                    "hetero": True,
                    "input_dim": 2,
                    "hidden_dims": [2],
                    "epochs": 2,
                },
                "directionality": {
                    "create_od_pair_nodes": True,
                    "compute_asymmetry_features": True,
                },
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=features,
        edges=edges,
        node_ids=node_ids,
        directed=True,
    )
    assert ("od_pair", "origin", "destination") in bundle.node_ids
    assert bundle.embeddings.shape[0] == 5
    assert "graph_flow_imbalance_ratio" in bundle.feature_names


def test_graph_feature_config_parses_directional_yaml_shape() -> None:
    config = GraphFeatureConfig.from_config(
        {
            "graph_embeddings": {
                "enabled": True,
                "backend": "native",
                "task_mode": "precompute_features",
                "graph": {
                    "directed": True,
                    "node_types": [
                        "source_h3",
                        "target_h3",
                        "od_pair",
                        "entity",
                        "time_bucket",
                    ],
                    "edge_types": [
                        ["source_h3", "flows_to", "target_h3"],
                        ["target_h3", "reverse_flows_to", "source_h3"],
                        ["entity", "active_from", "source_h3"],
                        ["entity", "active_to", "target_h3"],
                        ["entity", "observed_on", "od_pair"],
                    ],
                    "directionality": {
                        "materialize_reverse_edges": True,
                        "preserve_source_target_roles": True,
                        "create_od_pair_nodes": True,
                        "compute_asymmetry_features": True,
                    },
                },
                "walk_embeddings": {
                    "algorithm": "metapath2vec",
                    "metapaths": [
                        [
                            "source_h3",
                            "flows_to",
                            "target_h3",
                            "reverse_flows_to",
                            "source_h3",
                        ],
                    ],
                },
                "outputs": {
                    "directional_features": [
                        "source_target_embedding",
                        "target_source_embedding",
                    ],
                },
            },
        },
    )

    assert config.graph_schema is not None
    assert config.graph_schema.directed is True
    assert config.directionality.materialize_reverse_edges is True
    assert config.directionality.create_od_pair_nodes is True
    assert config.metapaths[0].relations == ("flows_to", "reverse_flows_to")
    assert config.transformer_config()["graph_embeddings"]["directionality"][
        "compute_asymmetry_features"
    ]


def test_graph_feature_bundle_exposes_schema_and_training_metadata() -> None:
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_sage": {
                "input_dim": 2,
                "hidden_dims": [2],
                "epochs": 2,
            },
        },
    )
    bundle = transformer.fit_transform(
        node_features=[[0.1, 0.2], [0.2, 0.3]],
        edges=[(0, 1)],
        node_count=2,
    )

    assert bundle.feature_schema_payload()["names"] == [
        "graph_sage_homo_00",
        "graph_sage_homo_01",
    ]
    assert bundle.feature_schema_payload()["kinds"] == ["Numeric", "Numeric"]
    assert bundle.training_config_metadata()["provenance"]["encoder"] == "graphsage"


def test_native_hinsage_feature_encoder_requires_typed_schema_and_builds_link_features() -> None:
    encoder = HinSageFeatureEncoder.from_config(
        {
            "input_dim": 2,
            "node_type_count": 3,
            "edge_type_triples": [(0, 0, 2), (1, 1, 2), (2, 2, 0)],
            "hidden_dims": [3],
            "epochs": 3,
            "negative_samples": 1,
            "neighbor_samples": [2, 0, 0],
        }
    )
    node_types = [0, 1, 2, 2, 2]
    edges = [
        (0, 2, 0),
        (0, 3, 0),
        (0, 4, 0),
        (1, 2, 1),
        (2, 0, 2),
    ]
    features = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.2],
        [0.7, 0.4],
        [0.9, 0.6],
    ]

    bundle = encoder.fit(features, edges, node_types)
    link_bundle = encoder.link_embeddings(bundle.embeddings, [(0, 2), (1, 2)])

    assert bundle.embeddings.shape == (5, 3)
    assert bundle.provenance["encoder"] == "hinsage"
    assert bundle.provenance["neighbor_samples"] == [2, 0, 0]
    assert link_bundle.embeddings.shape == (2, 12)
    assert link_bundle.feature_names[0] == "hinsage_link_00"
    assert len(encoder.loss_curve()) == 3


def test_graph_feature_transformer_uses_true_hinsage_when_schema_is_provided() -> None:
    transformer = GraphFeatureTransformer.from_config(
        {
            "graph_embeddings": {
                "encoder": {
                    "family": "hinsage",
                    "input_dim": 2,
                    "node_type_count": 2,
                    "edge_type_triples": [(0, 0, 1), (1, 1, 0)],
                    "hidden_dims": [2],
                    "epochs": 2,
                    "neighbor_samples": [1, 0],
                }
            }
        }
    )
    bundle = transformer.fit_transform(
        node_features=[[1.0, 0.0], [0.0, 1.0], [0.4, 0.2]],
        edges=[(0, 1, 0), (0, 2, 0), (1, 0, 1)],
        node_types=[0, 1, 1],
    )

    assert bundle.embeddings.shape == (3, 2)
    assert bundle.provenance["encoder"] == "hinsage"
