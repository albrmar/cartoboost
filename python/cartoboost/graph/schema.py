"""Graph schema helpers for lightweight graph feature pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .._native import graph_validate_directed_metapath as _native_validate_directed_metapath


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


@dataclass(frozen=True)
class EdgeType:
    """Typed relation triple used for heterogeneous graph schemas."""

    source: str
    relation: str
    target: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.source, self.relation, self.target)


@dataclass(frozen=True)
class GraphSchema:
    """Minimal heterogeneous graph schema descriptor."""

    node_types: list[str]
    edge_types: list[EdgeType]
    directed: bool = True
    timestamp_col: str | None = None
    directionality: DirectionalityConfig | None = None

    def validate(self) -> GraphSchema:
        if not self.node_types:
            raise ValueError("node_types must be non-empty")
        if not self.edge_types:
            raise ValueError("edge_types must be non-empty")
        if self.directionality is not None:
            self.directionality.validate(directed=self.directed)

        known_nodes = set(self.node_types)
        for edge in self.edge_types:
            if edge.source not in known_nodes:
                raise ValueError(f"unknown source node type: {edge.source}")
            if edge.target not in known_nodes:
                raise ValueError(f"unknown target node type: {edge.target}")
            if not edge.relation:
                raise ValueError("edge relation name must be non-empty")
        return self


@dataclass(frozen=True)
class DirectionalityConfig:
    """First-class source-target semantics for geotemporal graph features."""

    materialize_reverse_edges: bool = False
    preserve_source_target_roles: bool = True
    create_od_pair_nodes: bool = False
    compute_asymmetry_features: bool = False
    reverse_relation_suffix: str = "_reverse"
    reverse_relation_map: Mapping[Any, Any] | None = None
    directional_feature_prefix: str = "graph"
    directional_features: tuple[str, ...] = field(default_factory=tuple)

    def validate(self, *, directed: bool = True) -> DirectionalityConfig:
        if self.preserve_source_target_roles and not directed:
            raise ValueError(
                "preserve_source_target_roles requires a directed graph schema",
            )
        if not self.reverse_relation_suffix:
            raise ValueError("reverse_relation_suffix must be non-empty")
        if not self.directional_feature_prefix:
            raise ValueError("directional_feature_prefix must be non-empty")
        return self

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "materialize_reverse_edges": self.materialize_reverse_edges,
            "preserve_source_target_roles": self.preserve_source_target_roles,
            "create_od_pair_nodes": self.create_od_pair_nodes,
            "compute_asymmetry_features": self.compute_asymmetry_features,
            "reverse_relation_suffix": self.reverse_relation_suffix,
            "directional_feature_prefix": self.directional_feature_prefix,
        }
        if self.reverse_relation_map is not None:
            payload["reverse_relation_map"] = dict(self.reverse_relation_map)
        if self.directional_features:
            payload["directional_features"] = [
                str(_enum_value(feature)) for feature in self.directional_features
            ]
        return payload

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | DirectionalityConfig | None,
    ) -> DirectionalityConfig:
        if isinstance(config, DirectionalityConfig):
            return config
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise TypeError("directionality config must be a mapping")
        features = config.get("directional_features", ())
        if features is None:
            features_tuple: tuple[str, ...] = ()
        elif isinstance(features, str):
            features_tuple = (features,)
        else:
            features_tuple = tuple(str(_enum_value(feature)) for feature in features)
        reverse_relation_map = config.get("reverse_relation_map")
        if reverse_relation_map is not None and not isinstance(
            reverse_relation_map,
            Mapping,
        ):
            raise TypeError("reverse_relation_map must be a mapping when provided")
        return cls(
            materialize_reverse_edges=bool(
                config.get("materialize_reverse_edges", False),
            ),
            preserve_source_target_roles=bool(
                config.get("preserve_source_target_roles", True),
            ),
            create_od_pair_nodes=bool(config.get("create_od_pair_nodes", False)),
            compute_asymmetry_features=bool(
                config.get("compute_asymmetry_features", False),
            ),
            reverse_relation_suffix=str(
                config.get("reverse_relation_suffix", "_reverse"),
            ),
            reverse_relation_map=reverse_relation_map,
            directional_feature_prefix=str(
                config.get("directional_feature_prefix", "graph"),
            ),
            directional_features=features_tuple,
        )


@dataclass(frozen=True)
class DirectedMetaPath:
    """Node/relation/node metapath that preserves directed source-target roles."""

    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.steps) < 3:
            raise ValueError("directed metapath must contain node, relation, node")
        if len(self.steps) % 2 == 0:
            raise ValueError("directed metapath must alternate node/relation/node")
        if any(not step for step in self.steps):
            raise ValueError("directed metapath steps must be non-empty")

    @property
    def node_types(self) -> tuple[str, ...]:
        return self.steps[0::2]

    @property
    def relations(self) -> tuple[str, ...]:
        return self.steps[1::2]

    def validate(self, schema: GraphSchema) -> DirectedMetaPath:
        schema.validate()
        try:
            _native_validate_directed_metapath(
                list(self.steps),
                [edge.as_tuple() for edge in schema.edge_types],
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self


@dataclass(frozen=True)
class TemporalEdge:
    """Temporal typed edge payload."""

    source: Any
    target: Any
    relation: str | int
    timestamp: float
    weight: float = 1.0
    sign: int | None = None


def as_edge_types(values: Sequence[tuple[str, str, str]]) -> list[EdgeType]:
    """Build validated ``EdgeType`` objects from raw tuples."""
    edge_types: list[EdgeType] = []
    for value in values:
        if len(value) != 3:
            raise ValueError(f"edge type must have 3 values, got {value!r}")
        source, relation, target = value
        if not relation:
            raise ValueError("edge relation name must be non-empty")
        edge_types.append(EdgeType(source=source, relation=relation, target=target))
    return edge_types
