"""Configuration helpers for graph feature pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schema import (
    DirectedMetaPath,
    DirectionalityConfig,
    GraphSchema,
    as_edge_types,
)


@dataclass(frozen=True)
class GraphFeatureConfig:
    """Validated graph feature configuration.

    This mirrors the optional YAML-style ``graph_embeddings`` block while
    preserving CartoBoost's current dense-plus-sparse booster contract.
    """

    enabled: bool = True
    backend: str = "native"
    task_mode: str = "precompute_features"
    graph_schema: GraphSchema | None = None
    encoder: dict[str, Any] = field(default_factory=dict)
    metapaths: tuple[DirectedMetaPath, ...] = field(default_factory=tuple)
    outputs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GraphFeatureConfig:
        if not isinstance(config, Mapping):
            raise TypeError("graph feature config must be a mapping")

        root = config.get("graph_embeddings", config)
        if not isinstance(root, Mapping):
            raise TypeError("graph_embeddings config must be a mapping")

        graph_block = root.get("graph", {})
        graph_schema = None
        if graph_block:
            if not isinstance(graph_block, Mapping):
                raise TypeError("graph block must be a mapping")
            graph_schema = _graph_schema_from_mapping(
                graph_block,
                root_directionality=root.get("directionality", {}),
            )

        encoder = root.get("encoder", root.get("walk_embeddings", {}))
        if encoder is None:
            encoder = {}
        if not isinstance(encoder, Mapping):
            raise TypeError("encoder/walk_embeddings block must be a mapping")

        metapaths = tuple(_coerce_metapath(value) for value in _metapath_values(root, encoder))
        if graph_schema is not None:
            for metapath in metapaths:
                metapath.validate(graph_schema)

        outputs = root.get("outputs", {})
        if outputs is None:
            outputs = {}
        if not isinstance(outputs, Mapping):
            raise TypeError("outputs block must be a mapping")

        return cls(
            enabled=bool(root.get("enabled", True)),
            backend=str(root.get("backend", "native")),
            task_mode=str(root.get("task_mode", "precompute_features")),
            graph_schema=graph_schema,
            encoder=dict(encoder),
            metapaths=metapaths,
            outputs=dict(outputs),
        )

    @property
    def directionality(self) -> DirectionalityConfig:
        if self.graph_schema is None or self.graph_schema.directionality is None:
            return DirectionalityConfig()
        return self.graph_schema.directionality

    def transformer_config(self) -> dict[str, Any]:
        """Return the subset accepted by ``GraphFeatureTransformer.from_config``."""
        encoder = dict(self.encoder)
        if "family" not in encoder:
            encoder["family"] = "graphsage"
        return {
            "graph_embeddings": {
                "enabled": self.enabled,
                "backend": self.backend,
                "task_mode": self.task_mode,
                "encoder": encoder,
                "directionality": self.directionality.as_dict(),
                "outputs": dict(self.outputs),
            },
        }


def _graph_schema_from_mapping(
    graph_block: Mapping[str, Any],
    *,
    root_directionality: Any = None,
) -> GraphSchema:
    node_types = graph_block.get("node_types", [])
    if isinstance(node_types, str) or not isinstance(node_types, Sequence):
        raise TypeError("graph.node_types must be a sequence")

    edge_values = graph_block.get("edge_types", [])
    if isinstance(edge_values, str) or not isinstance(edge_values, Sequence):
        raise TypeError("graph.edge_types must be a sequence")

    directionality_block = graph_block.get("directionality", root_directionality or {})
    directionality = DirectionalityConfig.from_config(directionality_block)
    return GraphSchema(
        node_types=[str(value) for value in node_types],
        edge_types=as_edge_types([tuple(value) for value in edge_values]),
        directed=bool(graph_block.get("directed", True)),
        timestamp_col=(
            str(graph_block["timestamp_col"]) if graph_block.get("timestamp_col") else None
        ),
        directionality=directionality,
    ).validate()


def _metapath_values(
    root: Mapping[str, Any],
    encoder: Mapping[str, Any],
) -> Sequence[Any]:
    walk_embeddings = root.get("walk_embeddings", {})
    values: Any = ()
    if isinstance(walk_embeddings, Mapping):
        values = walk_embeddings.get("metapaths", walk_embeddings.get("meta_paths", ()))
    if not values:
        values = root.get("metapaths", root.get("meta_paths", ()))
    if not values:
        values = encoder.get("metapaths", encoder.get("meta_paths", ()))
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError("metapaths must be a sequence")
    return values


def _coerce_metapath(value: Any) -> DirectedMetaPath:
    if isinstance(value, DirectedMetaPath):
        return value
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("metapath must be a node/relation/node sequence")
    return DirectedMetaPath(tuple(str(step) for step in value))
