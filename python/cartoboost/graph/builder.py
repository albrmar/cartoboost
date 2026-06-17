"""Utilities for normalizing edge streams for local graph encoders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _ensure_unique(values: Iterable[Any], label: str) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        try:
            if value in seen:
                raise ValueError(f"duplicate {label}: {value!r}")
            seen.add(value)
        except TypeError as exc:
            raise ValueError(
                f"values for {label} must be hashable, got {value!r}",
            ) from exc
        ordered.append(value)
    return ordered


@dataclass(frozen=True)
class HomogeneousGraph:
    """Normalized homogeneous edge list."""

    node_ids: list[Any]
    node_to_index: dict[Any, int]
    node_count: int
    edges: list[tuple[int, int]]
    directed: bool = True


@dataclass(frozen=True)
class HeterogeneousGraph:
    """Normalized typed-edge list."""

    node_ids: list[Any]
    node_to_index: dict[Any, int]
    relation_ids: list[Any]
    relation_to_index: dict[Any, int]
    node_count: int
    relation_count: int
    edges: list[tuple[int, int, int]]
    directed: bool = True


def normalize_homogeneous_graph(
    edges: Sequence[tuple[Any, Any]],
    *,
    node_ids: Sequence[Any] | None = None,
    node_count: int | None = None,
    directed: bool = True,
) -> HomogeneousGraph:
    """Normalize a homogeneous edge list into integer node indices."""
    if node_ids is not None:
        normal_nodes = _ensure_unique(node_ids, "node id")
        node_to_index = {node: index for index, node in enumerate(normal_nodes)}
        resolved_node_count = len(node_ids)
    elif node_count is not None:
        if node_count <= 0:
            raise ValueError("node_count must be positive when provided")
        normal_nodes = list(range(node_count))
        node_to_index = {node: node for node in normal_nodes}
        resolved_node_count = node_count
    else:
        detected = []
        node_to_index: dict[Any, int] = {}
        for source, target in edges:
            for node in (source, target):
                if node not in node_to_index:
                    node_to_index[node] = len(detected)
                    detected.append(node)
        if not detected:
            raise ValueError("node_count must be provided for empty edge sets")
        normal_nodes = detected
        resolved_node_count = len(detected)

    normalized_edges: list[tuple[int, int]] = []
    for source, target in edges:
        if source not in node_to_index or target not in node_to_index:
            raise ValueError("edge endpoint not represented by node_ids/node_count")
        src = node_to_index[source]
        dst = node_to_index[target]
        normalized_edges.append((src, dst))
        if not directed and src != dst:
            normalized_edges.append((dst, src))

    if normalized_edges == []:
        # Keep an explicit empty graph with no edges for edge-case stability.
        normalized_edges = []

    return HomogeneousGraph(
        node_ids=normal_nodes,
        node_to_index=node_to_index,
        node_count=resolved_node_count,
        edges=normalized_edges,
        directed=directed,
    )


def normalize_heterogeneous_graph(
    edges: Sequence[tuple[Any, Any, Any]],
    *,
    node_ids: Sequence[Any] | None = None,
    relation_ids: Sequence[Any] | None = None,
    node_count: int | None = None,
    directed: bool = True,
    materialize_reverse_edges: bool = False,
    reverse_relation_suffix: str = "_reverse",
    reverse_relation_map: Mapping[Any, Any] | None = None,
) -> HeterogeneousGraph:
    """Normalize a typed edge list into integer node/relation indices."""
    if node_ids is not None:
        normal_nodes = _ensure_unique(node_ids, "node id")
        node_to_index = {node: index for index, node in enumerate(normal_nodes)}
        resolved_node_count = len(node_ids)
    elif node_count is not None:
        if node_count <= 0:
            raise ValueError("node_count must be positive when provided")
        normal_nodes = list(range(node_count))
        node_to_index = {node: node for node in normal_nodes}
        resolved_node_count = node_count
    else:
        detected = []
        node_to_index = {}
        for source, target, _relation in edges:
            for node in (source, target):
                if node not in node_to_index:
                    node_to_index[node] = len(detected)
                    detected.append(node)
        if not detected:
            raise ValueError("node_count must be provided for empty edge sets")
        normal_nodes = detected
        resolved_node_count = len(detected)

    if relation_ids is not None:
        normal_relations = _ensure_unique(relation_ids, "relation id")
        relation_to_index = {rel: index for index, rel in enumerate(normal_relations)}
    else:
        normal_relations = []
        relation_to_index: dict[Any, int] = {}
        for _source, _target, relation in edges:
            if relation not in relation_to_index:
                relation_to_index[relation] = len(normal_relations)
                normal_relations.append(relation)
    resolved_relation_count = len(normal_relations)

    if resolved_relation_count == 0:
        raise ValueError("relation set is empty")

    normalized_edges: list[tuple[int, int, int]] = []
    relation_lookup = reverse_relation_map or {}
    for source, target, relation in edges:
        if source not in node_to_index or target not in node_to_index:
            raise ValueError("edge endpoint not represented by node_ids/node_count")
        if relation not in relation_to_index:
            if relation_ids is not None:
                raise ValueError(f"relation {relation!r} not in relation_ids")
            relation_to_index[relation] = len(normal_relations)
            normal_relations.append(relation)
            resolved_relation_count = len(normal_relations)

        source_idx = node_to_index[source]
        target_idx = node_to_index[target]
        relation_idx = relation_to_index[relation]
        normalized_edges.append((source_idx, target_idx, relation_idx))
        if not directed:
            normalized_edges.append((target_idx, source_idx, relation_idx))
        elif materialize_reverse_edges:
            if relation in relation_lookup:
                reverse_relation = relation_lookup[relation]
            elif isinstance(relation, str):
                reverse_relation = f"{relation}{reverse_relation_suffix}"
            else:
                raise ValueError(
                    f"cannot materialize reverse edge for non-string relation {relation!r}",
                )

            if reverse_relation not in relation_to_index:
                relation_to_index[reverse_relation] = len(normal_relations)
                normal_relations.append(reverse_relation)
                resolved_relation_count = len(normal_relations)
            reverse_relation_idx = relation_to_index[reverse_relation]
            normalized_edges.append((target_idx, source_idx, reverse_relation_idx))

    return HeterogeneousGraph(
        node_ids=normal_nodes,
        node_to_index=node_to_index,
        relation_ids=normal_relations,
        relation_to_index=relation_to_index,
        node_count=resolved_node_count,
        relation_count=resolved_relation_count,
        edges=normalized_edges,
        directed=directed,
    )


def ensure_node_features_shape(
    features: Iterable[Sequence[float]],
    node_count: int,
) -> list[list[float]]:
    """Validate and materialize a dense feature table."""
    dense = [list(row) for row in features]
    if len(dense) != node_count:
        raise ValueError(
            f"expected {node_count} feature rows, got {len(dense)}",
        )
    return dense
