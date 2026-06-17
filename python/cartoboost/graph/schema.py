"""Graph schema helpers for lightweight graph feature pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


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

    def validate(self) -> GraphSchema:
        if not self.node_types:
            raise ValueError("node_types must be non-empty")
        if not self.edge_types:
            raise ValueError("edge_types must be non-empty")

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
