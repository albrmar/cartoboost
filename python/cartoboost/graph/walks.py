"""Random walk helpers for lightweight graph feature generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MetaPathWalkGenerator:
    """Generate deterministic metapath-constrained random walks."""

    metapath: tuple[Any, ...]
    walk_length: int = 8
    walks_per_node: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.metapath) < 2:
            raise ValueError("metapath must include at least source and target relation")
        if self.walk_length <= 1:
            raise ValueError("walk_length must be > 1")
        if self.walks_per_node <= 0:
            raise ValueError("walks_per_node must be positive")

    def generate(
        self,
        start_nodes: Sequence[int],
        edges: Sequence[tuple[int, int, Any]] | None = None,
        *,
        typed_edges: Sequence[tuple[int, int, Any]] | None = None,
    ) -> list[list[int]]:
        source_edges = edges if edges is not None else typed_edges
        if source_edges is None:
            raise TypeError("edges is required for meta-path generation")
        adjacency = self._build_adjacency(source_edges)
        if not adjacency:
            return []

        rng = random.Random(self.seed)
        walks: list[list[int]] = []
        metapath = self.metapath
        for start in start_nodes:
            if start not in adjacency:
                continue
            for _ in range(self.walks_per_node):
                walk = [start]
                current = start
                for step in range(self.walk_length - 1):
                    expected_relation = metapath[step % len(metapath)]
                    next_nodes = adjacency.get(current, {}).get(expected_relation, [])
                    if not next_nodes:
                        break
                    current = rng.choice(next_nodes)
                    walk.append(current)
                walks.append(walk)
        return walks

    def _build_adjacency(
        self,
        typed_edges: Sequence[tuple[int, int, Any]],
    ) -> dict[int, dict[Any, list[int]]]:
        adjacency: dict[int, dict[Any, list[int]]] = {}
        for source, target, relation in typed_edges:
            row = adjacency.setdefault(source, {})
            row.setdefault(relation, []).append(target)
        return adjacency


@dataclass(frozen=True)
class TemporalWalkGenerator:
    """Temporal random walks that enforce non-decreasing event time."""

    walk_length: int = 10
    walks_per_node: int = 4
    seed: int = 0
    min_time_gap: float = 0.0

    def __post_init__(self) -> None:
        if self.walk_length <= 1:
            raise ValueError("walk_length must be > 1")
        if self.walks_per_node <= 0:
            raise ValueError("walks_per_node must be positive")
        if self.min_time_gap < 0.0:
            raise ValueError("min_time_gap must be non-negative")

    def generate(
        self,
        temporal_edges: Sequence[tuple[int, int, float]],
        start_nodes: Sequence[int],
    ) -> list[list[int]]:
        adjacency = self._build_adjacency(temporal_edges)
        if not adjacency:
            return []

        rng = random.Random(self.seed)
        walks: list[list[int]] = []
        for start in start_nodes:
            if start not in adjacency:
                continue
            for _ in range(self.walks_per_node):
                walk = [start]
                current = start
                last_time = float("-inf")
                for _ in range(self.walk_length - 1):
                    candidates = [
                        (dst, event_time)
                        for dst, event_time in adjacency.get(current, [])
                        if event_time > last_time + self.min_time_gap
                    ]
                    if not candidates:
                        break
                    dst, event_time = rng.choice(candidates)
                    walk.append(dst)
                    current = dst
                    last_time = event_time
                walks.append(walk)
        return walks

    def _build_adjacency(
        self,
        temporal_edges: Sequence[tuple[int, int, float]],
    ) -> dict[int, list[tuple[int, float]]]:
        adjacency: dict[int, list[tuple[int, float]]] = {}
        for source, target, timestamp in temporal_edges:
            adjacency.setdefault(source, []).append((target, float(timestamp)))
        for neighbors in adjacency.values():
            neighbors.sort(key=lambda item: item[1])
        return adjacency


@dataclass(frozen=True)
class SignedEdgeSampler:
    """Negative edge sampler with optional sign-aware rejection."""

    negative_samples: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.negative_samples <= 0:
            raise ValueError("negative_samples must be positive")

    def sample(
        self,
        node_count: int,
        edges: Sequence[tuple[int, int]],
    ) -> dict[tuple[int, int], list[tuple[int, int]]]:
        if node_count <= 1:
            return {edge: [] for edge in edges}
        existing = {(src, dst) for src, dst in edges}
        rng = random.Random(self.seed)

        negatives: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for source, target in edges:
            if source < 0 or target < 0 or source >= node_count or target >= node_count:
                raise ValueError("edge endpoints must be in [0, node_count)")

            sampled = []
            while len(sampled) < self.negative_samples:
                negative_target = rng.randrange(node_count)
                if negative_target == target or (source, negative_target) in existing:
                    continue
                sampled.append((source, negative_target))
            negatives[(source, target)] = sampled
        return negatives
