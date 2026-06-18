"""Feature containers for graph-derived model inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GraphFeatureBundle:
    """Dense graph-derived features ready to be merged into CartoBoost training data."""

    embeddings: np.ndarray
    sparse_sets: dict[str, list[list[int]]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    node_ids: list[Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if self.feature_names and len(self.feature_names) != self.embeddings.shape[1]:
            raise ValueError("feature_names length must match embedding width")
        if self.node_ids is None:
            return
        if len(self.node_ids) != self.embeddings.shape[0]:
            raise ValueError("node_ids length must match row count")

    def augment_dense(self, dense: Any) -> np.ndarray:
        """Concatenate graph embeddings with an existing dense feature matrix."""
        dense_matrix = np.asarray(dense, dtype=np.float64)
        if dense_matrix.ndim != 2:
            raise ValueError("dense must be 2D")
        if dense_matrix.shape[0] != self.embeddings.shape[0]:
            raise ValueError("dense and embeddings must have the same row count")
        return np.hstack([dense_matrix, self.embeddings.astype(np.float64)])

    def with_directional_features(
        self,
        directional_features: Sequence[Sequence[float]],
        feature_names: list[str] | None = None,
    ) -> GraphFeatureBundle:
        extras = np.asarray(list(directional_features), dtype=np.float32)
        if extras.ndim != 2:
            raise ValueError("directional_features must be 2D")
        if extras.shape[0] != self.embeddings.shape[0]:
            raise ValueError("directional_features row count must match embeddings")
        if feature_names is not None and len(feature_names) != extras.shape[1]:
            raise ValueError("directional feature_names length must match feature width")

        combined_names = list(self.feature_names)
        if feature_names:
            combined_names.extend(feature_names)
        return GraphFeatureBundle(
            embeddings=np.hstack([self.embeddings, extras.astype(np.float32)]),
            sparse_sets=self.sparse_sets,
            feature_names=combined_names,
            node_ids=self.node_ids,
            provenance=dict(self.provenance),
        )

    def feature_schema_entries(self, prefix: str = "graph") -> list[str]:
        if self.feature_names:
            return list(self.feature_names)
        return [f"{prefix}_{index:02d}" for index in range(self.embeddings.shape[1])]

    def sparse_schema_names(self) -> list[str]:
        return list(self.sparse_sets.keys())

    def dense_schema_entries(self, prefix: str = "graph") -> list[tuple[str, str]]:
        return [(name, "numeric") for name in self.feature_schema_entries(prefix)]

    def feature_schema_payload(self, prefix: str = "graph") -> dict[str, list[Any]]:
        names = [*self.feature_schema_entries(prefix), *self.sparse_schema_names()]
        kinds = [
            *(["Numeric"] * self.embeddings.shape[1]),
            *(["SparseSet"] * len(self.sparse_sets)),
        ]
        return {"names": names, "kinds": kinds}

    def training_config_metadata(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_schema_entries(),
            "sparse_sets": self.sparse_schema_names(),
            "row_count": int(self.embeddings.shape[0]),
            "embedding_width": int(self.embeddings.shape[1]),
            "provenance": dict(self.provenance),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "embeddings": self.embeddings.tolist(),
            "feature_names": self.feature_names,
            "sparse_sets": self.sparse_sets,
            "node_ids": list(self.node_ids) if self.node_ids is not None else None,
            "provenance": dict(self.provenance),
        }
