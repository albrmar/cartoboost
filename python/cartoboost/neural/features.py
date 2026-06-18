"""Neural feature tooling for CartoBoost."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .._native import NeuralEmbeddingFeatures as _NativeNeuralEmbeddingFeatures


class ArtifactFallback(str):
    ZERO_VECTOR = "zero_vector"
    GLOBAL_MEAN = "global_mean_vector"
    PARENT_CELL = "parent_cell"


class NeuralEmbeddingFeatures:
    """Learn and persist embedding table artifacts for CartoBoost neural features."""

    def __init__(
        self,
        *,
        dim: int,
        fallback: str = ArtifactFallback.GLOBAL_MEAN,
        random_state: int | None = 42,
        parent_resolution: int | None = None,
    ) -> None:
        self.dim = int(dim)
        self.fallback = str(fallback)
        self.random_state = random_state
        self.parent_resolution = parent_resolution
        self._backend = _NativeNeuralEmbeddingFeatures(
            dim=self.dim,
            fallback=self.fallback,
            random_state=None if random_state is None else int(random_state),
            parent_resolution=parent_resolution,
        )

    def fit(self, ids: Iterable[Any], target: Iterable[Any]) -> NeuralEmbeddingFeatures:
        id_values = _as_1d_array(ids, np.uint64, "ids")
        target_values = _as_1d_array(target, np.float64, "target")
        if id_values.size != target_values.size:
            raise ValueError("ids and target must have the same length")

        self._backend.fit(id_values, target_values)
        return self

    def transform(self, ids: Iterable[Any]) -> np.ndarray:
        id_values = _as_1d_array(ids, np.uint64, "ids")
        if id_values.size == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.asarray(self._backend.transform(id_values), dtype=np.float32)

    def fit_transform(self, ids: Iterable[Any], target: Iterable[Any]) -> np.ndarray:
        id_values = _as_1d_array(ids, np.uint64, "ids")
        target_values = _as_1d_array(target, np.float64, "target")
        if id_values.size != target_values.size:
            raise ValueError("ids and target must have the same length")

        if id_values.size == 0:
            self.fit(id_values, target_values)
            return np.empty((0, self.dim), dtype=np.float32)

        return np.asarray(self._backend.fit_transform(id_values, target_values), dtype=np.float32)

    def export(self, path: str | Path) -> Path:
        path = Path(path)
        self._backend.export(str(path))
        return path

    @classmethod
    def from_artifact(cls, path: str | Path) -> NeuralEmbeddingFeatures:
        path = Path(path)
        native = _NativeNeuralEmbeddingFeatures.from_artifact(str(path))
        instance = cls(
            dim=native.dim,
            fallback=native.fallback,
            random_state=None,
            parent_resolution=native.parent_resolution,
        )
        instance._backend = native
        return instance

    def artifact_rows(self) -> list[dict[str, Any]]:
        return [
            {"id": int(row_id), "values": values}
            for row_id, values in self._backend.artifact_rows()
        ]


def _as_1d_array(values: Iterable[Any], dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 0:
        raise ValueError(f"{name} must be 1D")
    if array.ndim != 1:
        array = np.ravel(array)
    return np.ascontiguousarray(array, dtype=dtype)
