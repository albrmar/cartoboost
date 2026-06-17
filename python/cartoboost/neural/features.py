from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

_ARTIFACT_TYPE = "cartoboost.neural.embedding_table"
_ARTIFACT_VERSION = 1


class ArtifactFallback(str):
    ZERO_VECTOR = "zero_vector"
    GLOBAL_MEAN = "global_mean_vector"
    PARENT_CELL = "parent_cell"


def _to_u64_ids(values: Iterable[Any]) -> np.ndarray:
    ids = np.asarray(list(values), dtype=np.uint64)
    return ids


def _to_f32_vector(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return array.astype(np.float32)


def _id_seed(id_value: np.uint64, random_state: int | None) -> int:
    hasher = hashlib.blake2b(digest_size=16)
    if random_state is not None:
        hasher.update(str(int(random_state)).encode("utf-8"))
    hasher.update(str(int(id_value)).encode("utf-8"))
    return int.from_bytes(hasher.digest()[:8], byteorder="little", signed=False)


def _write_le_u64(value: int) -> bytes:
    return int(value).to_bytes(8, byteorder="little", signed=False)


def _write_le_u32(value: int) -> bytes:
    return int(value).to_bytes(4, byteorder="little", signed=False)


def _write_u8(value: int) -> bytes:
    return int(value).to_bytes(1, byteorder="little", signed=False)


def _build_checksum(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    fallback = metadata["fallback"]
    fallback_name = fallback["strategy"]

    hasher = hashlib.sha256()
    hasher.update(metadata["artifact_type"].encode("utf-8"))
    hasher.update(_write_le_u32(int(metadata["artifact_version"])))
    hasher.update(_write_u8(0))  # id_type "u64"
    hasher.update(_write_le_u64(int(metadata["dim"])))
    hasher.update(_write_le_u64(int(metadata["row_count"])))

    if fallback_name == ArtifactFallback.ZERO_VECTOR:
        hasher.update(_write_u8(0))
    elif fallback_name == ArtifactFallback.GLOBAL_MEAN:
        hasher.update(_write_u8(1))
    else:
        hasher.update(_write_u8(2))
        hasher.update(_write_u8(fallback["params"]["parent_resolution"]))

    for row in sorted(rows, key=lambda row: int(row["id"])):
        hasher.update(_write_le_u64(int(row["id"])))
        for value in row["values"]:
            hasher.update(struct.pack("<f", float(value)))

    return hasher.hexdigest()


def _as_fallback_metadata(strategy: str, parent_resolution: int | None = None) -> dict[str, Any]:
    if strategy == ArtifactFallback.ZERO_VECTOR:
        return {"strategy": ArtifactFallback.ZERO_VECTOR}
    if strategy == ArtifactFallback.GLOBAL_MEAN:
        return {"strategy": ArtifactFallback.GLOBAL_MEAN}
    if strategy != ArtifactFallback.PARENT_CELL:
        raise ValueError(f"unsupported fallback strategy {strategy}")

    if parent_resolution is None:
        raise ValueError("parent_resolution is required for parent_cell fallback")
    return {
        "strategy": ArtifactFallback.PARENT_CELL,
        "params": {"parent_resolution": int(parent_resolution)},
    }


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
        if dim <= 0:
            raise ValueError("dim must be positive")
        if fallback not in {
            ArtifactFallback.ZERO_VECTOR,
            ArtifactFallback.GLOBAL_MEAN,
            ArtifactFallback.PARENT_CELL,
        }:
            raise ValueError("fallback must be zero_vector, global_mean_vector, or parent_cell")
        if fallback == ArtifactFallback.PARENT_CELL and parent_resolution is None:
            raise ValueError("parent_resolution must be set for parent_cell fallback")

        self.dim = int(dim)
        self.fallback = fallback
        self.random_state = random_state
        self.parent_resolution = parent_resolution
        self._embeddings: dict[int, np.ndarray] = {}
        self._global_embedding = np.zeros(self.dim, dtype=np.float32)
        self._artifact_fallback = _as_fallback_metadata(fallback, parent_resolution)
        self._fitted = False

    def fit(self, ids: Iterable[Any], target: Iterable[Any]) -> NeuralEmbeddingFeatures:
        id_values = _to_u64_ids(ids)
        target_values = _to_f32_vector(target)
        if id_values.size != target_values.shape[0]:
            raise ValueError("ids and target must have the same length")

        if id_values.size == 0:
            self._embeddings = {}
            self._global_embedding = np.zeros(self.dim, dtype=np.float32)
            self._fitted = True
            return self

        sums: dict[int, float] = {}
        counts: dict[int, int] = {}
        for id_value, residual in zip(id_values, target_values, strict=True):
            key = int(id_value)
            sums[key] = sums.get(key, 0.0) + float(residual)
            counts[key] = counts.get(key, 0) + 1

        self._embeddings = {}
        for key, total in sums.items():
            mean = float(total / counts[key])
            rng = np.random.default_rng(_id_seed(np.uint64(key), self.random_state))
            vector = rng.normal(loc=0.0, scale=0.1, size=self.dim).astype(np.float32)
            if self.dim >= 1:
                vector[0] = vector[0] + mean
            self._embeddings[key] = vector

        all_vectors = np.stack(list(self._embeddings.values()), axis=0).astype(np.float32)
        self._global_embedding = all_vectors.mean(axis=0)
        self._fitted = True
        return self

    def transform(self, ids: Iterable[Any]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("transform called before fit or load")

        id_values = _to_u64_ids(ids)
        output = np.zeros((id_values.size, self.dim), dtype=np.float32)

        for row_index, id_value in enumerate(id_values):
            vector = self._embeddings.get(int(id_value))
            if vector is not None:
                output[row_index] = vector
                continue

            if self.fallback == ArtifactFallback.ZERO_VECTOR:
                continue

            output[row_index] = self._global_embedding

        return output

    def fit_transform(self, ids: Iterable[Any], target: Iterable[Any]) -> np.ndarray:
        self.fit(ids, target)
        return self.transform(ids)

    def export(self, path: str | Path) -> Path:
        if not self._fitted:
            raise RuntimeError("export called before fit")

        payload = self._to_artifact_payload()
        file_path = Path(path)
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return file_path

    @classmethod
    def from_artifact(cls, path: str | Path) -> NeuralEmbeddingFeatures:
        file_path = Path(path)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        rows = payload["rows"]

        if metadata["artifact_type"] != _ARTIFACT_TYPE:
            raise ValueError(f"unexpected artifact_type {metadata['artifact_type']}")
        if metadata["artifact_version"] != _ARTIFACT_VERSION:
            raise ValueError(f"unsupported artifact_version {metadata['artifact_version']}")
        if metadata["dim"] <= 0:
            raise ValueError("embedding dimension must be positive")
        if metadata["row_count"] != len(rows):
            raise ValueError(
                f"row_count mismatch: expected {metadata['row_count']}, got {len(rows)}"
            )

        expected = _build_checksum(metadata, rows)
        if metadata["checksum"] != expected:
            raise ValueError(f"checksum mismatch: expected {metadata['checksum']}, got {expected}")

        fallback = metadata.get("fallback", {})
        strategy = fallback.get("strategy", ArtifactFallback.GLOBAL_MEAN)
        instance = cls(
            dim=int(metadata["dim"]),
            fallback=strategy,
            parent_resolution=fallback.get("params", {}).get("parent_resolution"),
        )

        instance._embeddings = {
            int(row["id"]): np.asarray(row["values"], dtype=np.float32) for row in rows
        }

        for id_value, vector in instance._embeddings.items():
            if len(vector) != instance.dim:
                raise ValueError(
                    f"row {id_value} has {len(vector)} values but expected {instance.dim}"
                )

        vectors = list(instance._embeddings.values())
        if vectors:
            instance._global_embedding = np.mean(np.stack(vectors, axis=0), axis=0)
        else:
            instance._global_embedding = np.zeros(instance.dim, dtype=np.float32)

        instance._fitted = True
        return instance

    def _to_artifact_payload(self) -> dict[str, Any]:
        rows = [
            {
                "id": int(id_value),
                "values": np.asarray(values, dtype=np.float32).tolist(),
            }
            for id_value, values in sorted(self._embeddings.items(), key=lambda item: item[0])
        ]

        metadata = {
            "artifact_type": _ARTIFACT_TYPE,
            "artifact_version": _ARTIFACT_VERSION,
            "dim": self.dim,
            "id_type": "u64",
            "row_count": len(rows),
            "fallback": self._artifact_fallback,
            "checksum": "",
        }
        metadata["checksum"] = _build_checksum(metadata, rows)
        return {"metadata": metadata, "rows": rows}

    def artifact_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "id": int(id_value),
                "values": vector.tolist(),
            }
            for id_value, vector in sorted(self._embeddings.items(), key=lambda item: item[0])
        ]
