from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSchema:
    """Python helper for GeoBoost dense and sparse feature metadata."""

    dense: list[Any] | tuple[Any, ...]
    sparse_sets: list[Any] | tuple[Any, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dense": list(self.dense),
            "sparse_sets": list(self.sparse_sets or []),
        }

    def to_rust_payload(self, dense_width: int, sparse_names: list[str]) -> dict[str, Any]:
        dense_entries = list(self.dense)
        sparse_entries = list(self.sparse_sets or [])
        names = [_entry_name(entry, idx, "feature") for idx, entry in enumerate(dense_entries)]
        kinds = [_entry_kind(entry, "numeric") for entry in dense_entries]
        names.extend(
            _entry_name(entry, idx, "sparse_set") for idx, entry in enumerate(sparse_entries)
        )
        kinds.extend(_entry_kind(entry, "sparse_set") for entry in sparse_entries)
        _validate_length(names, kinds, dense_width, sparse_names)
        return {"names": names, "kinds": kinds}

    def to_json(self, dense_width: int, sparse_names: list[str]) -> str:
        return json.dumps(self.to_rust_payload(dense_width, sparse_names))


def normalize_feature_kind(kind: Any, entry: dict[str, Any] | None = None) -> Any:
    if isinstance(kind, dict):
        if "H3SparseSet" in kind:
            return "SparseSet"
        if "h3_sparse_set" in kind:
            return "SparseSet"
        if "zip_sparse_set" in kind:
            return "SparseSet"
        if "Periodic" in kind:
            return {"Periodic": {"period": _positive_period(kind["Periodic"]["period"])}}
        if "periodic" in kind:
            value = kind["periodic"]
            period = value.get("period", 24) if isinstance(value, dict) else value
            return {"Periodic": {"period": _positive_period(period)}}
    if kind in {"Numeric", "numeric"}:
        return "Numeric"
    if kind in {"SparseSet", "sparse_set", "sparse"}:
        return "SparseSet"
    if kind in {
        "ZoneSparseSet",
        "zone_sparse_set",
        "zone-sparse-set",
        "RegionSparseSet",
        "region_sparse_set",
        "region-sparse-set",
        "GeoZoneSparseSet",
        "geo_zone_sparse_set",
        "geo-zone-sparse-set",
        "AreaSparseSet",
        "area_sparse_set",
        "area-sparse-set",
        "GeoAbstractSparseSet",
        "geo_abstract_sparse_set",
        "geo-abstract-sparse-set",
        "H3SparseSet",
        "h3_sparse_set",
        "h3-sparse-set",
        "ZipSparseSet",
        "zip_sparse_set",
        "zip-sparse-set",
        "zip3_sparse_set",
        "zip3-sparse-set",
        "GeoSparseSet",
        "geo_sparse_set",
        "geo-sparse-set",
    }:
        return "SparseSet"
    if kind in {"Periodic", "periodic"}:
        period = 24 if entry is None else entry.get("period", 24)
        return {"Periodic": {"period": _positive_period(period)}}
    raise ValueError(f"unknown feature kind {kind!r}")


def _entry_name(entry: Any, idx: int, prefix: str) -> str:
    if isinstance(entry, dict):
        return str(entry.get("name", f"{prefix}_{idx}"))
    if isinstance(entry, tuple) and len(entry) == 2:
        return str(entry[0])
    return str(entry)


def _entry_kind(entry: Any, default: str) -> Any:
    if isinstance(entry, dict):
        return normalize_feature_kind(entry.get("kind", entry.get("role", default)), entry)
    if isinstance(entry, tuple) and len(entry) == 2:
        return normalize_feature_kind(entry[1])
    if default == "sparse_set":
        return "SparseSet"
    return "Numeric"


def _positive_period(period: Any) -> int:
    try:
        value = float(period)
    except (TypeError, ValueError) as exc:
        raise ValueError("periodic feature_schema entries require a positive period") from exc
    if not math.isfinite(value) or value <= 0 or not value.is_integer():
        raise ValueError("periodic feature_schema entries require a positive integer period")
    return int(value)


def _validate_length(
    names: list[str],
    kinds: list[Any],
    dense_width: int,
    sparse_names: list[str],
) -> None:
    expected = dense_width + len(sparse_names)
    if len(names) != len(kinds):
        raise ValueError("feature_schema names length must match kinds length")
    if len(names) != expected:
        raise ValueError(
            f"feature_schema length {len(names)} does not match dataset feature count {expected}"
        )
