from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

__all__ = [
    "build_s2_sparse_sets",
    "encode_s2_cells",
    "latlng_to_s2_id",
    "normalize_s2_id",
    "s2_parent_id",
]


def normalize_s2_id(value: Any) -> int:
    """Normalize an S2 cell ID into a non-negative integer."""

    if isinstance(value, bool):
        raise ValueError("S2 IDs must be non-negative integers or integer strings")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("S2 IDs must be non-negative")
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("S2 IDs must not be empty")
        if text.startswith("-"):
            raise ValueError("S2 IDs must be non-negative")
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text, 10)
        except ValueError as exc:
            raise ValueError("S2 IDs must be decimal or 0x-prefixed integer strings") from exc
    raise ValueError("S2 IDs must be non-negative integers or integer strings")


def latlng_to_s2_id(latitude: Any, longitude: Any, *, level: int) -> int:
    """Encode a latitude/longitude pair with the optional ``s2sphere`` package."""

    s2sphere = _load_s2sphere()
    lvl = _normalize_level(level, "level")
    lat = _normalize_coordinate(latitude, "latitude")
    lng = _normalize_coordinate(longitude, "longitude")
    lat_lng = s2sphere.LatLng.from_degrees(lat, lng)
    cell = s2sphere.CellId.from_lat_lng(lat_lng).parent(lvl)
    return normalize_s2_id(cell.id())


def s2_parent_id(cell: Any, *, parent_level: int) -> int:
    """Return an S2 parent cell ID with the optional ``s2sphere`` package."""

    s2sphere = _load_s2sphere()
    parent = _normalize_level(parent_level, "parent_level")
    cell_id = s2sphere.CellId(normalize_s2_id(cell))
    if parent > cell_id.level():
        raise ValueError("parent_level must be less than or equal to the cell level")
    return normalize_s2_id(cell_id.parent(parent).id())


def encode_s2_cells(
    latitudes: Iterable[Any],
    longitudes: Iterable[Any],
    *,
    level: int,
) -> list[int]:
    """Encode latitude/longitude rows into S2 integer cell IDs."""

    lat_values = list(latitudes)
    lng_values = list(longitudes)
    if len(lat_values) != len(lng_values):
        raise ValueError("latitudes and longitudes must have the same number of rows")
    if not lat_values:
        raise ValueError("latitudes and longitudes must contain at least one row")
    return [
        latlng_to_s2_id(latitude, longitude, level=level)
        for latitude, longitude in zip(lat_values, lng_values, strict=True)
    ]


def build_s2_sparse_sets(
    coordinates: dict[str, tuple[Iterable[Any], Iterable[Any]]],
    *,
    level: int,
    parent_levels: Iterable[int] = (),
) -> dict[str, list[list[int]]]:
    """Build sparse-set columns from named latitude/longitude coordinate pairs.

    ``coordinates`` maps output sparse-set names to ``(latitudes, longitudes)``.
    Each row contains the child S2 cell plus any requested S2 parent cells.
    """

    if not coordinates:
        raise ValueError("coordinates cannot be empty")
    child_level = _normalize_level(level, "level")
    parents = [_normalize_level(value, "parent_levels") for value in parent_levels]
    for parent in parents:
        if parent >= child_level:
            raise ValueError("parent_levels must be less than level")

    sparse_sets: dict[str, list[list[int]]] = {}
    expected_rows: int | None = None
    for name, pair in coordinates.items():
        if not name:
            raise ValueError("coordinate names must be non-empty")
        try:
            latitudes, longitudes = pair
        except (TypeError, ValueError) as exc:
            raise ValueError("coordinate values must be (latitudes, longitudes) pairs") from exc
        cells = encode_s2_cells(latitudes, longitudes, level=child_level)
        if expected_rows is None:
            expected_rows = len(cells)
        elif len(cells) != expected_rows:
            raise ValueError(
                f"coordinate feature '{name}' has {len(cells)} rows, expected {expected_rows}"
            )
        column: list[list[int]] = []
        for cell in cells:
            row = [cell]
            row.extend(s2_parent_id(cell, parent_level=parent) for parent in parents)
            column.append(sorted(set(row)))
        sparse_sets[name] = column
    return sparse_sets


def _load_s2sphere() -> Any:
    try:
        return importlib.import_module("s2sphere")
    except ImportError as exc:
        raise ImportError(
            "S2 auto-encoding requires the optional 's2sphere' package. "
            "Install with `pip install cartoboost[s2]` or `pip install s2sphere`."
        ) from exc


def _normalize_coordinate(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite coordinate")
    try:
        coordinate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite coordinate") from exc
    if not coordinate == coordinate or coordinate in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be a finite coordinate")
    return coordinate


def _normalize_level(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer S2 level")
    try:
        level = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer S2 level") from exc
    if level != value:
        raise ValueError(f"{field_name} must be an integer S2 level")
    if level < 0 or level > 30:
        raise ValueError(f"{field_name} must be between 0 and 30")
    return level
