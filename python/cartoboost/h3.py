from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def normalize_h3_id(value: Any) -> int:
    """Normalize an H3-style sparse ID."""

    if isinstance(value, bool):
        raise ValueError("H3 IDs must be non-negative integers or integer strings")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("H3 IDs must be non-negative")
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("H3 IDs must not be empty")
        if text.startswith("-"):
            raise ValueError("H3 IDs must be non-negative")
        if text.lower().startswith("0x"):
            return _parse_h3_string(text[2:], 16)
        if text.isdecimal():
            return int(text, 10)
        return _parse_h3_string(text, 16)
    raise ValueError("H3 IDs must be non-negative integers or integer strings")


def expand_h3_sparse_set(
    values: Iterable[Any],
    *,
    resolution: int,
    parent_resolutions: Iterable[int] = (),
) -> list[int]:
    """Return normalized IDs plus deterministic scaffold parent IDs.

    This is a test scaffold, not real H3 parent cartometry. It creates stable
    synthetic parent IDs so sparse-set code can exercise hierarchical columns
    without depending on an H3 library.
    """

    child_resolution = _normalize_resolution(resolution, "resolution")
    parents = [_normalize_resolution(value, "parent_resolutions") for value in parent_resolutions]
    for parent in parents:
        if parent >= child_resolution:
            raise ValueError("parent_resolutions must be less than resolution")

    expanded: set[int] = set()
    for value in values:
        cell = normalize_h3_id(value)
        expanded.add(cell)
        for parent in parents:
            expanded.add(scaffold_h3_parent_id(cell, child_resolution, parent))
    return sorted(expanded)


def scaffold_h3_parent_id(cell: Any, resolution: int, parent_resolution: int) -> int:
    """Build a deterministic synthetic parent ID for tests.

    The output is stable for a cell/resolution pair, but it is not a real H3
    parent and must not be used for cartospatial semantics.
    """

    child = _normalize_resolution(resolution, "resolution")
    parent = _normalize_resolution(parent_resolution, "parent_resolution")
    if parent >= child:
        raise ValueError("parent_resolution must be less than resolution")
    normalized_cell = normalize_h3_id(cell)
    resolution_gap = child - parent
    bucket = normalized_cell >> (resolution_gap * 3)
    return (1 << 63) | (parent << 56) | bucket


def _parse_h3_string(text: str, base: int) -> int:
    if not text:
        raise ValueError("H3 IDs must not be empty")
    try:
        value = int(text, base)
    except ValueError as exc:
        raise ValueError("H3 IDs must be decimal or hexadecimal integer strings") from exc
    if value < 0:
        raise ValueError("H3 IDs must be non-negative")
    return value


def _normalize_resolution(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer H3 resolution")
    try:
        resolution = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer H3 resolution") from exc
    if resolution != value:
        raise ValueError(f"{field_name} must be an integer H3 resolution")
    if resolution < 0 or resolution > 15:
        raise ValueError(f"{field_name} must be between 0 and 15")
    return resolution
