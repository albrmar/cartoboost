from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any


__all__ = ["build_zip_sparse_sets", "coerce_zip_to_feature_id"]


_NON_DIGITS = re.compile(r"\D")


def coerce_zip_to_feature_id(value: Any, *, strict: bool = False) -> int | None:
    """Convert a ZIP-like value to a deterministic non-negative integer ID."""
    text = _coerce_zip_string(value, strict=strict)
    if text is None:
        return None
    return int(text)


def build_zip_sparse_sets(
    origin_zip: Sequence[Any] | None = None,
    destination_zip: Sequence[Any] | None = None,
    *,
    include_raw: bool = True,
    zip3_only: bool = False,
    parent_prefixes: Sequence[int] | None = None,
    include_match_indicator: bool = True,
    strict: bool = False,
) -> dict[str, list[list[int]]]:
    """Build sparse-set columns for ZIP features.

    Returns sparse_set columns keyed by role/prefix so ZIP origin/destination columns
    are explicitly surfaced as geographic context.
    """
    if origin_zip is None and destination_zip is None:
        raise ValueError("origin_zip and destination_zip cannot both be None")

    if zip3_only:
        if include_raw:
            raise ValueError("zip3_only cannot be used with include_raw=True")
        parent_prefixes = (3,)
        include_raw = False
    parent_prefixes = _normalize_prefixes(parent_prefixes or (3, 2))
    ozip_codes = _coerce_zip_sequence(origin_zip, strict=strict, name="origin_zip") if origin_zip else []
    dzip_codes = _coerce_zip_sequence(destination_zip, strict=strict, name="destination_zip") if destination_zip else []
    if origin_zip and destination_zip and len(ozip_codes) != len(dzip_codes):
        raise ValueError("origin_zip and destination_zip must have the same number of rows")
    row_count = len(ozip_codes) if origin_zip else len(dzip_codes)
    sparse_sets: dict[str, list[list[int]]] = {}

    if origin_zip is not None:
        sparse_sets.update(
            _zip_sparse_columns(
                ozip_codes,
                prefix="ozip",
                include_raw=include_raw,
                parent_prefixes=parent_prefixes,
                row_count=row_count,
            )
        )
    if destination_zip is not None:
        sparse_sets.update(
            _zip_sparse_columns(
                dzip_codes,
                prefix="dzip",
                include_raw=include_raw,
                parent_prefixes=parent_prefixes,
                row_count=row_count,
            )
        )

    if include_match_indicator and origin_zip is not None and destination_zip is not None:
        sparse_sets["zip_match"] = [
            [1] if ozip == dzip and ozip is not None else []
            for ozip, dzip in zip(ozip_codes, dzip_codes)
        ]

    return sparse_sets


def _zip_sparse_columns(
    codes: list[str | None],
    *,
    prefix: str,
    include_raw: bool,
    parent_prefixes: tuple[int, ...],
    row_count: int,
) -> dict[str, list[list[int]]]:
    columns: dict[str, list[list[int]]] = {}
    if include_raw:
        columns[f"{prefix}_zip5"] = [[] for _ in range(row_count)]
    for level in parent_prefixes:
        if include_raw and level == 5:
            continue
        columns[f"{prefix}_zip_p{level}"] = [[] for _ in range(row_count)]

    for idx, code in enumerate(codes):
        if code is None:
            continue
        if include_raw:
            columns[f"{prefix}_zip5"][idx].append(int(code))
        for level in parent_prefixes:
            if include_raw and level == 5:
                continue
            columns[f"{prefix}_zip_p{level}"][idx].append(int(code[:level]))

    return columns


def _normalize_prefixes(prefixes: Sequence[int]) -> tuple[int, ...]:
    cleaned = []
    for value in prefixes:
        if not isinstance(value, int):
            raise ValueError("parent_prefixes must be integers")
        if value <= 0:
            raise ValueError("parent_prefixes must be positive")
        if value > 5:
            value = 5
        cleaned.append(value)
    if not cleaned:
        raise ValueError("parent_prefixes must contain at least one level")
    # remove duplicates while keeping original order
    unique: list[int] = []
    for value in cleaned:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def _coerce_zip_sequence(
    values: Sequence[Any],
    *,
    strict: bool,
    name: str,
) -> list[str | None]:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one row")
    coerced: list[str | None] = []
    for idx, value in enumerate(values):
        code = _coerce_zip_string(value, strict=strict)
        if code is None and strict:
            raise ValueError(f"{name} contains invalid ZIP value at row {idx}")
        coerced.append(code)
    return coerced


def _coerce_zip_string(value: Any, strict: bool) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        if strict:
            raise ValueError("boolean ZIP values are not supported")
        return None
    if isinstance(value, int):
        if value < 0:
            if strict:
                raise ValueError("ZIP values must be non-negative")
            return None
        text = str(value)
    elif isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            if strict:
                raise ValueError("ZIP values must be finite")
            return None
        if not value.is_integer():
            if strict:
                raise ValueError("ZIP values must be integer-like")
            return None
        if value < 0:
            if strict:
                raise ValueError("ZIP values must be non-negative")
            return None
        text = str(int(value))
    else:
        text = str(value).strip()
        if not text:
            if strict:
                raise ValueError("ZIP values must be non-empty")
            return None
        text = _NON_DIGITS.sub("", text)
        if not text:
            if strict:
                raise ValueError(f"ZIP value {value!r} has no digits")
            return None

    text = text.zfill(5)[:5]
    return text
