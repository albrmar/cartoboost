from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .._native_wrappers import NativeForecastWrapper

CoordinateInput = Mapping[str, Sequence[float]] | Sequence[tuple[str, float, float]]


class KrigingForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust ordinary kriging forecasting binding."""

    native_class_name = "KrigingForecaster"

    def __init__(
        self,
        coordinates: CoordinateInput,
        range: float = 1.0,
        nugget: float = 1.0e-6,
        sill: float = 1.0,
        variogram_model: str = "exponential",
        drift: str = "ordinary",
        anisotropy_angle_degrees: float = 0.0,
        anisotropy_scaling: float = 1.0,
        max_neighbors: int | None = None,
        min_neighbors: int = 1,
        max_distance: float | None = None,
        **params: Any,
    ) -> None:
        coordinate_rows = _normalize_coordinates(coordinates)
        super().__init__(
            coordinates=coordinate_rows,
            range=float(range),
            nugget=float(nugget),
            sill=float(sill),
            variogram_model=str(variogram_model),
            drift=str(drift),
            anisotropy_angle_degrees=float(anisotropy_angle_degrees),
            anisotropy_scaling=float(anisotropy_scaling),
            max_neighbors=None if max_neighbors is None else int(max_neighbors),
            min_neighbors=int(min_neighbors),
            max_distance=None if max_distance is None else float(max_distance),
        )
        self.coordinates = coordinate_rows
        self.range = float(range)
        self.nugget = float(nugget)
        self.sill = float(sill)
        self.variogram_model = str(variogram_model)
        self.drift = str(drift)
        self.anisotropy_angle_degrees = float(anisotropy_angle_degrees)
        self.anisotropy_scaling = float(anisotropy_scaling)
        self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
        self.min_neighbors = int(min_neighbors)
        self.max_distance = None if max_distance is None else float(max_distance)
        for key, value in params.items():
            setattr(self, key, value)


def _normalize_coordinates(coordinates: CoordinateInput) -> list[tuple[str, float, float]]:
    if isinstance(coordinates, Mapping):
        rows = []
        for series_id, pair in coordinates.items():
            if len(pair) != 2:
                raise ValueError("kriging coordinates must map series_id to (x, y)")
            rows.append((str(series_id), float(pair[0]), float(pair[1])))
        return rows
    return [(str(series_id), float(x), float(y)) for series_id, x, y in coordinates]


__all__ = ["KrigingForecaster"]
