from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from ..frequency import next_timestamps
from ..schema import (
    ForecastFrame,
    PredictionInterval,
)
from ..schema import (
    ForecastResult as TableForecastResult,
)


@dataclass(frozen=True)
class ForecastResult:
    mean: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.mean, dtype=dtype)


def _as_series_matrix(
    y: Any,
    *,
    timestamps: Any | None = None,
    series_ids: list[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[Any], bool]:
    inferred_timestamps = timestamps
    inferred_series_ids = series_ids
    if isinstance(y, dict):
        if not y:
            raise ValueError("y must contain at least one series")
        inferred_series_ids = list(y.keys()) if inferred_series_ids is None else inferred_series_ids
        values = [np.asarray(v, dtype=float) for v in y.values()]
        lengths = {value.shape[0] for value in values}
        if len(lengths) != 1:
            raise ValueError("all panel series must have the same length")
        matrix = np.column_stack(values)
        return (
            _validate_matrix(matrix),
            _coerce_timestamps(inferred_timestamps),
            inferred_series_ids,
            False,
        )

    if inferred_timestamps is None and hasattr(y, "index"):
        index = y.index
        if not callable(index):
            inferred_timestamps = index
    if inferred_series_ids is None and hasattr(y, "columns"):
        inferred_series_ids = list(y.columns)

    arr = np.asarray(getattr(y, "to_numpy", lambda: y)(), dtype=float)
    single_series = arr.ndim == 1
    if single_series:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("y must be a 1D series or a 2D panel with shape (time, series)")
    if inferred_series_ids is None:
        inferred_series_ids = [0] if single_series else list(range(arr.shape[1]))
    if len(inferred_series_ids) != arr.shape[1]:
        raise ValueError("series_ids length must match the number of series")
    return (
        _validate_matrix(arr),
        _coerce_timestamps(inferred_timestamps),
        inferred_series_ids,
        single_series,
    )


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        raise ValueError("y must contain at least one observation")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("y must contain only finite values")
    return np.asarray(matrix, dtype=float)


def _coerce_timestamps(timestamps: Any | None) -> np.ndarray | None:
    if timestamps is None:
        return None
    arr = np.asarray(timestamps)
    if arr.ndim != 1:
        raise ValueError("timestamps must be 1D")
    return arr


def _future_timestamps(timestamps: np.ndarray | None, horizon: int) -> np.ndarray | None:
    if timestamps is None:
        return None
    if len(timestamps) == 0:
        return None
    if len(timestamps) == 1:
        return np.arange(1, horizon + 1) + timestamps[-1]
    last = timestamps[-1]
    step = timestamps[-1] - timestamps[-2]
    return np.asarray([last + step * i for i in range(1, horizon + 1)])


def _validate_horizon(horizon: int) -> int:
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    return horizon


def _z_value(level: float) -> float:
    level = float(level)
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def _residual_scale(residuals: np.ndarray) -> np.ndarray:
    if residuals.shape[0] < 2:
        return np.zeros(residuals.shape[1], dtype=float)
    scale = np.nanstd(residuals, axis=0, ddof=1)
    return np.where(np.isfinite(scale), scale, 0.0)


def _with_residual_intervals(
    mean: np.ndarray,
    residuals: np.ndarray,
    level: float,
    *,
    grow_with_horizon: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    scale = _residual_scale(residuals)
    steps = np.sqrt(np.arange(1, mean.shape[0] + 1, dtype=float)).reshape(-1, 1)
    multiplier = steps if grow_with_horizon else 1.0
    width = _z_value(level) * scale.reshape(1, -1) * multiplier
    return mean - width, mean + width


class NaiveForecaster:
    """Last-observation forecaster for single series and equal-length panels."""

    def __init__(
        self,
        *,
        prediction_interval_levels: list[float] | tuple[float, ...] = (),
    ) -> None:
        self.prediction_interval_levels = tuple(
            float(level) for level in prediction_interval_levels
        )
        self.is_fitted_ = False

    def fit(
        self,
        y: Any,
        *,
        timestamps: Any | None = None,
        series_ids: list[Any] | None = None,
    ) -> NaiveForecaster:
        if isinstance(y, ForecastFrame):
            matrix, fitted_timestamps, fitted_series_ids, single_series = _matrix_from_frame(y)
            self.forecast_frame_metadata_ = y.to_metadata()
            self.forecast_freq_ = y.freq
            self.forecast_timestamp_col_ = y.timestamp_col
            self.forecast_target_col_ = y.target_col
            self.forecast_series_id_col_ = y.series_id_col
        else:
            matrix, fitted_timestamps, fitted_series_ids, single_series = _as_series_matrix(
                y, timestamps=timestamps, series_ids=series_ids
            )
            self.forecast_frame_metadata_ = None
            self.forecast_freq_ = None
            self.forecast_timestamp_col_ = None
            self.forecast_target_col_ = None
            self.forecast_series_id_col_ = None
        fitted = np.vstack([matrix[0], matrix[:-1]])
        residuals = matrix - fitted
        self.training_values_ = matrix
        self.last_values_ = matrix[-1].copy()
        self.fitted_values_ = _restore_shape(fitted, single_series)
        self.residuals_ = _restore_shape(residuals, single_series)
        self.timestamps_ = fitted_timestamps
        self.series_ids_ = fitted_series_ids
        self.single_series_ = single_series
        self.metadata_ = {
            "model": self.__class__.__name__,
            "n_obs": int(matrix.shape[0]),
            "n_series": int(matrix.shape[1]),
            "series_ids": list(fitted_series_ids),
            "has_timestamps": fitted_timestamps is not None,
        }
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        *,
        return_interval: bool = False,
        level: float = 0.95,
        future_frame: ForecastFrame | None = None,
    ) -> np.ndarray | ForecastResult | TableForecastResult:
        self._check_is_fitted()
        horizon = _validate_horizon(horizon)
        mean = np.tile(self.last_values_.reshape(1, -1), (horizon, 1))
        if self.forecast_frame_metadata_ is not None:
            return _table_forecast_result(
                self,
                mean,
                horizon,
                levels=self.prediction_interval_levels,
                future_frame=future_frame,
            )
        return self._format_prediction(mean, return_interval=return_interval, level=level)

    def predict_interval(self, horizon: int, *, level: float = 0.95) -> ForecastResult:
        return self.predict(horizon, return_interval=True, level=level)

    def _format_prediction(
        self,
        mean: np.ndarray,
        *,
        return_interval: bool,
        level: float,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray | ForecastResult:
        restored = _restore_shape(mean, self.single_series_)
        if not return_interval:
            return restored
        residuals = _matrix_view(self.residuals_)
        lower, upper = _with_residual_intervals(mean, residuals, level)
        result_metadata = dict(self.metadata_)
        if metadata:
            result_metadata.update(metadata)
        result_metadata["interval_level"] = float(level)
        return ForecastResult(
            mean=restored,
            lower=_restore_shape(lower, self.single_series_),
            upper=_restore_shape(upper, self.single_series_),
            timestamps=_future_timestamps(self.timestamps_, mean.shape[0]),
            metadata=result_metadata,
        )

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before predict")

    def get_params(self) -> dict[str, Any]:
        return {"prediction_interval_levels": list(self.prediction_interval_levels)}

    def get_metadata(self) -> dict[str, Any]:
        self._check_is_fitted()
        return dict(self.metadata_)


def _restore_shape(values: np.ndarray, single_series: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if single_series:
        return arr[:, 0].copy()
    return arr.copy()


def _matrix_view(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _matrix_from_frame(
    frame: ForecastFrame,
) -> tuple[np.ndarray, np.ndarray | None, list[Any], bool]:
    data = frame.to_pandas()
    if frame.series_id_col is None:
        return (
            data[frame.target_col].to_numpy(dtype=float).reshape(-1, 1),
            data[frame.timestamp_col].to_numpy(),
            ["__single__"],
            True,
        )
    series_ids = frame.series_ids
    groups = [data[data[frame.series_id_col] == series_id] for series_id in series_ids]
    lengths = {len(group) for group in groups}
    if len(lengths) != 1:
        raise ValueError("local panel forecasters require balanced panel history")
    timestamp_tuples = {tuple(group[frame.timestamp_col]) for group in groups}
    if len(timestamp_tuples) != 1:
        raise ValueError("local panel forecasters require aligned panel timestamps")
    matrix = np.column_stack([group[frame.target_col].to_numpy(dtype=float) for group in groups])
    return matrix, groups[0][frame.timestamp_col].to_numpy(), list(series_ids), False


def _table_forecast_result(
    model: Any,
    mean: np.ndarray,
    horizon: int,
    *,
    levels: tuple[float, ...],
    future_frame: ForecastFrame | None = None,
) -> TableForecastResult:
    import pandas as pd

    matrix = _matrix_view(mean)
    series_ids = list(getattr(model, "series_ids_", ["__single__"]))
    if getattr(model, "single_series_", False):
        series_ids = ["__single__"]
    if future_frame is not None:
        future = future_frame.to_pandas()
        if future_frame.series_id_col is None:
            timestamps_by_series = {
                "__single__": list(future[future_frame.timestamp_col])[:horizon]
            }
        else:
            timestamps_by_series = {}
            for series_id in series_ids:
                group = future[future[future_frame.series_id_col] == series_id]
                timestamps_by_series[series_id] = list(group[future_frame.timestamp_col])[:horizon]
    else:
        if model.forecast_freq_ is not None and model.timestamps_ is not None:
            timestamps = next_timestamps(model.timestamps_[-1], horizon, model.forecast_freq_)
        else:
            timestamps = list(range(1, horizon + 1))
        timestamps_by_series = {series_id: timestamps for series_id in series_ids}

    rows: dict[str, list[Any]] = {
        "series_id": [],
        "timestamp": [],
        "horizon": [],
        "model": [],
        "mean": [],
    }
    for series_index, series_id in enumerate(series_ids):
        timestamps = timestamps_by_series[series_id]
        if len(timestamps) < horizon:
            raise ValueError("future_frame must contain at least horizon rows per series")
        for step in range(horizon):
            rows["series_id"].append(series_id)
            rows["timestamp"].append(timestamps[step])
            rows["horizon"].append(step + 1)
            rows["model"].append(model.__class__.__name__)
            rows["mean"].append(float(matrix[step, series_index]))

    residuals = _matrix_view(model.residuals_)
    for interval in sorted(levels):
        lower, upper = _with_residual_intervals(matrix, residuals, interval)
        suffix = PredictionInterval(interval, [0.0], [0.0]).suffix
        rows[f"lower_{suffix}"] = _flatten_by_series(lower, len(series_ids), horizon)
        rows[f"upper_{suffix}"] = _flatten_by_series(upper, len(series_ids), horizon)
    table = pd.DataFrame(rows)
    try:
        table["timestamp"] = pd.to_datetime(table["timestamp"])
    except (TypeError, ValueError):
        pass
    return TableForecastResult(
        data=table,
        timestamp_col="timestamp",
        prediction_col="mean",
        series_id_col="series_id",
    )


def _flatten_by_series(values: np.ndarray, n_series: int, horizon: int) -> list[float]:
    matrix = _matrix_view(values)
    return [
        float(matrix[step, series_index])
        for series_index in range(n_series)
        for step in range(horizon)
    ]
