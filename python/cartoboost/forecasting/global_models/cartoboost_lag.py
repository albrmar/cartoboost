from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._native_wrappers import NativeForecastWrapper, _native_class
from ..lag_features import CalendarFeatureConfig, LagFeatureConfig, RollingFeatureConfig


@dataclass(frozen=True)
class ForecastResult:
    """Thin result container for native CartoBoost lag forecast outputs."""

    frame: Any
    predictions: np.ndarray
    feature_names: list[str]
    regressor_metadata: dict[str, Any]


class CartoBoostLagForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust CartoBoost lag forecasting binding."""

    native_class_name = "CartoBoostLagForecaster"

    def __init__(self, **params: Any) -> None:
        params = dict(params)
        self.time_col = params.pop("time_col", None)
        self.target_col = params.pop("target_col", None)
        self.panel_cols = list(params.pop("panel_cols", []))
        self.frequency = params.pop("frequency", params.pop("freq", "D"))
        self.covariate_features = list(params.get("covariate_features", []) or [])
        native_params = self._native_params(params)
        super().__init__(**native_params)

    def _native_params(self, params: dict[str, Any]) -> dict[str, Any]:
        regressor_params = dict(params.pop("regressor_params", {}) or {})
        unsupported_regressor = sorted(
            set(regressor_params)
            - {
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_samples_leaf",
                "min_gain",
                "splitters",
            }
        )
        if unsupported_regressor:
            raise ValueError(
                f"unsupported CartoBoostLagForecaster regressor_params: {unsupported_regressor}"
            )
        params.update(regressor_params)

        lag_config = params.pop("lag_config", None)
        if lag_config is not None:
            if not isinstance(lag_config, LagFeatureConfig):
                raise TypeError("lag_config must be a LagFeatureConfig")
            params.setdefault("lags", list(lag_config.lags))
            params.setdefault("difference_lags", list(lag_config.difference_lags))
            params.setdefault("rolling_trend_windows", list(lag_config.rolling_trend_windows))

        rolling_config = params.pop("rolling_config", None)
        if rolling_config is not None:
            if not isinstance(rolling_config, RollingFeatureConfig):
                raise TypeError("rolling_config must be a RollingFeatureConfig")
            unsupported_aggs = sorted(
                set(rolling_config.aggregations) - {"mean", "std", "min", "max"}
            )
            if unsupported_aggs:
                raise ValueError(
                    "native CartoBoostLagForecaster supports rolling mean/std/min/max only; "
                    f"unsupported: {unsupported_aggs}"
                )
            if rolling_config.include_expanding:
                raise ValueError(
                    "native CartoBoostLagForecaster does not support expanding features"
                )
            if rolling_config.min_periods is not None:
                raise ValueError("native CartoBoostLagForecaster requires complete rolling windows")
            if "mean" in rolling_config.aggregations:
                params.setdefault("rolling_windows", list(rolling_config.windows))
            if "std" in rolling_config.aggregations:
                params.setdefault("rolling_std_windows", list(rolling_config.windows))
            if "min" in rolling_config.aggregations:
                params.setdefault("rolling_min_windows", list(rolling_config.windows))
            if "max" in rolling_config.aggregations:
                params.setdefault("rolling_max_windows", list(rolling_config.windows))

        calendar_config = params.pop("calendar_config", None)
        if calendar_config is not None:
            if not isinstance(calendar_config, CalendarFeatureConfig):
                raise TypeError("calendar_config must be a CalendarFeatureConfig")
            supported = {"dayofweek", "month", "day"}
            requested = set(calendar_config.features)
            unsupported = sorted(requested - supported)
            if unsupported:
                raise ValueError(
                    "native CartoBoostLagForecaster supports calendar features "
                    f"{sorted(supported)}; unsupported: {unsupported}"
                )
            params.setdefault("calendar_features", bool(requested))

        unsupported = sorted(
            set(params)
            - {
                "lags",
                "rolling_windows",
                "rolling_std_windows",
                "rolling_min_windows",
                "rolling_max_windows",
                "ewm_alpha_percents",
                "covariate_features",
                "covariate_indicator_values",
                "covariate_calendar_interactions",
                "difference_lags",
                "rolling_trend_windows",
                "calendar_features",
                "rich_calendar_features",
                "recursive",
                "prediction_interval_levels",
                "trend_features",
                "target_mode",
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_samples_leaf",
                "min_gain",
                "splitters",
            }
        )
        if unsupported:
            raise ValueError(f"unsupported CartoBoostLagForecaster parameters: {unsupported}")
        return params

    def _coerce_fit_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if args and self.time_col is not None and self.target_col is not None:
            frame = self._native_frame_from_dataframe(args[0])
            if frame is not None:
                return (frame, *args[1:])
        return super()._coerce_fit_args(args)

    def _native_frame_from_dataframe(self, value: Any) -> Any | None:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised only in minimal installs.
            raise ImportError(
                "CartoBoostLagForecaster DataFrame input requires pandas. Install pandas to use "
                "time_col/target_col ergonomics."
            ) from exc
        if not isinstance(value, pd.DataFrame):
            return None
        required = [self.time_col, self.target_col, *self.panel_cols, *self.covariate_features]
        missing = [col for col in required if col not in value.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        data = value.sort_values([*self.panel_cols, self.time_col], kind="mergesort")
        if self.panel_cols:
            duplicate_mask = data.duplicated([*self.panel_cols, self.time_col], keep=False)
        else:
            duplicate_mask = data.duplicated([self.time_col], keep=False)
        if duplicate_mask.any():
            raise ValueError(
                "CartoBoostLagForecaster requires unique timestamps within each panel when "
                "coercing a DataFrame to the native ForecastFrame"
            )
        rows = []
        row_covariates = []
        for row in data.itertuples(index=False):
            row_values = dict(zip(data.columns, row, strict=True))
            if self.panel_cols:
                series_id = "|".join(str(row_values[col]) for col in self.panel_cols)
            else:
                series_id = "__single__"
            timestamp = pd.Timestamp(row_values[self.time_col]).strftime("%Y-%m-%dT%H:%M:%S")
            rows.append((series_id, timestamp, float(row_values[self.target_col])))
            row_covariates.append(
                {name: float(row_values[name]) for name in self.covariate_features}
            )
        native_frame_class = _native_class("ForecastFrame")
        if native_frame_class is None:
            raise NotImplementedError("Rust binding for ForecastFrame is not available.")
        return native_frame_class(
            rows,
            self.frequency,
            row_covariates=row_covariates if self.covariate_features else None,
        )


__all__ = ["CartoBoostLagForecaster", "ForecastResult"]
