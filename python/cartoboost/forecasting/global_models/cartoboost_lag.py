from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...regressor import CartoBoostRegressor
from ..frequency import next_timestamps, validate_horizon
from ..lag_features import (
    CalendarFeatureConfig,
    LagFeatureBuilder,
    LagFeatureConfig,
    RollingFeatureConfig,
    _require_pandas,
)
from ..schema import ForecastFrame, PredictionInterval
from ..schema import ForecastResult as TableForecastResult


@dataclass(frozen=True)
class ForecastResult:
    """Forecast output returned by CartoBoost lag forecasters."""

    frame: Any
    predictions: np.ndarray
    feature_names: list[str]
    regressor_metadata: dict[str, Any]


@dataclass
class CartoBoostLagForecaster:
    """Global recursive forecaster backed by :class:`CartoBoostRegressor`."""

    time_col: str | None = None
    target_col: str | None = None
    panel_cols: Sequence[str] = field(default_factory=list)
    lag_config: LagFeatureConfig = field(default_factory=LagFeatureConfig)
    rolling_config: RollingFeatureConfig | None = None
    calendar_config: CalendarFeatureConfig | None = None
    static_cols: Sequence[str] = field(default_factory=list)
    known_future_cols: Sequence[str] = field(default_factory=list)
    historical_covariate_cols: Sequence[str] = field(default_factory=list)
    holiday_fn: Callable[[Any], bool] | None = None
    regressor: CartoBoostRegressor | None = None
    regressor_params: dict[str, Any] | None = None
    lags: Sequence[int] | None = None
    rolling_windows: Sequence[int] | None = None
    calendar_features: bool | Sequence[str] = False
    recursive: bool = True
    prediction_interval_levels: Sequence[float] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.lags is not None:
            self.lag_config = LagFeatureConfig(lags=list(self.lags))
        if self.rolling_windows is not None:
            self.rolling_config = RollingFeatureConfig(
                windows=list(self.rolling_windows),
                aggregations=["mean"],
            )
        if self.calendar_features is True:
            self.calendar_config = CalendarFeatureConfig(
                features=["dayofweek", "month", "quarter", "day"]
            )
        elif self.calendar_features:
            self.calendar_config = CalendarFeatureConfig(features=list(self.calendar_features))
        self.panel_cols = list(self.panel_cols)
        self.static_cols = list(self.static_cols)
        self.known_future_cols = list(self.known_future_cols)
        self.historical_covariate_cols = list(self.historical_covariate_cols)
        self.prediction_interval_levels = tuple(
            float(level) for level in self.prediction_interval_levels
        )
        overlap = set(self.known_future_cols).intersection(self.historical_covariate_cols)
        if overlap:
            raise ValueError(
                "columns cannot be both known-future and historical-only covariates: "
                f"{sorted(overlap)}"
            )

    @property
    def feature_names(self) -> list[str]:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("CartoBoostLagForecaster is not fitted")
        return list(self.feature_names_)

    @property
    def regressor_metadata(self) -> dict[str, Any]:
        if not hasattr(self, "regressor_metadata_"):
            raise RuntimeError("CartoBoostLagForecaster is not fitted")
        return dict(self.regressor_metadata_)

    def fit(self, frame: Any, sample_weight: Any | None = None) -> CartoBoostLagForecaster:
        pd = _require_pandas()
        if isinstance(frame, ForecastFrame):
            self.forecast_frame_metadata_ = frame.to_metadata()
            self.time_col = frame.timestamp_col
            self.target_col = frame.target_col
            self.panel_cols = [] if frame.series_id_col is None else [frame.series_id_col]
            self.static_cols = list(frame.static_covariates)
            self.known_future_cols = list(frame.known_future_covariates)
            self.historical_covariate_cols = list(frame.historical_covariates)
            self.forecast_freq_ = frame.freq
            data = frame.to_pandas()
        else:
            self.forecast_frame_metadata_ = None
            self.forecast_freq_ = None
            data = pd.DataFrame(frame).copy()
        if self.time_col is None or self.target_col is None:
            raise ValueError("time_col and target_col are required unless fitting ForecastFrame")
        builder = self._new_builder()
        feature_frame = builder.fit_transform(data, drop_missing=True)
        if feature_frame.empty:
            raise ValueError("not enough history to build lag features")
        y = feature_frame[self.target_col].to_numpy(dtype=float)
        x = feature_frame[builder.feature_names]
        model = self.regressor or CartoBoostRegressor(**(self.regressor_params or {}))
        model.fit(x, y, sample_weight=sample_weight)
        training_predictions = np.asarray(model.predict(x), dtype=float)

        self.builder_ = builder
        self.regressor_ = model
        self.history_ = builder._sorted(data).drop(
            columns=["__cartoboost_single_panel__"],
            errors="ignore",
        )
        self.feature_names_ = builder.feature_names
        self.training_rows_ = int(len(feature_frame))
        self.training_residuals_ = y - training_predictions
        self.regressor_metadata_ = self._metadata(model)
        self.is_fitted_ = True
        return self

    def predict(
        self,
        future_frame: Any | None = None,
        *,
        horizon: int | None = None,
    ) -> ForecastResult | TableForecastResult:
        self._check_fitted()
        if isinstance(future_frame, int):
            horizon = future_frame
            future_frame = None
        if horizon is not None or future_frame is None:
            horizon = validate_horizon(horizon or 1)
            future_frame = self._future_frame_for_horizon(horizon)
            low_level_result = self._predict_frame(future_frame)
            return self._to_table_result(low_level_result, horizon)
        if isinstance(future_frame, ForecastFrame):
            future_frame = future_frame.to_pandas()
        return self._predict_frame(future_frame)

    def _predict_frame(self, future_frame: Any) -> ForecastResult:
        pd = _require_pandas()
        future = pd.DataFrame(future_frame).copy()
        self._validate_future_frame(future)

        future["__cartoboost_original_order__"] = np.arange(len(future), dtype=int)
        sorted_future = future.sort_values(
            [self.time_col, *self.panel_cols, "__cartoboost_original_order__"],
            kind="mergesort",
        )
        history = self.history_.copy()
        predictions_by_order: dict[int, float] = {}
        feature_rows = []
        for _, row in sorted_future.iterrows():
            features = self.builder_.transform_future_row(history, row)
            prediction = float(
                self.regressor_.predict(features.to_frame().T[self.feature_names_])[0]
            )
            original_order = int(row["__cartoboost_original_order__"])
            predictions_by_order[original_order] = prediction
            feature_rows.append(features)

            history_row = row.drop(labels=["__cartoboost_original_order__"]).copy()
            history_row[self.target_col] = prediction
            history = pd.concat([history, history_row.to_frame().T], ignore_index=True)

        output = future.drop(columns=["__cartoboost_original_order__"]).copy()
        output["forecast"] = [predictions_by_order[i] for i in range(len(output))]
        predictions = output["forecast"].to_numpy(dtype=float)
        result = ForecastResult(
            frame=output,
            predictions=predictions,
            feature_names=list(self.feature_names_),
            regressor_metadata=dict(self.regressor_metadata_),
        )
        self.last_prediction_features_ = pd.DataFrame(feature_rows, columns=self.feature_names_)
        return result

    def _future_frame_for_horizon(self, horizon: int) -> Any:
        pd = _require_pandas()
        if self.forecast_freq_ is None:
            raise ValueError("predict(horizon) requires fitted ForecastFrame with a regular freq")
        rows = []
        history = self.history_.copy()
        if self.panel_cols:
            for key, group in history.groupby(self.panel_cols, sort=False, dropna=False):
                key_values = key if isinstance(key, tuple) else (key,)
                timestamps = next_timestamps(
                    group[self.time_col].max(),
                    horizon,
                    self.forecast_freq_,
                )
                base = {col: value for col, value in zip(self.panel_cols, key_values, strict=True)}
                for timestamp in timestamps:
                    row = dict(base)
                    row[self.time_col] = timestamp
                    for col in self.static_cols:
                        row[col] = group[col].iloc[-1]
                    rows.append(row)
        else:
            timestamps = next_timestamps(history[self.time_col].max(), horizon, self.forecast_freq_)
            for timestamp in timestamps:
                row = {self.time_col: timestamp}
                for col in self.static_cols:
                    row[col] = history[col].iloc[-1]
                rows.append(row)
        return pd.DataFrame(rows)

    def _to_table_result(self, result: ForecastResult, horizon: int) -> TableForecastResult:
        frame = result.frame.copy()
        if self.panel_cols:
            frame["series_id"] = frame[self.panel_cols].astype(str).agg("|".join, axis=1)
        else:
            frame["series_id"] = "__single__"
        frame = frame.sort_values(["series_id", self.time_col], kind="mergesort").reset_index(
            drop=True
        )
        frame["horizon"] = frame.groupby("series_id").cumcount() + 1
        table = frame[["series_id", self.time_col, "horizon"]].rename(
            columns={self.time_col: "timestamp"}
        )
        table["model"] = self.__class__.__name__
        table["mean"] = frame["forecast"].astype(float)
        residual_scale = float(np.std(getattr(self, "training_residuals_", np.array([0.0]))))
        for level in sorted(self.prediction_interval_levels):
            suffix = PredictionInterval(level, [0.0], [0.0]).suffix
            width = residual_scale * float(level)
            table[f"lower_{suffix}"] = table["mean"] - width
            table[f"upper_{suffix}"] = table["mean"] + width
        return TableForecastResult(
            data=table,
            timestamp_col="timestamp",
            prediction_col="mean",
            series_id_col="series_id",
        )

    def _new_builder(self) -> LagFeatureBuilder:
        return LagFeatureBuilder(
            time_col=self.time_col,
            target_col=self.target_col,
            panel_cols=self.panel_cols,
            lag_config=self.lag_config,
            rolling_config=self.rolling_config,
            calendar_config=self.calendar_config,
            static_cols=self.static_cols,
            known_future_cols=self.known_future_cols,
            holiday_fn=self.holiday_fn,
        )

    def _validate_future_frame(self, future: Any) -> None:
        missing = [
            col
            for col in [self.time_col, *self.panel_cols, *self.known_future_cols]
            if col not in future.columns
        ]
        if missing:
            raise ValueError(f"missing required future columns: {missing}")
        forbidden = [col for col in self.historical_covariate_cols if col in future.columns]
        if forbidden:
            raise ValueError(
                "future covariates include historical-only columns that are not known at "
                f"prediction time: {forbidden}"
            )

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("CartoBoostLagForecaster is not fitted")

    @staticmethod
    def _metadata(model: CartoBoostRegressor) -> dict[str, Any]:
        metadata = {
            "backend": getattr(model, "_backend_used", None),
            "n_features_in": getattr(model, "n_features_in_", None),
        }
        if hasattr(model, "feature_names_in_"):
            metadata["feature_names_in"] = [str(name) for name in model.feature_names_in_]
        if hasattr(model, "metadata_"):
            metadata["native"] = model.metadata_
        if hasattr(model, "training_config_"):
            metadata["training_config"] = model.training_config_
        return metadata
