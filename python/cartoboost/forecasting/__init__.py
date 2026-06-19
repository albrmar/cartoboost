"""Forecasting API for CartoBoost."""

from .artifacts import ForecastArtifact, ForecastArtifactManifest
from .backtesting import BacktestFoldResult, BacktestResult, RollingOriginBacktester
from .base import BaseForecaster, PanelForecasterMixin, SingleSeriesForecasterMixin
from .config import ForecastingConfig
from .ensemble import WeightedEnsembleForecaster
from .frequency import (
    infer_frequency,
    next_timestamps,
    normalize_frequency,
    validate_horizon,
    validate_regular_frequency,
)
from .global_models import CartoBoostLagForecaster
from .lag_features import (
    CalendarFeatureConfig,
    LagFeatureBuilder,
    LagFeatureConfig,
    RollingFeatureConfig,
)
from .local import (
    AutoARIMAForecaster,
    AutoKalmanForecaster,
    AutoLocalLevelKalmanForecaster,
    ETSForecaster,
    KalmanForecaster,
    KrigingForecaster,
    LocalLevelKalmanForecaster,
    NaiveForecaster,
    OptimizedThetaForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
)
from .metrics import ForecastMetricSet
from .registry import ForecastModelSpec, ForecastRegistry
from .schema import ForecastFrame, ForecastResult, PredictionInterval
from .splitters import (
    ExpandingWindowSplitter,
    ForecastFold,
    RollingOriginSplitter,
    SlidingWindowSplitter,
)

__all__ = [
    "AutoARIMAForecaster",
    "AutoKalmanForecaster",
    "AutoLocalLevelKalmanForecaster",
    "BacktestFoldResult",
    "BacktestResult",
    "BaseForecaster",
    "CalendarFeatureConfig",
    "CartoBoostLagForecaster",
    "ETSForecaster",
    "ExpandingWindowSplitter",
    "ForecastArtifact",
    "ForecastArtifactManifest",
    "ForecastFold",
    "ForecastFrame",
    "ForecastMetricSet",
    "ForecastModelSpec",
    "ForecastRegistry",
    "ForecastResult",
    "ForecastingConfig",
    "infer_frequency",
    "KalmanForecaster",
    "LocalLevelKalmanForecaster",
    "KrigingForecaster",
    "LagFeatureBuilder",
    "LagFeatureConfig",
    "NaiveForecaster",
    "next_timestamps",
    "normalize_frequency",
    "OptimizedThetaForecaster",
    "PanelForecasterMixin",
    "PredictionInterval",
    "RollingFeatureConfig",
    "RollingOriginBacktester",
    "RollingOriginSplitter",
    "SeasonalNaiveForecaster",
    "SingleSeriesForecasterMixin",
    "SlidingWindowSplitter",
    "ThetaForecaster",
    "validate_horizon",
    "validate_regular_frequency",
    "WeightedEnsembleForecaster",
]
