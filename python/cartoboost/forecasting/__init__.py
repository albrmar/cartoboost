"""Forecasting API for CartoBoost."""

from .artifacts import ForecastArtifact, ForecastArtifactManifest
from .backtesting import BacktestFoldResult, BacktestResult, RollingOriginBacktester
from .base import BaseForecaster, PanelForecasterMixin, SingleSeriesForecasterMixin
from .config import ForecastingConfig
from .ensemble import WeightedEnsembleForecaster
from .global_models import CartoBoostLagForecaster
from .lag_features import (
    CalendarFeatureConfig,
    LagFeatureBuilder,
    LagFeatureConfig,
    RollingFeatureConfig,
)
from .local import (
    AutoARIMAForecaster,
    ETSForecaster,
    KalmanForecaster,
    KrigingForecaster,
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
    "KalmanForecaster",
    "KrigingForecaster",
    "LagFeatureBuilder",
    "LagFeatureConfig",
    "NaiveForecaster",
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
    "WeightedEnsembleForecaster",
]
