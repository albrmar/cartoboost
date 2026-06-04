"""Python interface for the clean-room GeoBoost-inspired regressor."""

from .evaluation import (
    grouped_blocked_cv,
    out_of_time_split,
    spatial_blocked_cv,
    temporal_blocked_cv,
)
from .explain import explain_shap, make_shap_explainer
from .metrics import (
    calibrated_intervals,
    conformal_residual_quantile,
    interval_coverage,
    jitter_volatility,
    mean_interval_width,
    pinball_loss,
    residual_morans_i,
)
from .regressor import GeoBoostRegressor
from .schema import FeatureSchema

__version__ = "0.1.0"

__all__ = [
    "FeatureSchema",
    "GeoBoostRegressor",
    "__version__",
    "calibrated_intervals",
    "conformal_residual_quantile",
    "explain_shap",
    "grouped_blocked_cv",
    "interval_coverage",
    "jitter_volatility",
    "make_shap_explainer",
    "mean_interval_width",
    "out_of_time_split",
    "pinball_loss",
    "residual_morans_i",
    "spatial_blocked_cv",
    "temporal_blocked_cv",
]
