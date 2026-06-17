"""Python interface for the clean-room CartoBoost-inspired regressor."""

from .evaluation import (
    grouped_blocked_cv,
    out_of_time_split,
    spatial_blocked_cv,
    temporal_blocked_cv,
)
from .explain import explain_shap, make_shap_explainer
from .geo import (
    build_geo_sparse_sets,
    build_zip_sparse_sets,
    coerce_geo_to_feature_id,
    coerce_zip_to_feature_id,
)
from .metrics import (
    calibrated_intervals,
    conformal_residual_quantile,
    interval_coverage,
    jitter_volatility,
    mean_interval_width,
    pinball_loss,
    residual_morans_i,
)
from .neural import (
    ArtifactFallback,
    NeuralEmbeddingFeatures,
    NeuralEmbeddingRegressor,
    benchmark_neural_vs_cartoboost,
)
from ._native import GraphSageEncoder, HeteroGraphSageEncoder
from . import graph
from .overlay import OverlayConfig, weighted_overlay
from .regressor import CartoBoostRegressor
from .schema import FeatureKind, FeatureSchema

__version__ = "0.1.14"

__all__ = [
    "ArtifactFallback",
    "GraphSageEncoder",
    "HeteroGraphSageEncoder",
    "FeatureSchema",
    "graph",
    "NeuralEmbeddingFeatures",
    "NeuralEmbeddingRegressor",
    "benchmark_neural_vs_cartoboost",
    "FeatureKind",
    "CartoBoostRegressor",
    "OverlayConfig",
    "__version__",
    "calibrated_intervals",
    "conformal_residual_quantile",
    "explain_shap",
    "grouped_blocked_cv",
    "interval_coverage",
    "jitter_volatility",
    "make_shap_explainer",
    "mean_interval_width",
    "build_geo_sparse_sets",
    "build_zip_sparse_sets",
    "coerce_geo_to_feature_id",
    "coerce_zip_to_feature_id",
    "out_of_time_split",
    "pinball_loss",
    "residual_morans_i",
    "spatial_blocked_cv",
    "temporal_blocked_cv",
    "weighted_overlay",
]
