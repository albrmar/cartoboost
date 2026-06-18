"""Python interface for the clean-room CartoBoost-inspired regressor."""

from . import graph
from ._native import GraphSageEncoder, HeteroGraphSageEncoder, HinSageEncoder, Node2VecEncoder
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
from .h3 import (
    build_h3_sparse_sets,
    encode_h3_cells,
    h3_parent_id,
    latlng_to_h3_id,
    normalize_h3_id,
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
from .overlay import OverlayConfig, weighted_overlay
from .regressor import CartoBoostRegressor
from .s2 import (
    build_s2_sparse_sets,
    encode_s2_cells,
    latlng_to_s2_id,
    normalize_s2_id,
    s2_parent_id,
)
from .schema import FeatureKind, FeatureSchema
from .standalone import (
    GraphSageLinkPredictor,
    GraphSageStandaloneRegressor,
    HeteroGraphSageLinkPredictor,
    HeteroGraphSageStandaloneRegressor,
    HinSageLinkPredictor,
    HinSageStandaloneRegressor,
    NeuralEmbeddingStandaloneRegressor,
    Node2VecLinkPredictor,
    Node2VecStandaloneRegressor,
)

__version__ = "0.1.21"

__all__ = [
    "ArtifactFallback",
    "GraphSageEncoder",
    "HeteroGraphSageEncoder",
    "HinSageEncoder",
    "Node2VecEncoder",
    "FeatureSchema",
    "graph",
    "NeuralEmbeddingFeatures",
    "NeuralEmbeddingRegressor",
    "NeuralEmbeddingStandaloneRegressor",
    "Node2VecStandaloneRegressor",
    "GraphSageStandaloneRegressor",
    "HeteroGraphSageStandaloneRegressor",
    "HinSageStandaloneRegressor",
    "Node2VecLinkPredictor",
    "GraphSageLinkPredictor",
    "HeteroGraphSageLinkPredictor",
    "HinSageLinkPredictor",
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
    "build_h3_sparse_sets",
    "build_s2_sparse_sets",
    "build_zip_sparse_sets",
    "coerce_geo_to_feature_id",
    "coerce_zip_to_feature_id",
    "encode_h3_cells",
    "encode_s2_cells",
    "h3_parent_id",
    "latlng_to_h3_id",
    "latlng_to_s2_id",
    "normalize_h3_id",
    "normalize_s2_id",
    "out_of_time_split",
    "pinball_loss",
    "residual_morans_i",
    "s2_parent_id",
    "spatial_blocked_cv",
    "temporal_blocked_cv",
    "weighted_overlay",
]
