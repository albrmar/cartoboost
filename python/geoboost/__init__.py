"""Python interface for the clean-room GeoBoost-inspired regressor."""

from .explain import explain_shap, make_shap_explainer
from .geo import build_zip_sparse_sets, coerce_zip_to_feature_id
from .regressor import GeoBoostRegressor
from .schema import FeatureSchema

__version__ = "0.1.0"

__all__ = [
    "FeatureSchema",
    "GeoBoostRegressor",
    "build_zip_sparse_sets",
    "coerce_zip_to_feature_id",
    "__version__",
    "explain_shap",
    "make_shap_explainer",
]
