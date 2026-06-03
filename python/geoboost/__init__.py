"""Python interface for the clean-room GeoBoost-inspired regressor."""

from .explain import explain_shap, make_shap_explainer
from .regressor import GeoBoostRegressor
from .schema import FeatureSchema

__version__ = "0.1.0"

__all__ = [
    "FeatureSchema",
    "GeoBoostRegressor",
    "__version__",
    "explain_shap",
    "make_shap_explainer",
]
