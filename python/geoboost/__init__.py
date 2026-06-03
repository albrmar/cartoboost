"""Python interface for the clean-room GeoBoost-inspired regressor."""

from .regressor import GeoBoostRegressor
from .schema import FeatureSchema

__version__ = "0.1.0"

__all__ = ["FeatureSchema", "GeoBoostRegressor", "__version__"]
