from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from ._native import GeoBoostRegressor as _NativeGeoBoostRegressor
except ImportError:  # pragma: no cover - exercised when extension is unavailable
    try:
        from ._geoboost import GeoBoostRegressor as _NativeGeoBoostRegressor
    except ImportError:
        _NativeGeoBoostRegressor = None


_VALID_SPLITTERS = {
    "axis",
    "diagonal_2d",
    "diagonal2d",
    "gaussian_2d",
    "gaussian2d",
    "radial",
    "periodic_time",
    "periodic_24",
    "sparse_set",
    "sparse",
}


class GeoBoostRegressor(RegressorMixin, BaseEstimator):
    """Small sklearn-style gradient boosted stump regressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        min_gain: float = 1e-8,
        loss: str = "l2",
        splitters: list[str] | None = None,
        leaf_predictor: str = "constant",
        linear_leaf_features: list[str] | None = None,
        fuzzy: bool = False,
        fuzzy_bandwidth: float = 0.0,
        l2_regularization: float = 1.0,
        random_state: int | None = None,
        n_threads: int | None = None,
        backend: str = "auto",
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.loss = loss
        self.splitters = splitters
        self.leaf_predictor = leaf_predictor
        self.linear_leaf_features = linear_leaf_features
        self.fuzzy = fuzzy
        self.fuzzy_bandwidth = fuzzy_bandwidth
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.n_threads = n_threads
        self.backend = backend
        self._model: Any | None = None
        self._backend_used: str | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_gain": self.min_gain,
            "loss": self.loss,
            "splitters": self.splitters,
            "leaf_predictor": self.leaf_predictor,
            "linear_leaf_features": self.linear_leaf_features,
            "fuzzy": self.fuzzy,
            "fuzzy_bandwidth": self.fuzzy_bandwidth,
            "l2_regularization": self.l2_regularization,
            "random_state": self.random_state,
            "n_threads": self.n_threads,
            "backend": self.backend,
        }

    def set_params(self, **params: Any) -> GeoBoostRegressor:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        self._validate_params()
        self._model = None
        self._backend_used = None
        return self

    def fit(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        sample_weight: Iterable[float] | None = None,
        feature_schema: Any | None = None,
        eval_set: Any | None = None,
    ) -> GeoBoostRegressor:
        del eval_set
        self._validate_params()
        feature_names = _feature_names(X)
        rows = _as_2d_float_list(X)
        targets = _as_1d_float_list(y)
        if len(rows) != len(targets):
            raise ValueError("X and y must contain the same number of rows")
        weights = _as_sample_weight_list(sample_weight, len(targets))
        self.n_features_in_ = len(rows[0])
        self.feature_schema_ = _feature_schema_metadata(feature_schema)
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)

        native_cls = _NativeGeoBoostRegressor
        if self.backend in {"auto", "rust"} and native_cls is not None:
            model = native_cls(
                n_estimators=int(self.n_estimators),
                learning_rate=float(self.learning_rate),
                max_depth=int(self.max_depth),
                min_samples_leaf=int(self.min_samples_leaf),
                min_gain=float(self.min_gain),
                splitters=list(self.splitters or ["axis"]),
                leaf_predictor=str(self.leaf_predictor),
                linear_leaf_features=_resolve_linear_leaf_features(
                    self.linear_leaf_features,
                    len(rows[0]),
                ),
                l2_regularization=float(self.l2_regularization),
                fuzzy=bool(self.fuzzy),
                fuzzy_bandwidth=float(self.fuzzy_bandwidth),
            )
            try:
                _fit_native(model, rows, targets, weights)
            except NotImplementedError:
                if self.backend == "rust" or not self._python_fallback_supported():
                    raise
            else:
                self._model = model
                self._backend_used = "rust"
                self.is_fitted_ = True
                return self

        if self.backend == "rust":
            raise ImportError("geoboost._native is not available; build with maturin first")
        if self.leaf_predictor != "constant" or (
            int(self.max_depth) != 0 and (self.fuzzy or not _is_axis_splitters(self.splitters))
        ):
            raise NotImplementedError(
                "the pure-Python fallback supports only axis splits with constant leaves"
            )

        model = _FallbackModel(
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            max_depth=int(self.max_depth),
            min_samples_leaf=int(self.min_samples_leaf),
            min_gain=float(self.min_gain),
        )
        model.fit(rows, targets, weights)
        self._model = model
        self._backend_used = "python"
        self.is_fitted_ = True
        return self

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GeoBoostRegressor is not fitted")
        rows = _as_2d_float_list(X)
        if hasattr(self, "n_features_in_") and len(rows[0]) != self.n_features_in_:
            raise ValueError(
                f"X has {len(rows[0])} features, but GeoBoostRegressor was fitted with "
                f"{self.n_features_in_} features"
            )
        return np.asarray(list(self._model.predict(rows)), dtype=float)

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("GeoBoostRegressor is not fitted")
        path = Path(path)
        if self._backend_used == "rust" and hasattr(self._model, "save"):
            self._model.save(path)
            return

        payload = {
            "params": self.get_params(),
            "model": self._model.to_dict(),
            "backend": "python",
            "feature_schema": getattr(self, "feature_schema_", None),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GeoBoostRegressor:
        path = Path(path)
        native_cls = _NativeGeoBoostRegressor
        if native_cls is not None:
            try:
                native_model = native_cls.load(path)
            except ValueError:
                pass
            else:
                estimator = cls(
                    n_estimators=native_model.n_estimators,
                    learning_rate=native_model.learning_rate,
                    max_depth=native_model.max_depth,
                    min_samples_leaf=native_model.min_samples_leaf,
                    min_gain=native_model.min_gain,
                    backend="auto",
                )
                estimator._model = native_model
                estimator._backend_used = "rust"
                estimator.n_features_in_ = native_model.feature_count
                estimator.feature_schema_ = None
                estimator.is_fitted_ = True
                return estimator

        payload = json.loads(path.read_text(encoding="utf-8"))
        estimator = cls(**payload["params"])
        estimator._model = _FallbackModel.from_dict(payload["model"])
        estimator._backend_used = "python"
        estimator.n_features_in_ = payload["model"].get("feature_count", 0) or 1
        estimator.feature_schema_ = payload.get("feature_schema")
        estimator.is_fitted_ = True
        return estimator

    def _validate_params(self) -> None:
        if int(self.n_estimators) <= 0:
            raise ValueError("n_estimators must be positive")
        learning_rate = float(self.learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if int(self.max_depth) < 0:
            raise ValueError("max_depth must be non-negative")
        if int(self.min_samples_leaf) <= 0:
            raise ValueError("min_samples_leaf must be positive")
        min_gain = float(self.min_gain)
        if not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError("min_gain must be finite and non-negative")
        if self.loss != "l2":
            raise NotImplementedError("Milestone 1 supports only loss='l2'")
        if self.leaf_predictor not in {"constant", "linear"}:
            raise ValueError("leaf_predictor must be 'constant' or 'linear'")
        if float(self.l2_regularization) < 0 or not math.isfinite(float(self.l2_regularization)):
            raise ValueError("l2_regularization must be finite and non-negative")
        if float(self.fuzzy_bandwidth) < 0 or not math.isfinite(float(self.fuzzy_bandwidth)):
            raise ValueError("fuzzy_bandwidth must be finite and non-negative")
        if self.backend not in {"auto", "rust", "python"}:
            raise ValueError("backend must be one of 'auto', 'rust', or 'python'")
        self._validate_splitters()

    def _validate_splitters(self) -> None:
        if self.splitters is None:
            return
        if isinstance(self.splitters, str):
            raise ValueError("splitters must be a list of splitter names")
        try:
            splitters = list(self.splitters)
        except TypeError as exc:
            raise ValueError("splitters must be a list of splitter names") from exc
        unknown = [splitter for splitter in splitters if splitter not in _VALID_SPLITTERS]
        if unknown:
            raise ValueError(f"unknown splitter(s): {unknown}")

    def _python_fallback_supported(self) -> bool:
        if self.leaf_predictor != "constant":
            return False
        if int(self.max_depth) == 0:
            return True
        return not self.fuzzy and _is_axis_splitters(self.splitters)


class _FallbackModel:
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        max_depth: int,
        min_samples_leaf: int,
        min_gain: float,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.init_value = 0.0
        self.feature_count = 0
        self.stumps: list[dict[str, float | int]] = []

    def fit(
        self,
        X: list[list[float]],
        y: list[float],
        sample_weight: list[float] | None = None,
    ) -> None:
        self.feature_count = len(X[0])
        weights = sample_weight or [1.0 for _ in y]
        self.init_value = _weighted_mean(y, weights)
        prediction = [self.init_value for _ in y]
        self.stumps = []
        if self.max_depth == 0:
            return
        for _ in range(self.n_estimators):
            residuals = [target - pred for target, pred in zip(y, prediction, strict=True)]
            stump = _best_stump(X, residuals, weights, self.min_samples_leaf)
            if stump is None:
                break
            for row_index, row in enumerate(X):
                feature = int(stump["feature"])
                value = (
                    stump["left_value"]
                    if row[feature] <= stump["threshold"]
                    else stump["right_value"]
                )
                prediction[row_index] += self.learning_rate * float(value)
            self.stumps.append(stump)

    def predict(self, X: list[list[float]]) -> list[float]:
        predictions = []
        for row in X:
            pred = self.init_value
            for stump in self.stumps:
                feature = int(stump["feature"])
                value = (
                    stump["left_value"]
                    if row[feature] <= stump["threshold"]
                    else stump["right_value"]
                )
                pred += self.learning_rate * float(value)
            predictions.append(pred)
        return predictions

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_gain": self.min_gain,
            "feature_count": self.feature_count,
            "init_value": self.init_value,
            "stumps": self.stumps,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> _FallbackModel:
        model = cls(
            n_estimators=int(payload["n_estimators"]),
            learning_rate=float(payload["learning_rate"]),
            max_depth=int(payload.get("max_depth", 1)),
            min_samples_leaf=int(payload["min_samples_leaf"]),
            min_gain=float(payload.get("min_gain", 0.0)),
        )
        model.init_value = float(payload["init_value"])
        model.feature_count = int(payload.get("feature_count", 0))
        model.stumps = list(payload["stumps"])
        return model


def _as_2d_float_list(values: Iterable[Iterable[float]]) -> list[list[float]]:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    rows = [[float(value) for value in row] for row in values]
    if not rows:
        raise ValueError("X must not be empty")
    width = len(rows[0])
    if width == 0:
        raise ValueError("X rows must contain at least one feature")
    if any(len(row) != width for row in rows):
        raise ValueError("X must be a rectangular 2D array")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("X must contain only finite values")
    return rows


def _feature_names(values: Any) -> list[str] | None:
    columns = getattr(values, "columns", None)
    if columns is None:
        return None
    return [str(column) for column in columns]


def _as_1d_float_list(values: Iterable[float]) -> list[float]:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("y must not be empty")
    if any(not math.isfinite(value) for value in rows):
        raise ValueError("y must contain only finite values")
    return rows


def _as_sample_weight_list(values: Iterable[float] | None, expected: int) -> list[float] | None:
    if values is None:
        return None
    weights = [float(value) for value in values]
    if len(weights) != expected:
        raise ValueError("sample_weight length must match y")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("sample_weight must contain only finite non-negative values")
    return weights


def _is_axis_splitters(splitters: Any) -> bool:
    return splitters is None or list(splitters) == ["axis"]


def _feature_schema_metadata(feature_schema: Any | None) -> Any | None:
    if feature_schema is None:
        return None
    if isinstance(feature_schema, str | int | float | bool):
        return feature_schema
    if isinstance(feature_schema, dict):
        return {str(key): _feature_schema_metadata(value) for key, value in feature_schema.items()}
    if isinstance(feature_schema, list | tuple):
        return [_feature_schema_metadata(value) for value in feature_schema]
    return {
        "type": type(feature_schema).__name__,
        "repr": repr(feature_schema),
    }


def _fit_native(
    model: Any,
    rows: list[list[float]],
    targets: list[float],
    sample_weight: list[float] | None,
) -> None:
    if sample_weight is None:
        model.fit(rows, targets)
        return
    try:
        model.fit(rows, targets, sample_weight)
    except TypeError as exc:
        raise NotImplementedError(
            "the native backend does not support sample_weight in this build"
        ) from exc


def _resolve_linear_leaf_features(features: list[str] | None, width: int) -> list[int] | None:
    if features is None:
        return None
    resolved: list[int] = []
    for feature in features:
        try:
            index = int(feature)
        except ValueError as exc:
            raise ValueError(
                "linear_leaf_features currently expects stringified integer feature indices"
            ) from exc
        if index < 0 or index >= width:
            raise ValueError(f"linear leaf feature index {index} is out of bounds")
        resolved.append(index)
    return resolved


def _best_stump(
    X: list[list[float]],
    residuals: list[float],
    weights: list[float],
    min_samples_leaf: int,
) -> dict[str, float | int] | None:
    best_loss: float | None = None
    best: dict[str, float | int] | None = None
    for feature in range(len(X[0])):
        thresholds = sorted({row[feature] for row in X})
        for threshold in thresholds:
            left_indices = [idx for idx, row in enumerate(X) if row[feature] <= threshold]
            right_indices = [idx for idx, row in enumerate(X) if row[feature] > threshold]
            left = [residuals[idx] for idx in left_indices]
            right = [residuals[idx] for idx in right_indices]
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue
            left_weights = [weights[idx] for idx in left_indices]
            right_weights = [weights[idx] for idx in right_indices]
            left_value = _weighted_mean(left, left_weights)
            right_value = _weighted_mean(right, right_weights)
            loss = _squared_error(left, left_weights, left_value) + _squared_error(
                right,
                right_weights,
                right_value,
            )
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best = {
                    "feature": feature,
                    "threshold": threshold,
                    "left_value": left_value,
                    "right_value": right_value,
                }
    return best


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / weight_sum


def _squared_error(values: list[float], weights: list[float], center: float) -> float:
    return sum(
        weight * (value - center) ** 2 for value, weight in zip(values, weights, strict=True)
    )
