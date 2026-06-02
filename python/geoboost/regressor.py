from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from ._native import GeoBoostRegressor as _NativeGeoBoostRegressor
except ImportError:  # pragma: no cover - exercised when extension is unavailable
    try:
        from ._geoboost import GeoBoostRegressor as _NativeGeoBoostRegressor
    except ImportError:
        _NativeGeoBoostRegressor = None


class GeoBoostRegressor:
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
        self._validate_params()

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
        del feature_schema, eval_set
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is planned but not implemented in Milestone 1")
        rows = _as_2d_float_list(X)
        targets = _as_1d_float_list(y)
        if len(rows) != len(targets):
            raise ValueError("X and y must contain the same number of rows")

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
            model.fit(rows, targets)
            self._model = model
            self._backend_used = "rust"
            return self

        if self.backend == "rust":
            raise ImportError("geoboost._native is not available; build with maturin first")
        if (
            self.leaf_predictor != "constant"
            or self.fuzzy
            or (self.splitters not in (None, ["axis"]))
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
        model.fit(rows, targets)
        self._model = model
        self._backend_used = "python"
        return self

    def predict(self, X: Iterable[Iterable[float]]) -> list[float]:
        if self._model is None:
            raise RuntimeError("GeoBoostRegressor is not fitted")
        return list(self._model.predict(_as_2d_float_list(X)))

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
                    max_depth=max(1, native_model.max_depth),
                    min_samples_leaf=native_model.min_samples_leaf,
                    min_gain=native_model.min_gain,
                    backend="auto",
                )
                estimator._model = native_model
                estimator._backend_used = "rust"
                return estimator

        payload = json.loads(path.read_text(encoding="utf-8"))
        estimator = cls(**payload["params"])
        estimator._model = _FallbackModel.from_dict(payload["model"])
        estimator._backend_used = "python"
        return estimator

    def _validate_params(self) -> None:
        if int(self.n_estimators) <= 0:
            raise ValueError("n_estimators must be positive")
        learning_rate = float(self.learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if int(self.max_depth) <= 0:
            raise ValueError("max_depth must be positive")
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
        self.stumps: list[dict[str, float | int]] = []

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        self.init_value = sum(y) / len(y)
        prediction = [self.init_value for _ in y]
        self.stumps = []
        for _ in range(self.n_estimators):
            residuals = [target - pred for target, pred in zip(y, prediction, strict=True)]
            stump = _best_stump(X, residuals, self.min_samples_leaf)
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
        model.stumps = list(payload["stumps"])
        return model


def _as_2d_float_list(values: Iterable[Iterable[float]]) -> list[list[float]]:
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


def _as_1d_float_list(values: Iterable[float]) -> list[float]:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("y must not be empty")
    if any(not math.isfinite(value) for value in rows):
        raise ValueError("y must contain only finite values")
    return rows


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
    min_samples_leaf: int,
) -> dict[str, float | int] | None:
    best_loss: float | None = None
    best: dict[str, float | int] | None = None
    for feature in range(len(X[0])):
        thresholds = sorted({row[feature] for row in X})
        for threshold in thresholds:
            left = [res for row, res in zip(X, residuals, strict=True) if row[feature] <= threshold]
            right = [res for row, res in zip(X, residuals, strict=True) if row[feature] > threshold]
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue
            left_value = sum(left) / len(left)
            right_value = sum(right) / len(right)
            loss = _squared_error(left, left_value) + _squared_error(right, right_value)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best = {
                    "feature": feature,
                    "threshold": threshold,
                    "left_value": left_value,
                    "right_value": right_value,
                }
    return best


def _squared_error(values: list[float], center: float) -> float:
    return sum((value - center) ** 2 for value in values)
