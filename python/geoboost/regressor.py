from __future__ import annotations

import json
import math
from collections.abc import Iterable
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .schema import normalize_feature_kind

try:
    from ._native import GeoBoostRegressor as _NativeGeoBoostRegressor
except ImportError:  # pragma: no cover - exercised when extension is unavailable
    try:
        from ._geoboost import GeoBoostRegressor as _NativeGeoBoostRegressor
    except ImportError:
        _NativeGeoBoostRegressor = None


_VALID_SPLITTERS = {
    "axis",
    "axis_histogram",
    "axis_hist",
    "histogram",
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
        sparse_sets: Any | None = None,
        eval_set: Any | None = None,
    ) -> GeoBoostRegressor:
        del eval_set
        self._validate_params()
        feature_names = _feature_names(X)
        dense_array = _as_2d_float_array(X)
        targets_array = _as_1d_float_array(y)
        if dense_array.shape[0] != targets_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        weights_array = _as_sample_weight_array(sample_weight, targets_array.shape[0])
        sparse_columns, sparse_names = _normalize_sparse_sets(sparse_sets, targets_array.shape[0])
        sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_columns)
        schema_json = _rust_feature_schema_json(feature_schema, dense_array.shape[1], sparse_names)
        schema_metadata = _feature_schema_metadata(feature_schema)
        self.n_features_in_ = dense_array.shape[1]
        self.n_sparse_sets_in_ = len(sparse_columns)
        self.sparse_set_names_ = sparse_names
        self.feature_schema_ = schema_metadata
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
                    dense_array.shape[1],
                ),
                l2_regularization=float(self.l2_regularization),
                fuzzy=bool(self.fuzzy),
                fuzzy_bandwidth=float(self.fuzzy_bandwidth),
            )
            try:
                _fit_native(
                    model,
                    dense_array,
                    targets_array,
                    weights_array,
                    sparse_columns,
                    sparse_offsets,
                    sparse_ids,
                    schema_json,
                )
            except NotImplementedError:
                if self.backend == "rust" or not self._python_fallback_supported():
                    raise
            else:
                self._model = model
                self._backend_used = "rust"
                self.feature_schema_ = (
                    json.loads(schema_json) if schema_json is not None else schema_metadata
                )
                self.metadata_ = _json_attr(model, "metadata_json")
                self.training_config_ = _json_attr(model, "training_config_json")
                self.requires_sparse_sets_ = bool(
                    getattr(model, "requires_sparse_sets", bool(sparse_columns))
                )
                self.is_fitted_ = True
                return self

        if self.backend == "rust":
            raise ImportError("geoboost._native is not available; build with maturin first")
        if sparse_columns:
            raise NotImplementedError("the pure-Python fallback does not support sparse_sets")
        rows = dense_array.tolist()
        targets = targets_array.tolist()
        weights = None if weights_array is None else weights_array.tolist()
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
        self.metadata_ = None
        self.training_config_ = None
        self.requires_sparse_sets_ = False
        self.is_fitted_ = True
        return self

    def predict(self, X: Iterable[Iterable[float]], sparse_sets: Any | None = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GeoBoostRegressor is not fitted")
        dense_array = _as_2d_float_array(X)
        if hasattr(self, "n_features_in_") and dense_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {dense_array.shape[1]} features, but GeoBoostRegressor was fitted with "
                f"{self.n_features_in_} features"
            )
        sparse_columns, sparse_names = _normalize_sparse_sets(
            sparse_sets,
            dense_array.shape[0],
            getattr(self, "sparse_set_names_", None),
        )
        sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_columns)
        expected_sparse_count = getattr(self, "n_sparse_sets_in_", 0)
        if sparse_columns and len(sparse_columns) != expected_sparse_count:
            raise ValueError(
                f"sparse_sets has {len(sparse_columns)} columns, but GeoBoostRegressor was "
                f"fitted with {expected_sparse_count}"
            )
        if (
            isinstance(sparse_sets, dict)
            and sparse_names
            and hasattr(self, "sparse_set_names_")
            and sparse_names != self.sparse_set_names_
        ):
            raise ValueError(
                f"sparse_sets columns {sparse_names!r} do not match fitted columns "
                f"{self.sparse_set_names_!r}"
            )
        if self._backend_used == "rust":
            if not sparse_columns and getattr(self, "requires_sparse_sets_", False):
                raise ValueError(
                    "sparse_sets are required for prediction with this sparse-list model"
                )
            try:
                return np.asarray(
                    self._model.predict_arrays(dense_array, sparse_offsets, sparse_ids),
                    dtype=float,
                )
            except TypeError:
                rows = dense_array.tolist()
                return np.asarray(list(self._model.predict(rows, sparse_columns)), dtype=float)
        if sparse_columns:
            raise NotImplementedError("the pure-Python fallback does not support sparse_sets")
        rows = dense_array.tolist()
        return np.asarray(list(self._model.predict(rows)), dtype=float)

    def __call__(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Make the estimator directly usable as a SHAP model callable."""
        return self.predict(X)

    def make_shap_explainer(
        self,
        background: Any,
        *,
        sparse_sets: Any | None = None,
        sparse_id_vocabulary: dict[str, list[int]] | None = None,
        algorithm: str = "auto",
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Build a SHAP explainer for dense predictions."""
        from .explain import make_shap_explainer

        return make_shap_explainer(
            self,
            background,
            sparse_sets=sparse_sets,
            sparse_id_vocabulary=sparse_id_vocabulary,
            algorithm=algorithm,
            feature_names=feature_names,
            **kwargs,
        )

    def explain_shap(
        self,
        X: Any,
        *,
        background: Any,
        sparse_sets: Any | None = None,
        background_sparse_sets: Any | None = None,
        sparse_id_vocabulary: dict[str, list[int]] | None = None,
        algorithm: str = "auto",
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a SHAP Explanation for dense predictions."""
        from .explain import explain_shap

        return explain_shap(
            self,
            X,
            background=background,
            sparse_sets=sparse_sets,
            background_sparse_sets=background_sparse_sets,
            sparse_id_vocabulary=sparse_id_vocabulary,
            algorithm=algorithm,
            feature_names=feature_names,
            **kwargs,
        )

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
        native_error: ValueError | None = None
        if native_cls is not None:
            try:
                native_model = native_cls.load(path)
            except ValueError as exc:
                native_error = exc
            else:
                estimator = cls(
                    n_estimators=native_model.n_estimators,
                    learning_rate=native_model.learning_rate,
                    max_depth=native_model.max_depth,
                    min_samples_leaf=native_model.min_samples_leaf,
                    min_gain=native_model.min_gain,
                    splitters=list(getattr(native_model, "splitters", ["axis"])),
                    leaf_predictor=str(getattr(native_model, "leaf_predictor", "constant")),
                    linear_leaf_features=[
                        str(feature)
                        for feature in getattr(native_model, "linear_leaf_features", [])
                    ],
                    fuzzy=bool(getattr(native_model, "fuzzy", False)),
                    fuzzy_bandwidth=float(getattr(native_model, "fuzzy_bandwidth", 0.0)),
                    l2_regularization=float(getattr(native_model, "l2_regularization", 1.0)),
                    backend="auto",
                )
                estimator._model = native_model
                estimator._backend_used = "rust"
                estimator.n_features_in_ = native_model.feature_count
                estimator.feature_schema_ = _json_attr(native_model, "feature_schema_json")
                estimator.sparse_set_names_ = _sparse_names_from_feature_schema(
                    estimator.feature_schema_
                )
                estimator.n_sparse_sets_in_ = len(estimator.sparse_set_names_)
                estimator.metadata_ = _json_attr(native_model, "metadata_json")
                estimator.training_config_ = _json_attr(native_model, "training_config_json")
                estimator.requires_sparse_sets_ = bool(
                    getattr(native_model, "requires_sparse_sets", False)
                )
                estimator.is_fitted_ = True
                return estimator

        payload = json.loads(path.read_text(encoding="utf-8"))
        if native_error is not None and _looks_like_native_artifact(payload):
            raise native_error
        estimator = cls(**payload["params"])
        estimator._model = _FallbackModel.from_dict(payload["model"])
        estimator._backend_used = "python"
        estimator.n_features_in_ = payload["model"].get("feature_count", 0) or 1
        estimator.n_sparse_sets_in_ = 0
        estimator.sparse_set_names_ = []
        estimator.feature_schema_ = payload.get("feature_schema")
        estimator.metadata_ = None
        estimator.training_config_ = None
        estimator.requires_sparse_sets_ = False
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
        unknown = [splitter for splitter in splitters if not _is_valid_splitter_name(splitter)]
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
    return _as_2d_float_array(values).tolist()


def _as_2d_float_array(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    try:
        array = np.asarray(values, dtype=np.float64, order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a rectangular 2D array") from exc
    if array.ndim != 2:
        raise ValueError("X must be a rectangular 2D array")
    if array.shape[0] == 0:
        raise ValueError("X must not be empty")
    if array.shape[1] == 0:
        raise ValueError("X rows must contain at least one feature")
    if not np.all(np.isfinite(array)):
        raise ValueError("X must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _feature_names(values: Any) -> list[str] | None:
    columns = getattr(values, "columns", None)
    if columns is None:
        return None
    return [str(column) for column in columns]


def _as_1d_float_list(values: Iterable[float]) -> list[float]:
    return _as_1d_float_array(values).tolist()


def _as_1d_float_array(values: Any) -> np.ndarray:
    try:
        rows = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must be a 1D numeric array") from exc
    if rows.ndim != 1:
        raise ValueError("y must be a 1D numeric array")
    if rows.shape[0] == 0:
        raise ValueError("y must not be empty")
    if not np.all(np.isfinite(rows)):
        raise ValueError("y must contain only finite values")
    return np.ascontiguousarray(rows, dtype=np.float64)


def _as_sample_weight_list(values: Iterable[float] | None, expected: int) -> list[float] | None:
    weights = _as_sample_weight_array(values, expected)
    return None if weights is None else weights.tolist()


def _as_sample_weight_array(values: Any | None, expected: int) -> np.ndarray | None:
    if values is None:
        return None
    try:
        weights = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_weight must be a 1D numeric array") from exc
    if weights.ndim != 1:
        raise ValueError("sample_weight must be a 1D numeric array")
    if weights.shape[0] != expected:
        raise ValueError("sample_weight length must match y")
    if not np.all(np.isfinite(weights)) or bool(np.any(weights < 0.0)):
        raise ValueError("sample_weight must contain only finite non-negative values")
    return np.ascontiguousarray(weights, dtype=np.float64)


def _is_axis_splitters(splitters: Any) -> bool:
    return splitters is None or list(splitters) == ["axis"]


def _feature_schema_metadata(feature_schema: Any | None) -> Any | None:
    if feature_schema is None:
        return None
    if hasattr(feature_schema, "to_dict"):
        return feature_schema.to_dict()
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


def _normalize_sparse_sets(
    values: Any | None,
    expected_rows: int,
    expected_names: list[str] | None = None,
) -> tuple[list[list[list[int]]], list[str]]:
    if values is None:
        return [], []
    if isinstance(values, dict):
        mapping = {str(name): column for name, column in values.items()}
        if expected_names is not None:
            missing = [name for name in expected_names if name not in mapping]
            unknown = [name for name in mapping if name not in expected_names]
            if missing or unknown:
                raise ValueError(
                    f"sparse_sets columns do not match fitted columns; missing={missing}, "
                    f"unknown={unknown}"
                )
            items = [(name, mapping[name]) for name in expected_names]
        else:
            items = list(mapping.items())
    else:
        items = [(f"sparse_set_{idx}", column) for idx, column in enumerate(values)]
    columns: list[list[list[int]]] = []
    names: list[str] = []
    for name, column in items:
        rows = []
        for row_index, row in enumerate(column):
            ids = []
            for value in row:
                ident = _normalize_sparse_id(value)
                if ident < 0:
                    raise ValueError(
                        f"sparse_sets column {name!r} row {row_index} contains a negative ID"
                    )
                ids.append(ident)
            rows.append(ids)
        if len(rows) != expected_rows:
            raise ValueError(
                "each sparse_sets column must have the same number of rows as the dense input"
            )
        columns.append(rows)
        names.append(name)
    return columns, names


def _normalize_sparse_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("sparse_sets IDs must be non-negative integers")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    raise ValueError("sparse_sets IDs must be non-negative integers")


def _encode_sparse_columns(
    columns: list[list[list[int]]],
) -> tuple[list[list[int]], list[list[int]]]:
    encoded_offsets: list[list[int]] = []
    encoded_ids: list[list[int]] = []
    for column in columns:
        offsets = [0]
        ids: list[int] = []
        for row in column:
            ids.extend(row)
            offsets.append(len(ids))
        encoded_offsets.append(offsets)
        encoded_ids.append(ids)
    return encoded_offsets, encoded_ids


def _rust_feature_schema_json(
    feature_schema: Any | None,
    dense_width: int,
    sparse_names: list[str],
) -> str | None:
    if feature_schema is None:
        if not sparse_names:
            return None
        payload = {
            "names": [f"feature_{idx}" for idx in range(dense_width)] + sparse_names,
            "kinds": ["Numeric" for _ in range(dense_width)] + ["SparseSet" for _ in sparse_names],
        }
        return json.dumps(payload)
    payload = _rust_feature_schema_payload(feature_schema, dense_width, sparse_names)
    return json.dumps(payload)


def _rust_feature_schema_payload(
    feature_schema: Any,
    dense_width: int,
    sparse_names: list[str],
) -> dict[str, Any]:
    if hasattr(feature_schema, "to_rust_payload"):
        payload = feature_schema.to_rust_payload(dense_width, sparse_names)
        names = [str(name) for name in payload["names"]]
        kinds = [_rust_feature_kind(kind) for kind in payload["kinds"]]
        _validate_schema_length(names, kinds, dense_width, sparse_names)
        return {"names": names, "kinds": kinds}

    if isinstance(feature_schema, dict) and "names" in feature_schema and "kinds" in feature_schema:
        names = [str(name) for name in feature_schema["names"]]
        kinds = [_rust_feature_kind(kind) for kind in feature_schema["kinds"]]
        _validate_schema_length(names, kinds, dense_width, sparse_names)
        return {"names": names, "kinds": kinds}

    if isinstance(feature_schema, dict) and (
        "dense" in feature_schema or "sparse_sets" in feature_schema
    ):
        dense_entries = list(feature_schema.get("dense", []))
        sparse_entries = list(feature_schema.get("sparse_sets", []))
        names = [
            _schema_entry_name(entry, idx, "feature") for idx, entry in enumerate(dense_entries)
        ]
        kinds = [_schema_entry_kind(entry, "numeric") for entry in dense_entries]
        names.extend(
            _schema_entry_name(entry, idx, "sparse_set") for idx, entry in enumerate(sparse_entries)
        )
        kinds.extend(_schema_entry_kind(entry, "sparse_set") for entry in sparse_entries)
        _validate_schema_length(names, kinds, dense_width, sparse_names)
        return {"names": names, "kinds": kinds}

    if isinstance(feature_schema, dict):
        names = [str(name) for name in feature_schema]
        kinds = []
        for value in feature_schema.values():
            if isinstance(value, dict):
                kinds.append(_schema_entry_kind(value, "numeric"))
            else:
                kinds.append("Numeric")
        _validate_schema_length(names, kinds, dense_width, sparse_names)
        return {"names": names, "kinds": kinds}

    raise ValueError(
        "feature_schema must be a Rust schema {'names','kinds'} mapping or a "
        "{'dense','sparse_sets'} mapping"
    )


def _schema_entry_name(entry: Any, idx: int, prefix: str) -> str:
    if isinstance(entry, dict):
        return str(entry.get("name", f"{prefix}_{idx}"))
    if isinstance(entry, tuple) and len(entry) == 2:
        return str(entry[0])
    return str(entry)


def _schema_entry_kind(entry: Any, default: str) -> Any:
    if isinstance(entry, dict):
        return _rust_feature_kind(entry.get("kind", entry.get("role", default)), entry)
    if isinstance(entry, tuple) and len(entry) == 2:
        return _rust_feature_kind(entry[1])
    if default == "sparse_set":
        return "SparseSet"
    return "Numeric"


def _rust_feature_kind(kind: Any, entry: dict[str, Any] | None = None) -> Any:
    return normalize_feature_kind(kind, entry)


def _validate_schema_length(
    names: list[str],
    kinds: list[Any],
    dense_width: int,
    sparse_names: list[str],
) -> None:
    expected = dense_width + len(sparse_names)
    if len(names) != len(kinds):
        raise ValueError("feature_schema names length must match kinds length")
    if len(names) != expected:
        raise ValueError(
            f"feature_schema length {len(names)} does not match dataset feature count {expected}"
        )


def _json_attr(model: Any, attr: str) -> Any | None:
    payload = getattr(model, attr, None)
    if payload is None:
        return None
    return json.loads(payload)


def _sparse_names_from_feature_schema(feature_schema: Any | None) -> list[str]:
    if not isinstance(feature_schema, dict):
        return []
    names = feature_schema.get("names")
    kinds = feature_schema.get("kinds")
    if not isinstance(names, list) or not isinstance(kinds, list):
        return []
    return [
        str(name)
        for name, kind in zip(names, kinds, strict=False)
        if kind == "SparseSet" or kind == {"SparseSet": {}}
    ]


def _looks_like_native_artifact(payload: Any) -> bool:
    return isinstance(payload, dict) and "artifact_version" in payload and "trees" in payload


def _is_valid_splitter_name(splitter: Any) -> bool:
    if not isinstance(splitter, str):
        return False
    if splitter in _VALID_SPLITTERS:
        return True
    if splitter.startswith("axis_histogram:") or splitter.startswith("axis_hist:"):
        try:
            bins = int(splitter.split(":", 1)[1])
        except ValueError:
            return False
        return bins >= 2
    if not splitter.startswith("periodic:"):
        return False
    try:
        period = float(splitter.removeprefix("periodic:"))
    except ValueError:
        return False
    return math.isfinite(period) and period > 0.0


def _fit_native(
    model: Any,
    rows: np.ndarray,
    targets: np.ndarray,
    sample_weight: np.ndarray | None,
    sparse_sets: list[list[list[int]]],
    sparse_offsets: list[list[int]],
    sparse_ids: list[list[int]],
    feature_schema_json: str | None,
) -> None:
    if hasattr(model, "fit_arrays"):
        try:
            model.fit_arrays(
                rows,
                targets,
                sample_weight,
                sparse_offsets,
                sparse_ids,
                feature_schema_json,
            )
            return
        except TypeError:
            pass

    row_list = rows.tolist()
    target_list = targets.tolist()
    weight_list = None if sample_weight is None else sample_weight.tolist()
    try:
        model.fit(row_list, target_list, weight_list, sparse_sets, feature_schema_json)
    except TypeError as exc:
        if sparse_sets or feature_schema_json is not None:
            raise NotImplementedError(
                "the native backend does not support sparse_sets or feature_schema in this build"
            ) from exc
        if sample_weight is None:
            try:
                model.fit(row_list, target_list)
                return
            except TypeError:
                pass
        else:
            try:
                model.fit(row_list, target_list, weight_list)
                return
            except TypeError:
                pass
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
