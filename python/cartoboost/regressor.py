from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Iterable
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from ._native import CartoBoostRegressor as _NativeRegressorModel
from .schema import FeatureKind, normalize_feature_kind

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


class CartoBoostRegressor(RegressorMixin, BaseEstimator):
    """Small sklearn-style gradient boosted stump regressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        min_gain: float = 1e-8,
        loss: str = "l2",
        quantile_alpha: float = 0.5,
        huber_delta: float = 1.0,
        log_offset: float = 1.0,
        loss_params: dict[str, Any] | None = None,
        splitters: list[str] | None = None,
        leaf_predictor: str = "constant",
        linear_leaf_features: list[str] | None = None,
        fuzzy: bool = False,
        fuzzy_bandwidth: float = 0.0,
        fuzzy_kernel: str = "linear",
        l2_regularization: float = 1.0,
        constant_l2_regularization: float = 0.0,
        random_state: int | None = None,
        n_threads: int | None = None,
        monotonic_constraints: list[int] | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.loss = loss
        self.quantile_alpha = quantile_alpha
        self.huber_delta = huber_delta
        self.log_offset = log_offset
        self.loss_params = loss_params
        self.splitters = splitters
        self.leaf_predictor = leaf_predictor
        self.linear_leaf_features = linear_leaf_features
        self.fuzzy = fuzzy
        self.fuzzy_bandwidth = fuzzy_bandwidth
        self.fuzzy_kernel = fuzzy_kernel
        self.l2_regularization = l2_regularization
        self.constant_l2_regularization = constant_l2_regularization
        self.random_state = random_state
        self.n_threads = n_threads
        self.monotonic_constraints = monotonic_constraints
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
            "quantile_alpha": self.quantile_alpha,
            "huber_delta": self.huber_delta,
            "log_offset": self.log_offset,
            "loss_params": self.loss_params,
            "splitters": self.splitters,
            "leaf_predictor": self.leaf_predictor,
            "linear_leaf_features": self.linear_leaf_features,
            "fuzzy": self.fuzzy,
            "fuzzy_bandwidth": self.fuzzy_bandwidth,
            "fuzzy_kernel": self.fuzzy_kernel,
            "l2_regularization": self.l2_regularization,
            "constant_l2_regularization": self.constant_l2_regularization,
            "random_state": self.random_state,
            "n_threads": self.n_threads,
            "monotonic_constraints": self.monotonic_constraints,
        }

    def set_params(self, **params: Any) -> CartoBoostRegressor:
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
    ) -> CartoBoostRegressor:
        del eval_set
        self._validate_params()
        if hasattr(self, "_constant_prediction_value_"):
            delattr(self, "_constant_prediction_value_")
        feature_names = _feature_names(X)
        dense_array = _as_2d_float_array(X, check_finite=False)
        targets_array = _as_1d_float_array(y)
        if dense_array.shape[0] != targets_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        weights_array = _as_sample_weight_array(sample_weight, targets_array.shape[0])
        loss_params = _resolved_loss_params(
            self.loss,
            self.quantile_alpha,
            self.huber_delta,
            self.log_offset,
            self.loss_params,
        )
        sparse_columns, sparse_names = _normalize_sparse_sets(sparse_sets, targets_array.shape[0])
        sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_columns)
        schema_json = _rust_feature_schema_json(feature_schema, dense_array.shape[1], sparse_names)
        schema_metadata = _feature_schema_metadata(feature_schema)
        self.n_features_in_ = dense_array.shape[1]
        if (
            self.monotonic_constraints is not None
            and len(self.monotonic_constraints) != self.n_features_in_
        ):
            raise ValueError(
                f"monotonic_constraints has length {len(self.monotonic_constraints)}, "
                f"but X has {self.n_features_in_} features"
            )
        self.n_sparse_sets_in_ = len(sparse_columns)
        self.sparse_set_names_ = sparse_names
        self.feature_schema_ = schema_metadata
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)

        model = _NativeRegressorModel(
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            max_depth=int(self.max_depth),
            min_samples_leaf=int(self.min_samples_leaf),
            min_gain=float(self.min_gain),
            loss=str(self.loss),
            quantile_alpha=float(loss_params["quantile_alpha"]),
            huber_delta=float(loss_params["huber_delta"]),
            log_offset=float(loss_params["log_offset"]),
            splitters=list(self.splitters or ["axis"]),
            leaf_predictor=str(self.leaf_predictor),
            linear_leaf_features=_resolve_linear_leaf_features(
                self.linear_leaf_features,
                dense_array.shape[1],
            ),
            l2_regularization=float(self.l2_regularization),
            constant_l2_regularization=float(self.constant_l2_regularization),
            fuzzy=bool(self.fuzzy),
            fuzzy_bandwidth=float(self.fuzzy_bandwidth),
            fuzzy_kernel=str(self.fuzzy_kernel),
            monotonic_constraints=(
                None
                if self.monotonic_constraints is None
                else [int(value) for value in self.monotonic_constraints]
            ),
        )
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
        if int(self.max_depth) == 0 and self.leaf_predictor == "constant" and not self.fuzzy:
            training_targets = _training_targets(
                targets_array.tolist(),
                self.loss,
                float(loss_params["log_offset"]),
            )
            if weights_array is None:
                self._constant_prediction_value_ = _initial_value(
                    training_targets,
                    None,
                    self.loss,
                    float(loss_params["quantile_alpha"]),
                )
            else:
                weight_sum = float(np.sum(weights_array))
                self._constant_prediction_value_ = (
                    _initial_value(
                        training_targets,
                        weights_array.tolist(),
                        self.loss,
                        float(loss_params["quantile_alpha"]),
                    )
                    if weight_sum > 0.0
                    else 0.0
                )
            self._constant_prediction_value_ = _inverse_prediction(
                self._constant_prediction_value_,
                self.loss,
            )
        self.is_fitted_ = True
        return self

    def predict(self, X: Iterable[Iterable[float]], sparse_sets: Any | None = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("CartoBoostRegressor is not fitted")
        expected_sparse_count = getattr(self, "n_sparse_sets_in_", 0)
        if hasattr(self, "_constant_prediction_value_") and not getattr(
            self, "requires_sparse_sets_", False
        ):
            rows, cols = _shape_2d(X)
            if hasattr(self, "n_features_in_") and cols != self.n_features_in_:
                raise ValueError(
                    f"X has {cols} features, but CartoBoostRegressor was fitted with "
                    f"{self.n_features_in_} features"
                )
            return np.broadcast_to(
                np.asarray(self._constant_prediction_value_, dtype=float),
                (rows,),
            )
        dense_array = _as_2d_float_array(X, check_finite=False)
        if hasattr(self, "n_features_in_") and dense_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {dense_array.shape[1]} features, but CartoBoostRegressor was fitted with "
                f"{self.n_features_in_} features"
            )
        if not getattr(self, "requires_sparse_sets_", False):
            sparse_columns: list[list[list[int]]] = []
            sparse_names: list[str] = []
            sparse_offsets: list[list[int]] = []
            sparse_ids: list[list[int]] = []
        elif _is_empty_sparse_sets(sparse_sets) and expected_sparse_count == 0:
            sparse_columns: list[list[list[int]]] = []
            sparse_names: list[str] = []
            sparse_offsets: list[list[int]] = []
            sparse_ids: list[list[int]] = []
        else:
            sparse_columns, sparse_names = _normalize_sparse_sets(
                sparse_sets,
                dense_array.shape[0],
                getattr(self, "sparse_set_names_", None),
            )
            sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_columns)
        if sparse_columns and len(sparse_columns) != expected_sparse_count:
            raise ValueError(
                f"sparse_sets has {len(sparse_columns)} columns, but CartoBoostRegressor was "
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
        if not sparse_columns and getattr(self, "requires_sparse_sets_", False):
            raise ValueError("sparse_sets are required for prediction with this sparse-list model")
        try:
            return np.asarray(
                self._model.predict_arrays(dense_array, sparse_offsets, sparse_ids),
                dtype=float,
            )
        except TypeError:
            rows = dense_array.tolist()
            return np.asarray(list(self._model.predict(rows, sparse_columns)), dtype=float)

    def predict_additive_values(
        self,
        X: Iterable[Iterable[float]],
        sparse_sets: Any | None = None,
    ) -> np.ndarray:
        """Return additive prediction components whose row sums equal ``predict(X)``."""
        if self._model is None:
            raise RuntimeError("CartoBoostRegressor is not fitted")
        dense_array, sparse_columns, sparse_offsets, sparse_ids = self._prediction_inputs(
            X,
            sparse_sets,
        )
        if hasattr(self._model, "predict_additive_arrays"):
            return np.asarray(
                self._model.predict_additive_arrays(
                    dense_array,
                    sparse_offsets,
                    sparse_ids,
                ),
                dtype=float,
            )
        rows = dense_array.tolist()
        return np.asarray(self._model.predict_additive(rows, sparse_columns), dtype=float)

    def _prediction_inputs(
        self,
        X: Iterable[Iterable[float]],
        sparse_sets: Any | None,
    ) -> tuple[np.ndarray, list[list[list[int]]], list[list[int]], list[list[int]]]:
        expected_sparse_count = getattr(self, "n_sparse_sets_in_", 0)
        dense_array = _as_2d_float_array(X, check_finite=False)
        if hasattr(self, "n_features_in_") and dense_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {dense_array.shape[1]} features, but CartoBoostRegressor was fitted with "
                f"{self.n_features_in_} features"
            )
        if not getattr(self, "requires_sparse_sets_", False):
            sparse_columns: list[list[list[int]]] = []
            sparse_names: list[str] = []
            sparse_offsets: list[list[int]] = []
            sparse_ids: list[list[int]] = []
        elif _is_empty_sparse_sets(sparse_sets) and expected_sparse_count == 0:
            sparse_columns = []
            sparse_names = []
            sparse_offsets = []
            sparse_ids = []
        else:
            sparse_columns, sparse_names = _normalize_sparse_sets(
                sparse_sets,
                dense_array.shape[0],
                getattr(self, "sparse_set_names_", None),
            )
            sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_columns)
        if sparse_columns and len(sparse_columns) != expected_sparse_count:
            raise ValueError(
                f"sparse_sets has {len(sparse_columns)} columns, but CartoBoostRegressor was "
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
        if not sparse_columns and getattr(self, "requires_sparse_sets_", False):
            raise ValueError("sparse_sets are required for prediction with this sparse-list model")
        return dense_array, sparse_columns, sparse_offsets, sparse_ids

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
        decomposition: str = "features",
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
            decomposition=decomposition,
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
        decomposition: str = "features",
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
            decomposition=decomposition,
            **kwargs,
        )

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("CartoBoostRegressor is not fitted")
        path = Path(path)
        if hasattr(self._model, "save"):
            self._model.save(path)
            return
        raise NotImplementedError("native model does not support save")

    def save_weights(self, path: str | Path, *, format: str = "auto") -> None:
        """Save a versioned, prediction-ready weights artifact.

        JSON weights artifacts are the stable CartoBoost interchange format.
        ONNX export is optional and currently supports dense axis-tree models
        with constant leaves.
        """
        if self._model is None:
            raise RuntimeError("CartoBoostRegressor is not fitted")
        path = Path(path)
        resolved_format = _resolve_weights_format(path, format)
        if resolved_format == "onnx":
            artifact = self._weights_artifact_payload()
            _save_weights_onnx(artifact, path)
            return
        if resolved_format != "json":
            raise ValueError("format must be one of 'auto', 'json', or 'onnx'")

        if hasattr(self._model, "save_weights"):
            self._model.save_weights(path)
            return
        raise NotImplementedError("native model does not support save_weights")

    @classmethod
    def load(cls, path: str | Path) -> CartoBoostRegressor:
        path = Path(path)
        native_model = _NativeRegressorModel.load(path)
        return cls._from_native_model(native_model)

    @classmethod
    def load_weights(cls, path: str | Path) -> CartoBoostRegressor:
        path = Path(path)
        native_model = _NativeRegressorModel.load_weights(path)
        return cls._from_native_model(native_model)

    @classmethod
    def _from_native_model(cls, native_model: Any) -> CartoBoostRegressor:
        estimator = cls(
            n_estimators=native_model.n_estimators,
            learning_rate=native_model.learning_rate,
            max_depth=native_model.max_depth,
            min_samples_leaf=native_model.min_samples_leaf,
            min_gain=native_model.min_gain,
            loss=str(getattr(native_model, "loss", "l2")),
            quantile_alpha=float(getattr(native_model, "quantile_alpha", 0.5)),
            huber_delta=float(getattr(native_model, "huber_delta", 1.0)),
            log_offset=float(getattr(native_model, "log_offset", 1.0)),
            splitters=list(getattr(native_model, "splitters", ["axis"])),
            leaf_predictor=str(getattr(native_model, "leaf_predictor", "constant")),
            linear_leaf_features=[
                str(feature) for feature in getattr(native_model, "linear_leaf_features", [])
            ],
            fuzzy=bool(getattr(native_model, "fuzzy", False)),
            fuzzy_bandwidth=float(getattr(native_model, "fuzzy_bandwidth", 0.0)),
            fuzzy_kernel=str(getattr(native_model, "fuzzy_kernel", "linear")),
            l2_regularization=float(getattr(native_model, "l2_regularization", 1.0)),
            constant_l2_regularization=float(
                getattr(native_model, "constant_l2_regularization", 0.0)
            ),
            monotonic_constraints=list(getattr(native_model, "monotonic_constraints", [])) or None,
        )
        estimator._model = native_model
        estimator._backend_used = "rust"
        estimator.n_features_in_ = native_model.feature_count
        estimator.feature_schema_ = _json_attr(native_model, "feature_schema_json")
        estimator.sparse_set_names_ = _sparse_names_from_feature_schema(estimator.feature_schema_)
        estimator.n_sparse_sets_in_ = len(estimator.sparse_set_names_)
        estimator.metadata_ = _json_attr(native_model, "metadata_json")
        estimator.training_config_ = _json_attr(native_model, "training_config_json")
        estimator.requires_sparse_sets_ = bool(getattr(native_model, "requires_sparse_sets", False))
        estimator.is_fitted_ = True
        return estimator

    def _weights_artifact_payload(self) -> dict[str, Any]:
        if hasattr(self._model, "save_weights"):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "weights.json"
                self._model.save_weights(temp_path)
                return json.loads(temp_path.read_text(encoding="utf-8"))
        raise NotImplementedError("native model does not support save_weights")

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
        if self.loss not in {
            "l2",
            "squared_error",
            "l1",
            "mae",
            "absolute_error",
            "least_absolute_deviation",
            "lad",
            "huber",
            "log_l2",
            "log",
            "log_squared_error",
            "quantile",
            "pinball",
        }:
            raise ValueError("loss must be 'l2', 'l1', 'huber', 'log_l2', or 'quantile'")
        loss_params = _resolved_loss_params(
            self.loss,
            self.quantile_alpha,
            self.huber_delta,
            self.log_offset,
            self.loss_params,
        )
        quantile_alpha = float(loss_params["quantile_alpha"])
        if not math.isfinite(quantile_alpha) or quantile_alpha <= 0.0 or quantile_alpha >= 1.0:
            raise ValueError("quantile_alpha must be finite and in (0, 1)")
        huber_delta = float(loss_params["huber_delta"])
        if not math.isfinite(huber_delta) or huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive and finite")
        log_offset = float(loss_params["log_offset"])
        if not math.isfinite(log_offset) or log_offset <= 0.0:
            raise ValueError("log_offset must be positive and finite")
        if self.loss in {"log_l2", "log", "log_squared_error"} and log_offset != 1.0:
            raise ValueError("log_l2 currently supports log_offset=1.0")
        if self.leaf_predictor not in {"constant", "linear"}:
            raise ValueError("leaf_predictor must be 'constant' or 'linear'")
        if (
            self.loss in {"l1", "mae", "absolute_error", "least_absolute_deviation", "lad"}
            and self.leaf_predictor != "constant"
        ):
            raise ValueError(f"{self.loss} loss requires leaf_predictor='constant'")
        if self.loss in {"quantile", "pinball"} and self.leaf_predictor != "constant":
            raise ValueError("quantile loss requires leaf_predictor='constant'")
        if (
            self.loss in {"huber", "log_l2", "log", "log_squared_error"}
            and self.leaf_predictor != "constant"
        ):
            raise ValueError(f"{self.loss} loss requires leaf_predictor='constant'")
        if float(self.l2_regularization) < 0 or not math.isfinite(float(self.l2_regularization)):
            raise ValueError("l2_regularization must be finite and non-negative")
        constant_l2 = float(self.constant_l2_regularization)
        if constant_l2 < 0 or not math.isfinite(constant_l2):
            raise ValueError("constant_l2_regularization must be finite and non-negative")
        if float(self.fuzzy_bandwidth) < 0 or not math.isfinite(float(self.fuzzy_bandwidth)):
            raise ValueError("fuzzy_bandwidth must be finite and non-negative")
        if self.fuzzy_kernel not in {
            "linear",
            "triangular",
            "gaussian",
            "exponential",
            "bisquare",
            "epanechnikov",
            "tricube",
        }:
            raise ValueError(
                "fuzzy_kernel must be 'linear', 'gaussian', 'exponential', "
                "'bisquare', 'epanechnikov', or 'tricube'"
            )
        if self.monotonic_constraints is not None:
            constraints = list(self.monotonic_constraints)
            if any(int(value) not in {-1, 0, 1} for value in constraints):
                raise ValueError("monotonic_constraints values must be -1, 0, or 1")
            if self.leaf_predictor != "constant":
                raise ValueError("monotonic constraints require leaf_predictor='constant'")
            if self.fuzzy:
                raise ValueError("monotonic constraints require fuzzy=False")
            if not _is_axis_splitters(self.splitters) and not _is_axis_hist_splitters(
                self.splitters
            ):
                raise ValueError("monotonic constraints support only axis splitters")
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


def _shape_2d(values: Any) -> tuple[int, int]:
    values = _to_numpy(values)
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) == 2:
        rows, cols = int(shape[0]), int(shape[1])
        if rows == 0:
            raise ValueError("X must not be empty")
        if cols == 0:
            raise ValueError("X rows must contain at least one feature")
        return rows, cols
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a rectangular 2D array") from exc
    if array.ndim != 2:
        raise ValueError("X must be a rectangular 2D array")
    if array.shape[0] == 0:
        raise ValueError("X must not be empty")
    if array.shape[1] == 0:
        raise ValueError("X rows must contain at least one feature")
    return int(array.shape[0]), int(array.shape[1])


def _as_2d_float_array(values: Any, *, check_finite: bool = True) -> np.ndarray:
    values = _to_numpy(values)
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
    if check_finite and not np.all(np.isfinite(array)):
        raise ValueError("X must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _feature_names(values: Any) -> list[str] | None:
    columns = getattr(values, "columns", None)
    if columns is None:
        return None
    return [str(column) for column in columns]


def _as_1d_float_array(values: Any) -> np.ndarray:
    values = _to_numpy(values)
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


def _as_sample_weight_array(values: Any | None, expected: int) -> np.ndarray | None:
    if values is None:
        return None
    values = _to_numpy(values)
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


def _is_axis_hist_splitters(splitters: Any) -> bool:
    if splitters is None:
        return True
    names = list(splitters)
    return bool(names) and all(
        name in {"axis", "axis_histogram", "axis_hist", "histogram"}
        or str(name).startswith(("axis_histogram:", "axis_hist:"))
        for name in names
    )


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
    elif hasattr(values, "columns"):
        mapping = {str(name): values[name] for name in values.columns}
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
        for row_index, row in enumerate(_sequence_values(column)):
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


def _is_empty_sparse_sets(values: Any | None) -> bool:
    if values is None:
        return True
    if isinstance(values, dict):
        return len(values) == 0
    columns = getattr(values, "columns", None)
    if columns is not None:
        return len(columns) == 0
    try:
        return len(values) == 0
    except TypeError:
        return False


def _to_numpy(values: Any) -> Any:
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return values


def _sequence_values(values: Any) -> Any:
    if hasattr(values, "to_list"):
        return values.to_list()
    if hasattr(values, "tolist"):
        return values.tolist()
    return values


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
            "kinds": [FeatureKind.NUMERIC for _ in range(dense_width)]
            + [FeatureKind.SPARSE_SET for _ in sparse_names],
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
        kinds = [_schema_entry_kind(entry, FeatureKind.NUMERIC) for entry in dense_entries]
        names.extend(
            _schema_entry_name(entry, idx, "sparse_set") for idx, entry in enumerate(sparse_entries)
        )
        kinds.extend(_schema_entry_kind(entry, FeatureKind.SPARSE_SET) for entry in sparse_entries)
        _validate_schema_length(names, kinds, dense_width, sparse_names)
        return {"names": names, "kinds": kinds}

    if isinstance(feature_schema, dict):
        names = [str(name) for name in feature_schema]
        kinds = []
        for value in feature_schema.values():
            if isinstance(value, dict):
                kinds.append(_schema_entry_kind(value, FeatureKind.NUMERIC))
            else:
                kinds.append(FeatureKind.NUMERIC)
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


def _schema_entry_kind(entry: Any, default: FeatureKind) -> Any:
    match entry:
        case dict():
            return _rust_feature_kind(entry.get("kind", entry.get("role", default)), entry)
        case tuple() if len(entry) == 2:
            return _rust_feature_kind(entry[1])
        case _ if default is FeatureKind.SPARSE_SET:
            return FeatureKind.SPARSE_SET
        case _:
            return FeatureKind.NUMERIC


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
        if _is_sparse_set_schema_kind(kind)
    ]


def _is_sparse_set_schema_kind(kind: Any) -> bool:
    match kind:
        case FeatureKind.SPARSE_SET:
            return True
        case {FeatureKind.SPARSE_SET: dict()}:
            return True
        case _:
            return False


def _looks_like_native_artifact(payload: Any) -> bool:
    return isinstance(payload, dict) and "artifact_version" in payload and "trees" in payload


def _looks_like_native_weights_artifact(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("artifact_type") == "cartoboost.weights"
        and isinstance(payload.get("model"), dict)
        and _looks_like_native_artifact(payload["model"])
    )


def _resolve_weights_format(path: Path, requested: str) -> str:
    normalized = requested.lower()
    if normalized != "auto":
        return normalized
    return "onnx" if path.suffix.lower() == ".onnx" else "json"


def _save_weights_onnx(artifact: dict[str, Any], path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("ONNX export requires installing the optional 'onnx' package") from exc

    model_payload = _onnx_model_payload(artifact)
    attrs = _onnx_tree_ensemble_attrs(model_payload)
    feature_count = int(model_payload["feature_count"])

    node = helper.make_node(
        "TreeEnsembleRegressor",
        inputs=["X"],
        outputs=["predictions"],
        domain="ai.onnx.ml",
        aggregate_function="SUM",
        base_values=[float(model_payload["init_prediction"])],
        n_targets=1,
        post_transform="NONE",
        **attrs,
    )
    graph = helper.make_graph(
        [node],
        "cartoboost_tree_ensemble",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, feature_count])],
        [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, [None, 1])],
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="cartoboost",
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("ai.onnx.ml", 3),
        ],
    )
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, path)


def _onnx_model_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    if _looks_like_native_artifact(artifact):
        return artifact
    if _looks_like_native_weights_artifact(artifact):
        return artifact["model"]
    raise NotImplementedError("ONNX export requires a native CartoBoost artifact")


def _onnx_tree_ensemble_attrs(model_payload: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, list[Any]] = {
        "nodes_treeids": [],
        "nodes_nodeids": [],
        "nodes_featureids": [],
        "nodes_modes": [],
        "nodes_values": [],
        "nodes_truenodeids": [],
        "nodes_falsenodeids": [],
        "nodes_missing_value_tracks_true": [],
        "target_treeids": [],
        "target_nodeids": [],
        "target_ids": [],
        "target_weights": [],
    }
    learning_rate = float(model_payload["learning_rate"])
    for tree_id, tree in enumerate(model_payload.get("trees", [])):
        next_id = 0

        def next_node_id() -> int:
            nonlocal next_id
            node_id = next_id
            next_id += 1
            return node_id

        def visit(node: dict[str, Any], tree_id: int = tree_id) -> int:
            node_id = next_node_id()
            if "Leaf" in node:
                _append_onnx_node(attrs, tree_id, node_id, 0, "LEAF", 0.0, 0, 0, 0)
                attrs["target_treeids"].append(tree_id)
                attrs["target_nodeids"].append(node_id)
                attrs["target_ids"].append(0)
                attrs["target_weights"].append(learning_rate * float(node["Leaf"]["value"]))
                return node_id
            if "LinearLeaf" in node:
                raise NotImplementedError("ONNX export does not support linear leaf models")
            if "Branch" not in node:
                raise ValueError("unsupported CartoBoost node encoding in weights artifact")
            branch = node["Branch"]
            split = branch["split"]
            if "Axis" not in split:
                raise NotImplementedError("ONNX export currently supports only axis splits")
            axis = split["Axis"]
            left_id = visit(branch["left"])
            right_id = visit(branch["right"])
            _append_onnx_node(
                attrs,
                tree_id,
                node_id,
                int(axis["feature"]),
                "BRANCH_LEQ",
                float(axis["threshold"]),
                left_id,
                right_id,
                1 if bool(axis.get("missing_goes_left", True)) else 0,
            )
            return node_id

        visit(tree["root"])
    return attrs


def _append_onnx_node(
    attrs: dict[str, list[Any]],
    tree_id: int,
    node_id: int,
    feature_id: int,
    mode: str,
    value: float,
    true_id: int,
    false_id: int,
    missing_tracks_true: int,
) -> None:
    attrs["nodes_treeids"].append(tree_id)
    attrs["nodes_nodeids"].append(node_id)
    attrs["nodes_featureids"].append(feature_id)
    attrs["nodes_modes"].append(mode)
    attrs["nodes_values"].append(value)
    attrs["nodes_truenodeids"].append(true_id)
    attrs["nodes_falsenodeids"].append(false_id)
    attrs["nodes_missing_value_tracks_true"].append(missing_tracks_true)


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


def _resolved_loss_params(
    loss: str,
    quantile_alpha: float,
    huber_delta: float,
    log_offset: float,
    loss_params: dict[str, Any] | None,
) -> dict[str, float]:
    params = dict(loss_params or {})
    return {
        "quantile_alpha": float(params.get("alpha", params.get("quantile_alpha", quantile_alpha))),
        "huber_delta": float(params.get("delta", params.get("huber_delta", huber_delta))),
        "log_offset": float(params.get("offset", params.get("log_offset", log_offset))),
    }


def _training_targets(values: list[float], loss: str, log_offset: float) -> list[float]:
    if loss not in {"log_l2", "log", "log_squared_error"}:
        return values
    if any(value + log_offset <= 0.0 for value in values):
        raise ValueError("log_l2 targets must be greater than -log_offset")
    return [math.log(value + log_offset) for value in values]


def _inverse_prediction(prediction: float, loss: str) -> float:
    if loss in {"log_l2", "log", "log_squared_error"}:
        return math.expm1(prediction)
    return prediction


def _initial_value(
    values: list[float],
    weights: list[float] | None,
    loss: str,
    quantile_alpha: float,
) -> float:
    resolved_weights = weights or [1.0 for _ in values]
    return _leaf_value(values, resolved_weights, loss, quantile_alpha)


def _leaf_value(
    values: list[float], weights: list[float], loss: str, quantile_alpha: float
) -> float:
    if loss in {"quantile", "pinball"}:
        return _weighted_quantile(values, weights, quantile_alpha)
    if loss in {"l1", "mae", "absolute_error", "least_absolute_deviation", "lad"}:
        return _weighted_quantile(values, weights, 0.5)
    return _weighted_mean(values, weights)


def _weighted_quantile(values: list[float], weights: list[float], alpha: float) -> float:
    pairs = sorted(
        (value, weight)
        for value, weight in zip(values, weights, strict=True)
        if math.isfinite(value) and math.isfinite(weight) and weight > 0.0
    )
    if not pairs:
        return 0.0
    total_weight = sum(weight for _, weight in pairs)
    threshold = alpha * total_weight
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(pairs[-1][0])


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / weight_sum
