from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error

from ..regressor import CartoBoostRegressor
from .features import NeuralEmbeddingFeatures


class NeuralEmbeddingRegressor:
    """Two-stage estimator that appends offline neural embeddings as dense columns.

    The estimator can train embeddings on residuals (the default) or on the full
    target. In both cases, the final model remains `CartoBoostRegressor`.
    """

    def __init__(
        self,
        *,
        dim: int = 16,
        fallback: str = "global_mean_vector",
        random_state: int | None = 42,
        neural_transformer: NeuralEmbeddingFeatures | None = None,
        use_residual: bool = True,
        drop_id_column: bool = True,
        id_column: int | str | None = None,
        base_model_kwargs: dict[str, Any] | None = None,
        final_model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.dim = int(dim)
        self.fallback = fallback
        self.random_state = random_state
        self.use_residual = use_residual
        self.drop_id_column = drop_id_column
        self.id_column = id_column
        self.neural_transformer = neural_transformer or NeuralEmbeddingFeatures(
            dim=dim,
            fallback=fallback,
            random_state=random_state,
        )
        self.base_model_kwargs = dict(base_model_kwargs or {})
        self.final_model_kwargs = dict(final_model_kwargs or self.base_model_kwargs)

        self._base_model: CartoBoostRegressor | None = None
        self._final_model: CartoBoostRegressor | None = None
        self._fit_timings_ms: dict[str, float] = {}
        self._feature_names_in_: np.ndarray | None = None
        self._n_features_in_ = 0

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any | None = None,
        sparse_sets: Any | None = None,
        id_column: int | str | None = None,
        ids: np.ndarray | list[Any] | None = None,
        **fit_kwargs: Any,
    ) -> NeuralEmbeddingRegressor:
        resolved_id_column = self._resolve_id_column(id_column)
        if resolved_id_column is None and ids is None:
            raise ValueError("id_column or ids is required for embedding lookup")

        dense, id_values = self._prepare_dense_and_ids(X, ids, resolved_id_column)
        target = _to_1d_float_array(y)

        if dense.shape[0] != target.shape[0]:
            raise ValueError("X and y must contain the same number of rows")

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (target.shape[0],):
                raise ValueError("sample_weight length must match y")

        fit_kwargs = dict(fit_kwargs)
        fit_kwargs.update({"sparse_sets": sparse_sets, "sample_weight": sample_weight})

        if self.use_residual:
            base_model = self._build_base_model()
            start = time.perf_counter()
            base_model.fit(dense, target, **fit_kwargs)
            self._fit_timings_ms["base_fit_ms"] = _ms_since(start)

            residual = target - base_model.predict(dense, sparse_sets=sparse_sets)
            residual_target = residual
            self._base_model = base_model
        else:
            residual_target = target

        start = time.perf_counter()
        self.neural_transformer.fit(id_values, residual_target)
        self._fit_timings_ms["neural_fit_ms"] = _ms_since(start)

        neural_features = self.neural_transformer.transform(id_values)
        if neural_features.shape != (dense.shape[0], self.dim):
            raise ValueError("neural transformer output dimension mismatch")

        x_train = _append_features(dense, neural_features)
        final_model = self._build_final_model()

        start = time.perf_counter()
        final_model.fit(x_train, target, **fit_kwargs)
        self._fit_timings_ms["final_fit_ms"] = _ms_since(start)
        self._final_model = final_model

        self._n_features_in_ = x_train.shape[1]
        self._feature_names_in_ = _build_neural_feature_names(
            getattr(X, "columns", None),
            dense.shape[1],
            self.dim,
        )
        return self

    def predict(
        self,
        X: Any,
        *,
        sparse_sets: Any | None = None,
        ids: np.ndarray | list[Any] | None = None,
        id_column: int | str | None = None,
    ) -> np.ndarray:
        if self._final_model is None:
            raise RuntimeError("NeuralEmbeddingRegressor is not fitted")

        dense, id_values = self._prepare_dense_and_ids(
            X,
            ids,
            self._resolve_id_column(id_column),
        )
        neural = self.neural_transformer.transform(id_values)
        x_pred = _append_features(dense, neural)
        return self._final_model.predict(x_pred, sparse_sets=sparse_sets)

    def transform(
        self,
        X: Any,
        *,
        ids: np.ndarray | list[Any] | None = None,
        id_column: int | str | None = None,
    ) -> np.ndarray:
        if self._final_model is None:
            raise RuntimeError("NeuralEmbeddingRegressor is not fitted")

        dense, id_values = self._prepare_dense_and_ids(
            X,
            ids,
            self._resolve_id_column(id_column),
        )
        return _append_features(dense, self.neural_transformer.transform(id_values))

    def score(
        self,
        X: Any,
        y: Any,
        *,
        sparse_sets: Any | None = None,
        id_column: int | str | None = None,
        ids: np.ndarray | list[Any] | None = None,
    ) -> float:
        predictions = self.predict(
            X,
            sparse_sets=sparse_sets,
            ids=ids,
            id_column=id_column,
        )
        return float(mean_absolute_error(y, predictions))

    def benchmark(
        self,
        X: Any,
        y: Any,
        *,
        sparse_sets: Any | None = None,
        id_column: int | str | None = None,
        ids: np.ndarray | list[Any] | None = None,
    ) -> dict[str, float]:
        start = time.perf_counter()
        pred = self.predict(X, sparse_sets=sparse_sets, ids=ids, id_column=id_column)
        mae = float(mean_absolute_error(y, pred))
        return {"mae": mae, "predict_ms": _ms_since(start)}

    @property
    def timings(self) -> dict[str, float]:
        return dict(self._fit_timings_ms)

    @property
    def n_features_in_(self) -> int:
        return self._n_features_in_

    @property
    def feature_names_in_(self) -> np.ndarray | None:
        return self._feature_names_in_

    @property
    def model(self) -> CartoBoostRegressor:
        if self._final_model is None:
            raise RuntimeError("NeuralEmbeddingRegressor is not fitted")
        return self._final_model

    def _build_base_model(self) -> CartoBoostRegressor:
        return CartoBoostRegressor(**self.base_model_kwargs)

    def _build_final_model(self) -> CartoBoostRegressor:
        return CartoBoostRegressor(**self.final_model_kwargs)

    def _resolve_id_column(self, id_column: int | str | None) -> int | str | None:
        if id_column is not None:
            return id_column
        return self.id_column

    def _prepare_dense_and_ids(
        self,
        X: Any,
        ids: np.ndarray | list[Any] | None,
        id_column: int | str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if ids is not None and id_column is not None:
            raise ValueError("provide either ids or id_column, not both")

        if ids is not None:
            id_values = _to_u64_ids(ids)
            dense = _to_2d_float_array(X)
            return dense, id_values

        if id_column is None:
            raise ValueError("id_column or ids is required")

        if isinstance(id_column, int):
            dense = _to_2d_float_array(X)
            if id_column < 0 or id_column >= dense.shape[1]:
                raise ValueError("id_column index is out of range")
            id_values = _to_u64_ids(dense[:, id_column])

            if self.drop_id_column:
                dense = np.delete(dense, id_column, axis=1)

            return dense, id_values

        if not hasattr(X, "__getitem__"):
            raise ValueError("id_column string lookup requires object with named columns")

        try:
            id_values = _to_u64_ids(X[id_column])
        except (TypeError, KeyError) as exc:
            raise ValueError(f"id_column {id_column!r} not found") from exc

        if self.drop_id_column:
            dense_obj = _drop_named_column(X, id_column)
            dense = _to_2d_float_array(dense_obj)
        else:
            dense = _to_2d_float_array(X)

        return dense, id_values


def benchmark_neural_vs_cartoboost(
    X: Any,
    y: Any,
    *,
    ids: np.ndarray | list[Any],
    split_ratio: float = 0.8,
    neural_kwargs: dict[str, Any] | None = None,
    cartoboost_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x_array = _to_2d_float_array(X)
    y_array = _to_1d_float_array(y)
    id_values = _to_u64_ids(ids)

    if x_array.shape[0] != y_array.shape[0] or x_array.shape[0] != id_values.shape[0]:
        raise ValueError("X, y, and ids must have the same number of rows")
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be in (0, 1)")

    split_index = int(x_array.shape[0] * split_ratio)
    if split_index <= 0 or split_index >= x_array.shape[0]:
        raise ValueError("split_ratio must keep both train and validation sets")

    x_train = x_array[:split_index]
    x_valid = x_array[split_index:]
    y_train = y_array[:split_index]
    y_valid = y_array[split_index:]
    id_train = id_values[:split_index]
    id_valid = id_values[split_index:]

    cartooboost_params = dict(cartoboost_kwargs or {})
    cartooboost = CartoBoostRegressor(**cartooboost_params)

    start = time.perf_counter()
    cartooboost.fit(x_train, y_train)
    carto_fit_ms = _ms_since(start)

    start = time.perf_counter()
    cart_pred = cartooboost.predict(x_valid)
    carto_pred_ms = _ms_since(start)
    carto_mae = float(mean_absolute_error(y_valid, cart_pred))

    hybrid = NeuralEmbeddingRegressor(
        dim=16,
        drop_id_column=False,
        **(neural_kwargs or {}),
    )
    start = time.perf_counter()
    hybrid.fit(x_train, y_train, ids=id_train)
    hybrid_fit_ms = _ms_since(start)

    start = time.perf_counter()
    hybrid_pred = hybrid.predict(x_valid, ids=id_valid)
    hybrid_pred_ms = _ms_since(start)
    hybrid_mae = float(mean_absolute_error(y_valid, hybrid_pred))

    return {
        "n_rows": int(x_array.shape[0]),
        "n_features": int(x_array.shape[1]),
        "structured_mae": carto_mae,
        "hybrid_mae": hybrid_mae,
        "improvement": carto_mae - hybrid_mae,
        "cartoboost_fit_ms": carto_fit_ms,
        "cartoboost_predict_ms": carto_pred_ms,
        "hybrid_fit_ms": hybrid_fit_ms,
        "hybrid_predict_ms": hybrid_pred_ms,
    }


def _to_2d_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("X must be a 2D numeric array or 2D table")
    if array.shape[0] == 0:
        raise ValueError("X must not be empty")
    if array.shape[1] == 0:
        raise ValueError("X must include at least one feature")
    return np.ascontiguousarray(array, dtype=np.float64)


def _to_1d_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("y must be a 1D numeric array")
    if array.shape[0] == 0:
        raise ValueError("y must not be empty")
    return np.ascontiguousarray(array, dtype=np.float64)


def _to_u64_ids(values: np.ndarray | list[Any] | tuple[Any, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("ids must be 1D")
    if array.ndim != 1:
        array = np.ravel(array)

    float_ids = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(float_ids)):
        raise ValueError("ids must be finite")

    return np.asarray(np.rint(float_ids).astype(np.uint64), dtype=np.uint64)


def _drop_named_column(values: Any, column: str) -> Any:
    if hasattr(values, "drop"):
        try:
            return values.drop(columns=[column])
        except Exception:
            pass

    if hasattr(values, "drop_columns"):
        try:
            return values.drop_columns([column])
        except Exception:
            pass

    if hasattr(values, "columns"):
        columns = list(values.columns)
        try:
            index = columns.index(column)
        except ValueError as exc:
            raise ValueError(f"id_column {column!r} not found") from exc

        array = _to_2d_float_array(values)
        if 0 <= index < array.shape[1]:
            return np.delete(array, index, axis=1)

    return _to_2d_float_array(values)


def _append_features(dense: np.ndarray, neural_features: np.ndarray) -> np.ndarray:
    if dense.shape[0] != neural_features.shape[0]:
        raise ValueError("X and neural features must have the same row count")
    return np.hstack([dense, np.asarray(neural_features, dtype=np.float64)])


def _build_neural_feature_names(
    columns: list[str] | None,
    base_feature_count: int,
    dim: int,
) -> np.ndarray | None:
    if columns is None:
        return None

    base_names = list(columns)

    if len(base_names) != base_feature_count:
        return None

    neural_names = [f"neural_embedding_{index:02}" for index in range(dim)]
    return np.array([*base_names, *neural_names], dtype=object)


def _ms_since(start_time: float) -> float:
    return float((time.perf_counter() - start_time) * 1000.0)
