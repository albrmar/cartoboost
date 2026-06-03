from pathlib import Path

import numpy as np
import pytest
from geoboost import GeoBoostRegressor, make_shap_explainer

shap = pytest.importorskip("shap")


def _assert_additive(explanation, prediction):
    reconstructed = np.asarray(explanation.base_values) + explanation.values.sum(axis=1)
    assert reconstructed == pytest.approx(prediction)


def _fit_or_skip(model, *args, **kwargs):
    try:
        return model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))


def test_shap_explainer_accepts_geoboost_estimator_directly():
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    model = GeoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        min_samples_leaf=1,
        backend="python",
    ).fit(X, y)

    explainer = shap.Explainer(model, X, algorithm="exact")
    explanation = explainer(X[:2])

    assert isinstance(explanation, shap.Explanation)
    assert explanation.values.shape == (2, 1)
    _assert_additive(explanation, model.predict(X[:2]))


def test_explain_shap_returns_additive_explanation_for_python_backend():
    X = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    y = np.asarray([0.0, 0.5, 2.0, 2.5, 4.0, 4.5], dtype=float)
    model = GeoBoostRegressor(
        n_estimators=4,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        backend="python",
    ).fit(X, y)

    explanation = model.explain_shap(X[:3], background=X, algorithm="exact")

    assert explanation.values.shape == (3, 2)
    assert explanation.data.shape == (3, 2)
    _assert_additive(explanation, model.predict(X[:3]))


def test_make_shap_explainer_public_helper_matches_method():
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    model = GeoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        min_samples_leaf=1,
        backend="python",
    ).fit(X, y)

    helper_explainer = make_shap_explainer(model, X, algorithm="exact")
    method_explainer = model.make_shap_explainer(X, algorithm="exact")

    helper_explanation = helper_explainer(X[:2])
    method_explanation = method_explainer(X[:2])

    assert helper_explanation.values == pytest.approx(method_explanation.values)
    assert helper_explanation.base_values == pytest.approx(method_explanation.base_values)


def test_shap_preserves_pandas_feature_names():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"distance_m": [0.0, 1.0, 2.0, 3.0], "hour": [0.0, 1.0, 0.0, 1.0]})
    y = np.asarray([0.0, 1.5, 2.0, 3.5], dtype=float)
    model = GeoBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        min_samples_leaf=1,
        backend="python",
    ).fit(X, y)

    explanation = model.explain_shap(X.iloc[:2], background=X, algorithm="exact")

    assert list(explanation.feature_names) == ["distance_m", "hour"]
    _assert_additive(explanation, model.predict(X.iloc[:2]))


@pytest.mark.parametrize(
    ("splitters", "X", "y", "extra"),
    [
        (["axis"], [[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 10.0, 10.0], {}),
        (
            ["diagonal_2d"],
            [[-2.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 1.0]],
            [-10.0, -10.0, 10.0, 10.0],
            {},
        ),
        (
            ["gaussian_2d"],
            [[0.0, 0.0], [0.2, 0.1], [3.0, 3.0], [3.2, 3.1]],
            [5.0, 5.0, -5.0, -5.0],
            {},
        ),
        (
            ["periodic:24"],
            [[0.0], [1.0], [12.0], [13.0]],
            [10.0, 10.0, 0.0, 0.0],
            {"feature_schema": {"dense": [{"name": "hour", "kind": "periodic", "period": 24}]}},
        ),
    ],
)
def test_shap_additivity_for_rust_dense_splitters(splitters, X, y, extra):
    rows = np.asarray(X, dtype=float)
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        splitters=splitters,
        backend="rust",
    )
    _fit_or_skip(model, rows, y, **extra)

    explanation = model.explain_shap(rows[:2], background=rows, algorithm="exact")

    assert isinstance(explanation, shap.Explanation)
    _assert_additive(explanation, model.predict(rows[:2]))


@pytest.mark.parametrize(
    "params",
    [
        {"fuzzy": True, "fuzzy_bandwidth": 1.0},
        {"leaf_predictor": "linear", "linear_leaf_features": ["0"], "l2_regularization": 0.0},
    ],
)
def test_shap_additivity_for_rust_fuzzy_and_linear_leaf(params):
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float)
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        backend="rust",
        **params,
    )
    _fit_or_skip(model, X, y)

    explanation = model.explain_shap(X[:2], background=X, algorithm="exact")

    _assert_additive(explanation, model.predict(X[:2]))


def test_shap_additivity_after_save_load(tmp_path: Path):
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        min_samples_leaf=1,
        backend="rust",
    )
    _fit_or_skip(model, X, y)
    path = tmp_path / "model.geoboost.json"
    model.save(path)
    loaded = GeoBoostRegressor.load(path)

    explanation = loaded.explain_shap(X[:2], background=X, algorithm="exact")

    _assert_additive(explanation, loaded.predict(X[:2]))


def test_shap_supports_sparse_set_models_with_augmented_features():
    X = np.asarray([[0.0], [0.0], [0.0], [0.0]], dtype=float)
    y = np.asarray([10.0, 10.0, 0.0, 0.0], dtype=float)
    sparse_sets = {"route_cells": [[7], [7, 11], [3], []]}
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        splitters=["sparse_set"],
        backend="rust",
    )
    _fit_or_skip(model, X, y, sparse_sets=sparse_sets)

    explanation = model.explain_shap(
        X[:2],
        background=X,
        sparse_sets={"route_cells": sparse_sets["route_cells"][:2]},
        background_sparse_sets=sparse_sets,
        algorithm="exact",
    )

    assert list(explanation.feature_names) == [
        "feature_0",
        "route_cells=3",
        "route_cells=7",
        "route_cells=11",
    ]
    assert explanation.values.shape == (2, 4)
    _assert_additive(
        explanation,
        model.predict(X[:2], sparse_sets={"route_cells": sparse_sets["route_cells"][:2]}),
    )
