from pathlib import Path

import pytest
from geoboost import GeoBoostRegressor


def test_get_params_and_set_params_reset_model():
    regressor = GeoBoostRegressor(n_estimators=3, backend="python")
    assert regressor.get_params()["n_estimators"] == 3

    returned = regressor.set_params(learning_rate=0.2)

    assert returned is regressor
    assert regressor.learning_rate == 0.2


def test_fit_predict_and_roundtrip_python_backend(tmp_path: Path):
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    regressor = GeoBoostRegressor(
        n_estimators=8,
        learning_rate=0.4,
        min_samples_leaf=1,
        backend="python",
    )

    regressor.fit(X, y)
    predictions = regressor.predict([[0.0], [3.0]])

    assert predictions[0] < predictions[1]

    model_path = tmp_path / "model.json"
    regressor.save(model_path)
    loaded = GeoBoostRegressor.load(model_path)

    assert loaded.predict([[0.0], [3.0]]) == pytest.approx(predictions)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        GeoBoostRegressor(backend="python").predict([[1.0]])


def test_rust_backend_accepts_special_splitters():
    regressor = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        splitters=["diagonal_2d"],
        backend="rust",
    )
    try:
        regressor.fit(
            [[-2.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 1.0]], [-10.0, -10.0, 10.0, 10.0]
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    assert regressor.predict([[-2.0, -1.0], [2.0, 1.0]]) == pytest.approx([-10.0, 10.0])


def test_rust_backend_accepts_linear_fuzzy_and_sparse_options():
    linear = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=3,
        leaf_predictor="linear",
        linear_leaf_features=["0"],
        l2_regularization=0.0,
        backend="rust",
    )
    sparse = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        splitters=["sparse_set"],
        backend="rust",
    )
    fuzzy = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        fuzzy=True,
        fuzzy_bandwidth=1.0,
        backend="rust",
    )

    try:
        linear.fit([[0.0], [1.0], [2.0], [3.0]], [3.0, 5.0, 7.0, 9.0])
        sparse.fit([[7.0], [7.0], [3.0], [4.0]], [25.0, 25.0, -5.0, -5.0])
        fuzzy.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 10.0, 10.0])
    except ImportError as exc:
        pytest.skip(str(exc))

    assert linear.predict([[0.0], [3.0]]) == pytest.approx([4.5, 7.5])
    assert sparse.predict([[7.0], [3.0]]) == pytest.approx([25.0, -5.0])
    assert fuzzy.predict([[1.5]]) == pytest.approx([5.0])
