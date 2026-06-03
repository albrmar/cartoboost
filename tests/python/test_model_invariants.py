from __future__ import annotations

import numpy as np
import pytest
from geoboost import GeoBoostRegressor


def _radial_fixture() -> tuple[np.ndarray, np.ndarray]:
    values = np.linspace(-3.0, 3.0, 17)
    xx, yy = np.meshgrid(values, values)
    x = np.column_stack([xx.ravel(), yy.ravel()])
    y = np.where(np.hypot(x[:, 0], x[:, 1]) <= 1.5, 10.0, -10.0)
    return x, y


def test_native_fuzzy_gaussian_serialization_preserves_predictions(tmp_path):
    x, y = _radial_fixture()
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.75,
        max_depth=1,
        min_samples_leaf=4,
        min_gain=0.0,
        splitters=["gaussian_2d"],
        fuzzy=True,
        fuzzy_bandwidth=0.5,
        backend="rust",
    )

    try:
        model.fit(x, y)
    except ImportError as exc:
        pytest.skip(str(exc))

    probes = np.array(
        [
            [0.0, 0.0],
            [1.45, 0.0],
            [1.55, 0.0],
            [3.0, 3.0],
        ]
    )
    before = model.predict(probes)
    model_path = tmp_path / "fuzzy_gaussian.geoboost"
    model.save(model_path)

    restored = GeoBoostRegressor.load(model_path)

    assert restored.predict(probes) == pytest.approx(before)
    assert restored.n_features_in_ == 2


def test_python_api_rejects_unsupported_objectives_and_backends():
    with pytest.raises(NotImplementedError, match="loss='l2'"):
        GeoBoostRegressor(loss="l1").fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="backend"):
        GeoBoostRegressor(backend="cuda").fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="leaf_predictor"):
        GeoBoostRegressor(leaf_predictor="spline").fit([[0.0], [1.0]], [0.0, 1.0])


def test_python_api_rejects_invalid_training_arrays():
    model = GeoBoostRegressor(backend="python")

    with pytest.raises(ValueError, match="same number of rows"):
        model.fit([[0.0], [1.0]], [0.0])
    with pytest.raises(ValueError, match="rectangular"):
        model.fit([[0.0], [1.0, 2.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        model.fit([[0.0], [float("nan")]], [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        model.fit([[0.0], [1.0]], [0.0, float("inf")])


def test_python_fallback_rejects_native_only_features():
    with pytest.raises(NotImplementedError, match="pure-Python fallback"):
        GeoBoostRegressor(splitters=["gaussian_2d"], backend="python").fit(
            [[0.0, 0.0], [1.0, 1.0]],
            [0.0, 1.0],
        )
    with pytest.raises(NotImplementedError, match="pure-Python fallback"):
        GeoBoostRegressor(fuzzy=True, fuzzy_bandwidth=1.0, backend="python").fit(
            [[0.0], [1.0]],
            [0.0, 1.0],
        )


def test_linear_leaf_feature_indices_are_validated():
    with pytest.raises(ValueError, match="stringified integer"):
        GeoBoostRegressor(
            leaf_predictor="linear",
            linear_leaf_features=["distance"],
            backend="rust",
        ).fit([[0.0], [1.0]], [0.0, 1.0])

    with pytest.raises(ValueError, match="out of bounds"):
        GeoBoostRegressor(
            leaf_predictor="linear",
            linear_leaf_features=["2"],
            backend="rust",
        ).fit([[0.0], [1.0]], [0.0, 1.0])
