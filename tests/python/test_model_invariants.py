from __future__ import annotations

import json

import numpy as np
import pytest
from geoboost import GeoBoostRegressor
from geoboost import regressor as regressor_module


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


def test_real_native_save_load_restores_public_params_and_metadata(tmp_path):
    x = np.array([[0.0, 0.0], [0.2, 0.0], [3.0, 0.0], [0.0, 3.0]])
    y = np.array([5.0, 5.0, -1.0, -1.0])
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.25,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["gaussian_2d"],
        fuzzy=True,
        fuzzy_bandwidth=0.5,
        leaf_predictor="constant",
        backend="rust",
    )

    try:
        model.fit(x, y)
    except ImportError as exc:
        pytest.skip(str(exc))

    path = tmp_path / "native-geoboost.json"
    before = model.predict(x)
    model.save(path)

    restored = GeoBoostRegressor.load(path)

    assert restored.get_params() == {
        "n_estimators": 2,
        "learning_rate": 0.25,
        "max_depth": 1,
        "min_samples_leaf": 1,
        "min_gain": 0.0,
        "loss": "l2",
        "quantile_alpha": 0.5,
        "splitters": ["gaussian_2d"],
        "leaf_predictor": "constant",
        "linear_leaf_features": [],
        "fuzzy": True,
        "fuzzy_bandwidth": 0.5,
        "l2_regularization": 1.0,
        "random_state": None,
        "n_threads": None,
        "backend": "auto",
        "monotonic_constraints": None,
    }
    assert restored.metadata_["library_name"] == "geoboost-core"
    assert restored.training_config_["splitters"] == ["Gaussian2D"]
    assert restored.requires_sparse_sets_ is False
    assert restored.predict(x) == pytest.approx(before)


def test_real_native_save_weights_load_weights_restores_predictions(tmp_path):
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 5.0, 5.0])
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        backend="rust",
    )

    try:
        model.fit(x, y)
    except ImportError as exc:
        pytest.skip(str(exc))

    path = tmp_path / "native.weights.json"
    before = model.predict(x)
    model.save_weights(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    restored = GeoBoostRegressor.load_weights(path)

    assert payload["artifact_type"] == "geoboost.weights"
    assert payload["weights_artifact_version"] == 1
    assert payload["backend"] == "rust"
    assert restored.predict(x) == pytest.approx(before)


def test_native_sparse_list_prediction_requires_sparse_sets_after_load(tmp_path):
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [7.0, 7.0, -2.0, -2.0]
    sparse_sets = {"route_cells": [[10, 20], [20, 30], [40], []]}
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        splitters=["sparse_set"],
        backend="rust",
    )

    try:
        model.fit(x, y, sparse_sets=sparse_sets)
    except ImportError as exc:
        pytest.skip(str(exc))

    path = tmp_path / "native-sparse-geoboost.json"
    model.save(path)
    restored = GeoBoostRegressor.load(path)

    assert restored.requires_sparse_sets_ is True
    with pytest.raises(ValueError, match="sparse_sets are required"):
        restored.predict(x)
    assert restored.predict(x, sparse_sets=sparse_sets) == pytest.approx(y)


def test_native_artifact_version_mismatch_errors_clearly(tmp_path):
    if regressor_module._NativeGeoBoostRegressor is None:
        pytest.skip("native extension is not installed")

    path = tmp_path / "future-native-artifact.json"
    path.write_text(
        json.dumps(
            {
                "artifact_version": 999,
                "init_prediction": 0.0,
                "learning_rate": 0.1,
                "feature_count": 1,
                "target_name": None,
                "trees": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported model artifact version 999"):
        GeoBoostRegressor.load(path)


def test_python_api_rejects_unsupported_objectives_and_backends():
    with pytest.raises(ValueError, match="loss"):
        GeoBoostRegressor(loss="l1").fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="quantile_alpha"):
        GeoBoostRegressor(loss="quantile", quantile_alpha=1.0).fit([[0.0], [1.0]], [0.0, 1.0])
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
