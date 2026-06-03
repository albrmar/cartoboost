import pytest
from geoboost import GeoBoostRegressor


def test_zero_depth_predicts_constant_mean_python_backend():
    model = GeoBoostRegressor(max_depth=0, backend="python")

    model.fit([[0.0], [1.0], [2.0]], [3.0, 6.0, 9.0])

    assert model.predict([[0.0], [10.0]]) == pytest.approx([6.0, 6.0])


def test_zero_depth_roundtrip_preserves_constant_model(tmp_path):
    model = GeoBoostRegressor(max_depth=0, backend="python")
    model.fit([[0.0], [1.0]], [2.0, 8.0])
    model_path = tmp_path / "zero_depth.json"

    model.save(model_path)
    restored = GeoBoostRegressor.load(model_path)

    assert restored.max_depth == 0
    assert restored.predict([[0.0], [1.0]]) == pytest.approx([5.0, 5.0])


def test_zero_depth_constant_model_accepts_valid_native_only_splitters():
    model = GeoBoostRegressor(
        max_depth=0,
        splitters=["gaussian_2d"],
        fuzzy=True,
        fuzzy_bandwidth=1.0,
        backend="python",
    )

    model.fit([[0.0, 0.0], [1.0, 1.0]], [2.0, 6.0])

    assert model._backend_used == "python"
    assert model.predict([[10.0, 10.0]]) == pytest.approx([4.0])


def test_negative_max_depth_is_rejected():
    model = GeoBoostRegressor(max_depth=-1, backend="python")

    with pytest.raises(ValueError, match="non-negative"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])
