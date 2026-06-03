from __future__ import annotations

import math
import tempfile
from pathlib import Path

import geoboost.regressor as regressor_module
import pytest
from geoboost import GeoBoostRegressor
from hypothesis import given, settings
from hypothesis import strategies as st


@st.composite
def small_regression_cases(draw):
    rows = draw(st.integers(min_value=2, max_value=8))
    cols = draw(st.integers(min_value=1, max_value=3))
    value = st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    flat_x = draw(st.lists(value, min_size=rows * cols, max_size=rows * cols))
    y = draw(st.lists(value, min_size=rows, max_size=rows))
    x = [flat_x[row * cols : (row + 1) * cols] for row in range(rows)]
    return x, y


@settings(max_examples=12, deadline=None)
@given(small_regression_cases())
def test_python_backend_predictions_are_finite(case):
    x, y = case
    model = GeoBoostRegressor(
        n_estimators=3,
        learning_rate=0.25,
        max_depth=2,
        min_samples_leaf=1,
        min_gain=0.0,
        backend="python",
    )

    model.fit(x, y)
    predictions = model.predict(x)

    assert len(predictions) == len(x)
    assert all(math.isfinite(float(prediction)) for prediction in predictions)


@settings(max_examples=8, deadline=None)
@given(small_regression_cases())
def test_python_backend_save_load_preserves_predictions(case):
    x, y = case
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        backend="python",
    )

    model.fit(x, y)
    before = model.predict(x)
    native_cls = regressor_module._NativeGeoBoostRegressor
    try:
        regressor_module._NativeGeoBoostRegressor = None
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.json"
            model.save(path)

            restored = GeoBoostRegressor.load(path)
    finally:
        regressor_module._NativeGeoBoostRegressor = native_cls

    assert restored.predict(x) == pytest.approx(before)
