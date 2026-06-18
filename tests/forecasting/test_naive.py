import pytest
from cartoboost.forecasting.local import NaiveForecaster


def test_naive_requires_fit_before_predict():
    with pytest.raises(RuntimeError, match="fitted"):
        NaiveForecaster().predict(2)


def test_naive_converts_series_and_delegates_to_native(install_fake_native):
    native = install_fake_native("NaiveForecaster")

    result = NaiveForecaster(prediction_interval_levels=[0.8]).fit([1.0, 2.0, 4.0]).predict(2)

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == ("init", {"prediction_interval_levels": (0.8,)})
    assert native.calls[1][0] == "fit"
    assert isinstance(native.calls[1][1], native.frame_class)
    assert native.calls[1][1].rows == [
        ("__single__", "1970-01-01T00:00:00", 1.0),
        ("__single__", "1970-01-02T00:00:00", 2.0),
        ("__single__", "1970-01-03T00:00:00", 4.0),
    ]
    assert native.calls[2] == ("predict", (2,), {})
