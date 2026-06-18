import numpy as np
import pytest
from cartoboost.forecasting.local import NaiveForecaster


def test_naive_requires_fit_before_predict():
    with pytest.raises(RuntimeError, match="fitted"):
        NaiveForecaster().predict(2)


def test_naive_predicts_last_value_with_intervals_and_timestamps():
    timestamps = np.array(
        ["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[D]"
    )
    model = NaiveForecaster().fit([1.0, 2.0, 4.0], timestamps=timestamps)

    np.testing.assert_allclose(model.predict(3), [4.0, 4.0, 4.0])
    result = model.predict(2, return_interval=True, level=0.8)

    np.testing.assert_allclose(result.mean, [4.0, 4.0])
    assert result.lower.shape == (2,)
    assert result.upper.shape == (2,)
    assert result.metadata["n_obs"] == 3
    np.testing.assert_array_equal(
        result.timestamps, np.array(["2026-01-04", "2026-01-05"], dtype="datetime64[D]")
    )


def test_naive_supports_panel_dicts():
    model = NaiveForecaster().fit({"pickup_1": [1.0, 3.0], "pickup_2": [2.0, 5.0]})

    np.testing.assert_allclose(model.predict(2), [[3.0, 5.0], [3.0, 5.0]])
    assert model.metadata_["series_ids"] == ["pickup_1", "pickup_2"]
