from __future__ import annotations

import math

import numpy as np
import pytest
from cartoboost.metrics.wrmsse import rmsse_scale, wrmsse


def test_wrmsse_matches_manual_m5_style_reference() -> None:
    y_train = np.array(
        [
            [10.0, 12.0, 14.0, 16.0],
            [20.0, 21.0, 23.0, 26.0],
        ]
    )
    y_true = np.array(
        [
            [18.0, 20.0],
            [30.0, 34.0],
        ]
    )
    y_pred = np.array(
        [
            [17.0, 23.0],
            [32.0, 31.0],
        ]
    )
    weights = np.array([2.0, 1.0])

    result = wrmsse(
        y_train,
        y_true,
        y_pred,
        weights,
        seasonal_period=1,
        series_ids=["PULocationID=1", "PULocationID=2"],
        return_breakdown=True,
    )

    first_rmsse = math.sqrt((1.0 + 9.0) / 2.0 / 4.0)
    second_rmsse = math.sqrt((4.0 + 9.0) / 2.0 / (14.0 / 3.0))
    expected = (2.0 / 3.0) * first_rmsse + (1.0 / 3.0) * second_rmsse
    assert result["wrmsse"] == pytest.approx(expected)
    assert result["wrmsse"] == pytest.approx(1.1387538887346516)
    assert result["series"][0]["series_id"] == "PULocationID=1"
    assert result["series"][0]["scale"] == pytest.approx(4.0)
    assert result["series"][1]["scale"] == pytest.approx(14.0 / 3.0)


def test_wrmsse_returns_scalar_by_default() -> None:
    score = wrmsse(
        [[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]],
        [[4.0], [9.0]],
        [[5.0], [8.0]],
        [1.0, 3.0],
    )

    assert isinstance(score, float)
    assert score > 0.0


def test_rmsse_scale_rejects_constant_or_short_history() -> None:
    with pytest.raises(ValueError, match="longer than seasonal_period"):
        rmsse_scale([1.0], seasonal_period=1)

    with pytest.raises(ValueError, match="scale is zero"):
        rmsse_scale([2.0, 2.0, 2.0], seasonal_period=1)


def test_wrmsse_rejects_invalid_shapes_and_weights() -> None:
    with pytest.raises(ValueError, match="same shape"):
        wrmsse([[1.0, 2.0]], [[3.0, 4.0]], [[3.0]], [1.0])

    with pytest.raises(ValueError, match="same number of series"):
        wrmsse([[1.0, 2.0]], [[3.0], [4.0]], [[3.0], [4.0]], [1.0, 1.0])

    with pytest.raises(ValueError, match="non-negative"):
        wrmsse([[1.0, 2.0]], [[3.0]], [[4.0]], [-1.0])

    with pytest.raises(ValueError, match="positive value"):
        wrmsse([[1.0, 2.0]], [[3.0]], [[4.0]], [0.0])
