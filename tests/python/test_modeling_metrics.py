import numpy as np
import pytest
from cartoboost import (
    calibrated_intervals,
    conformal_residual_quantile,
    interval_coverage,
    jitter_volatility,
    mean_interval_width,
    pinball_loss,
    residual_morans_i,
)


def test_conformal_residual_quantile_builds_calibrated_intervals():
    y_true = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    y_pred = np.array([9.0, 13.5, 13.0, 15.5, 21.0])

    quantile = conformal_residual_quantile(y_true, y_pred, alpha=0.5)
    lower, upper = calibrated_intervals([11.0, 20.0], quantile)

    assert quantile == pytest.approx(1.5)
    assert lower == pytest.approx([9.5, 18.5])
    assert upper == pytest.approx([12.5, 21.5])


def test_calibrated_intervals_can_compute_quantile_from_calibration_data():
    lower, upper = calibrated_intervals(
        [5.0],
        y_calibration=[0.0, 2.0, 4.0],
        calibration_predictions=[0.0, 1.0, 7.0],
        alpha=0.5,
    )

    assert lower == pytest.approx([2.0])
    assert upper == pytest.approx([8.0])


def test_pinball_interval_coverage_and_width_metrics():
    y_true = np.array([0.0, 2.0, 4.0, 8.0])
    y_pred = np.array([1.0, 1.0, 5.0, 6.0])
    lower = np.array([-1.0, 1.0, 4.5, 5.5])
    upper = np.array([1.0, 3.0, 7.5, 7.0])

    assert pinball_loss(y_true, y_pred, quantile=0.8) == pytest.approx(0.7)
    assert interval_coverage(y_true, lower, upper) == pytest.approx(0.5)
    assert mean_interval_width(lower, upper) == pytest.approx(2.125)


def test_jitter_volatility_uses_per_sample_instability():
    predictions = np.array(
        [
            [10.0, 20.0],
            [12.0, 18.0],
            [14.0, 22.0],
        ]
    )

    assert jitter_volatility(predictions) == pytest.approx(np.mean(np.std(predictions, axis=0)))
    assert jitter_volatility(predictions, baseline=[12.0, 20.0]) == pytest.approx(
        np.mean([np.sqrt(8.0 / 3.0), np.sqrt(8.0 / 3.0)])
    )


def test_residual_morans_i_supports_inverse_distance_and_radius_weights():
    coordinates = np.array([[0.0], [1.0], [2.0], [3.0]])
    clustered_residuals = np.array([1.0, 1.0, -1.0, -1.0])

    inverse_i = residual_morans_i(
        coordinates,
        clustered_residuals,
        weights="inverse_distance",
    )
    radius_i = residual_morans_i(
        coordinates,
        clustered_residuals,
        weights="radius",
        radius=1.1,
    )

    assert inverse_i == pytest.approx(-1.0 / 13.0)
    assert radius_i == pytest.approx(1.0 / 3.0)


def test_metric_validation_rejects_bad_shapes_and_weights():
    with pytest.raises(ValueError, match="same shape"):
        pinball_loss([1.0], [1.0, 2.0], quantile=0.5)
    with pytest.raises(ValueError, match="lower bounds"):
        interval_coverage([1.0], [2.0], [0.0])
    with pytest.raises(ValueError, match="at least two jitter repeats"):
        jitter_volatility([[1.0, 2.0]])
    with pytest.raises(ValueError, match="positive value"):
        residual_morans_i([[0.0], [1.0]], [1.0, -1.0], weights="radius")
