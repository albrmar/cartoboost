from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from cartoboost.forecasting.probabilistic import (
    ConformalCalibrator,
    pinball_loss,
    rank_probability_score,
    repair_non_crossing_quantiles,
)


def _load_m6_module():
    path = Path(__file__).resolve().parents[2] / "python" / "cartoboost" / "metrics" / "m6.py"
    spec = importlib.util.spec_from_file_location("cartoboost_metrics_m6", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probabilistic_helpers_repair_score_and_calibrate() -> None:
    assert repair_non_crossing_quantiles([12.0, 10.0, 13.0]).tolist() == [12.0, 12.0, 13.0]
    assert pinball_loss([10.0, 20.0], [11.0, 18.0], 0.5) == pytest.approx(0.75)
    assert rank_probability_score([0.2, 0.3, 0.5], 2) == pytest.approx(0.145)

    calibrator = ConformalCalibrator(alpha=0.5).fit(
        [10.0, 20.0],
        [11.0, 18.0],
        train_end_exclusive=4,
        calibration_start=4,
        calibration_end_exclusive=6,
        test_start=6,
    )
    interval = calibrator.predict_interval([30.0], test_start=6)
    assert interval.residual_quantile == pytest.approx(2.0)
    assert interval.lower.tolist() == [28.0]
    assert interval.upper.tolist() == [32.0]


def test_conformal_rejects_overlapping_train_calibration_test_inputs() -> None:
    with pytest.raises(ValueError, match="training rows must end"):
        ConformalCalibrator(alpha=0.1).fit(
            [1.0],
            [1.0],
            train_end_exclusive=5,
            calibration_start=4,
            calibration_end_exclusive=6,
            test_start=6,
        )


def test_m6_metric_helper_combines_pinball_and_rps() -> None:
    m6 = _load_m6_module()

    summary = m6.evaluate_m6_metrics(
        [10.0, 20.0],
        [11.0, 18.0],
        0.5,
        [0.2, 0.3, 0.5],
        2,
    )

    assert summary.pinball_loss == pytest.approx(0.75)
    assert summary.rank_probability_score == pytest.approx(0.145)
    assert summary.combined_score == pytest.approx(0.4475)
