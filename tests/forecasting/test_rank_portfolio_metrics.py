from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from cartoboost import _native
from cartoboost.forecasting.probabilistic import (
    ConformalCalibrator,
    pinball_loss,
    rank_probability_score,
    repair_non_crossing_quantiles,
)


def _load_rank_portfolio_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "cartoboost"
        / "metrics"
        / "rank_portfolio.py"
    )
    spec = importlib.util.spec_from_file_location("cartoboost_metrics_rank_portfolio", path)
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


def test_rank_portfolio_metric_helper_combines_pinball_and_rps() -> None:
    rank_portfolio = _load_rank_portfolio_module()

    summary = rank_portfolio.evaluate_rank_portfolio_metrics(
        [10.0, 20.0],
        [11.0, 18.0],
        0.5,
        [0.2, 0.3, 0.5],
        2,
    )

    assert summary.pinball_loss == pytest.approx(0.75)
    assert summary.rank_probability_score == pytest.approx(0.145)
    assert summary.combined_score == pytest.approx(0.4475)


def test_rank_portfolio_portfolio_and_rank_diagnostics_are_native_backed() -> None:
    rank_portfolio = _load_rank_portfolio_module()

    portfolio = rank_portfolio.portfolio_summary(
        [
            {
                "side": "short",
                "weight": -0.25,
                "actual_return": -0.04,
                "predicted_return": -0.02,
            },
            {
                "side": "short",
                "weight": -0.25,
                "actual_return": 0.02,
                "predicted_return": -0.01,
            },
            {
                "side": "long",
                "weight": 0.5,
                "actual_return": 0.06,
                "predicted_return": 0.03,
            },
        ]
    )
    hit_rates = rank_portfolio.rank_hit_rates(
        [
            {"observed_rank_bucket": 0, "predicted_rank_bucket": 0},
            {"observed_rank_bucket": 2, "predicted_rank_bucket": 1},
            {"observed_rank_bucket": 4, "predicted_rank_bucket": 0},
        ],
        bucket_count=5,
    )

    assert portfolio["long_count"] == 1
    assert portfolio["short_count"] == 2
    assert portfolio["gross_exposure"] == pytest.approx(1.0)
    assert portfolio["net_exposure"] == pytest.approx(0.0)
    assert portfolio["net_return"] == pytest.approx(0.035)
    assert hit_rates["asset_count"] == 3
    assert hit_rates["exact_bucket_rate"] == pytest.approx(1.0 / 3.0)
    assert hit_rates["within_one_bucket_rate"] == pytest.approx(2.0 / 3.0)
    assert hit_rates["directional_extreme_count"] == 2
    assert hit_rates["directional_extreme_rate"] == pytest.approx(0.5)


def test_extreme_portfolio_decisions_are_native_backed() -> None:
    decisions = _native.extreme_portfolio_decisions_value(
        [
            ("a", -0.03, -0.02),
            ("b", 0.01, -0.01),
            ("c", 0.02, 0.00),
            ("d", 0.03, 0.01),
            ("e", 0.04, 0.02),
            ("f", 0.05, 0.03),
            ("g", -0.01, -0.03),
            ("h", 0.06, 0.04),
            ("i", 0.07, 0.05),
            ("j", 0.08, 0.06),
        ]
    )

    assert decisions == [
        ("i", "long", 0.25, 0.07, 0.05),
        ("j", "long", 0.25, 0.08, 0.06),
        ("a", "short", -0.25, -0.03, -0.02),
        ("g", "short", -0.25, -0.01, -0.03),
    ]


def test_rank_buckets_are_native_backed() -> None:
    assert _native.rank_buckets_value([2.0, 1.0, 1.0, 5.0, 4.0], 5) == [2, 0, 1, 4, 3]
    assert _native.rank_buckets_value([10.0, 20.0, 30.0, 40.0], 2) == [0, 0, 1, 1]


def test_rank_scored_assets_are_native_backed() -> None:
    rows = json.loads(
        _native.rank_scored_assets_value(
            [
                ("a", -0.03, -0.02),
                ("b", 0.01, -0.01),
                ("c", 0.04, 0.03),
            ],
            3,
            [[1.0 / 3.0] * 3] * 3,
            0.0,
        )
    )

    assert [row["series_id"] for row in rows] == ["a", "b", "c"]
    assert [row["observed_rank_bucket"] for row in rows] == [0, 1, 2]
    assert [row["predicted_rank_bucket"] for row in rows] == [0, 1, 2]
    assert rows[0]["rank_probabilities"] == pytest.approx([1.0 / 3.0] * 3)
    assert rows[0]["rps"] == pytest.approx(5.0 / 18.0)
    assert rows[1]["rps"] == pytest.approx(1.0 / 9.0)
    assert rows[2]["rps"] == pytest.approx(5.0 / 18.0)


def test_rank_portfolio_summary_is_native_backed() -> None:
    summary = json.loads(
        _native.rank_portfolio_summary_value(
            [
                ("a", -0.03, -0.02),
                ("b", 0.01, -0.01),
                ("c", 0.04, 0.03),
            ],
            3,
            [[1.0 / 3.0] * 3] * 3,
            0.0,
        )
    )

    assert summary["asset_count"] == 3
    assert summary["mean_rps"] == pytest.approx(2.0 / 9.0)
    assert len(summary["decisions"]) == 2
    assert summary["portfolio"]["long_count"] == 1
    assert summary["portfolio"]["short_count"] == 1
    assert summary["rank_hit_rates"]["exact_bucket_rate"] == pytest.approx(1.0)


def test_rank_portfolio_decision_loss_is_native_backed() -> None:
    loss = _native.rank_portfolio_decision_loss_value(
        [
            ("a", -0.05, -0.04),
            ("b", 0.00, 0.01),
            ("c", 0.05, 0.04),
        ],
        3,
        [[1.0 / 3.0] * 3] * 3,
        0.0,
        1.0e-4,
    )

    assert loss < 0.0
    assert (
        _native.rank_portfolio_decision_loss_value(
            [
                ("a", -0.05, 0.04),
                ("b", 0.00, 0.01),
                ("c", 0.05, -0.04),
            ],
            3,
            [[1.0 / 3.0] * 3] * 3,
            0.0,
            1.0e-4,
        )
        > loss
    )


def test_rank_probability_calibration_is_native_backed() -> None:
    rank_portfolio = _load_rank_portfolio_module()

    calibration = rank_portfolio.rank_probability_calibration(
        [0, 1, 2, 2],
        [0, 1, 1, 2],
        bucket_count=3,
        validation_support=4,
    )
    probabilities = rank_portfolio.calibrated_rank_bucket_probabilities(
        1,
        bucket_count=3,
        calibration=calibration,
    )

    assert calibration["metadata"]["bucket_count"] == 3
    assert calibration["metadata"]["validation_support"] == 4
    assert calibration["metadata"]["fallback"] == "none"
    assert calibration["shrinkage"] == pytest.approx(4.0 / 64.0)
    assert calibration["probabilities"][1][2] == pytest.approx(0.4)
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[2] > probabilities[0]
