from __future__ import annotations

import numpy as np
import pytest
from geoboost import GeoBoostRegressor


def test_huber_clips_outlier_influence_against_l2_native_backend() -> None:
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0.0, 0.1, 0.2, 0.3, 10.0, 10.1, 10.2, 1000.0])

    l2 = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=2,
        loss="l2",
    ).fit(X, y)
    huber = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=2,
        loss="huber",
        loss_params={"delta": 1.0},
    ).fit(X, y)

    assert huber.predict([[6.0]])[0] < l2.predict([[6.0]])[0]


def test_log_l2_rejects_invalid_targets_native_backend() -> None:
    X = np.arange(4, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError, match="log_l2 targets"):
        GeoBoostRegressor(loss="log_l2").fit(X, [-1.0, 0.0, 1.0, 2.0])


def test_log_l2_predictions_return_original_scale_native_backend() -> None:
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.array([0.0, 1.0, 3.0, 7.0])

    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=0,
        loss="log_l2",
    ).fit(X, y)

    prediction = model.predict([[0.0]])[0]
    assert prediction == pytest.approx(np.expm1(np.log1p(y).mean()))


def test_loss_params_override_objective_defaults() -> None:
    model = GeoBoostRegressor(loss="huber", loss_params={"delta": 2.5})

    assert model.get_params()["loss_params"] == {"delta": 2.5}
    model.set_params(loss_params={"delta": 1.5})
    assert model.loss_params == {"delta": 1.5}
