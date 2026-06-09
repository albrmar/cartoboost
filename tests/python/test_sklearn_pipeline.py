from __future__ import annotations

import numpy as np
import optuna
from geoboost import GeoBoostRegressor
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def test_estimator_clones_and_exposes_fitted_metadata():
    model = GeoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
    )

    cloned = clone(model)
    assert cloned.get_params()["n_estimators"] == 2

    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    cloned.fit(x, y)

    pred = cloned.predict(x)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == y.shape
    assert cloned.n_features_in_ == 1


def test_pipeline_and_grid_search_work_with_sklearn():
    x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    y = np.array([-5.0, -5.0, -5.0, 5.0, 5.0, 5.0])
    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                GeoBoostRegressor(
                    n_estimators=3,
                    max_depth=1,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )

    search = GridSearchCV(
        pipeline,
        param_grid={"model__learning_rate": [0.25, 0.5]},
        cv=2,
        scoring="neg_mean_squared_error",
    )
    search.fit(x, y)

    pred = search.predict(x)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
    assert "model__learning_rate" in search.best_params_


def test_optuna_study_can_tune_geoboost_regressor():
    x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    y = np.array([-5.0, -5.0, -5.0, 5.0, 5.0, 5.0])

    def objective(trial: optuna.Trial) -> float:
        model = GeoBoostRegressor(
            n_estimators=trial.suggest_int("n_estimators", 1, 4),
            learning_rate=trial.suggest_float("learning_rate", 0.1, 0.5),
            max_depth=trial.suggest_int("max_depth", 1, 2),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 2),
        )
        scores = cross_val_score(
            model,
            x,
            y,
            cv=2,
            scoring="neg_mean_squared_error",
        )
        return float(-scores.mean())

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    study.optimize(objective, n_trials=2)

    assert set(study.best_params) == {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
    }
    best_model = GeoBoostRegressor(**study.best_params)
    predictions = best_model.fit(x, y).predict(x)

    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()
