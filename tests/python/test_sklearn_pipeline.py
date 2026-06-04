from __future__ import annotations

import numpy as np
from geoboost import GeoBoostRegressor
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
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
