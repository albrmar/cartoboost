import builtins

import numpy as np
import pytest
from cartoboost.forecasting.local import ETSForecaster


def test_ets_missing_statsmodels_has_clear_error(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="requires statsmodels"):
        ETSForecaster().fit([1.0, 2.0, 3.0])


def test_ets_predicts_with_residual_interval_when_statsmodels_available():
    pytest.importorskip("statsmodels")
    y = np.array([10.0, 11.0, 13.0, 16.0, 20.0, 25.0])
    model = ETSForecaster(trend="add").fit(y)

    forecast = model.predict(2, return_interval=True)

    assert forecast.mean.shape == (2,)
    assert forecast.lower.shape == (2,)
    assert forecast.upper.shape == (2,)
    assert forecast.metadata["interval_method"] == "residual_normal_fallback"
