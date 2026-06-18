import builtins

import numpy as np
import pytest
from cartoboost.forecasting.local import AutoARIMAForecaster


def test_auto_arima_missing_pmdarima_raises_by_default(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("pmdarima"):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="requires pmdarima"):
        AutoARIMAForecaster().fit([1.0, 2.0, 3.0])


def test_auto_arima_fallback_policy_uses_naive_when_pmdarima_missing(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("pmdarima"):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    model = AutoARIMAForecaster(error_policy="fallback").fit([2.0, 4.0, 8.0])

    np.testing.assert_allclose(model.predict(2), [8.0, 8.0])
    assert model.metadata_["backend"] == "naive_fallback"


def test_auto_arima_fallback_preserves_panel_shape(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("pmdarima"):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    model = AutoARIMAForecaster(error_policy="fallback").fit(
        np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
    )

    np.testing.assert_allclose(model.predict(2), [[3.0, 7.0], [3.0, 7.0]])
