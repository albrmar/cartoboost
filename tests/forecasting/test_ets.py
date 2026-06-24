import pytest
from cartoboost.forecasting.local.ets import ETSForecaster


def test_ets_validates_parameters():
    with pytest.raises(ValueError, match="trend"):
        ETSForecaster(trend="bad")
    with pytest.raises(ValueError, match="additive"):
        ETSForecaster(seasonal="mul", seasonal_periods=2)
    with pytest.raises(ValueError, match="seasonal_periods"):
        ETSForecaster(seasonal="add", seasonal_periods=1)
    with pytest.raises(ValueError, match="alpha"):
        ETSForecaster(alpha=0.0)
    with pytest.raises(ValueError, match="beta"):
        ETSForecaster(beta=1.2)
    with pytest.raises(ValueError, match="gamma"):
        ETSForecaster(seasonal="add", seasonal_periods=2, gamma=-0.1)


def test_ets_fit_predict_uses_rust_binding():
    model = ETSForecaster()

    model.fit([10.0, 12.0, 14.0, 16.0])
    result = model.predict(2)

    assert [row[3] for row in result.predictions()] == ["ets", "ets"]
    assert [row[2] for row in result.predictions()] == [1, 2]


def test_ets_exposes_native_diagnostics_for_visualization():
    values = [
        64.0,
        58.0,
        52.0,
        49.0,
        66.0,
        60.0,
        54.0,
        50.0,
    ]
    model = ETSForecaster(
        trend="additive",
        seasonal="additive",
        seasonal_periods=4,
        alpha=0.5,
        beta=0.1,
        gamma=0.2,
    )

    model.fit(values)

    assert len(model.fitted_values()) == len(values)
    assert len(model.residuals()) == len(values)
    assert len(model.levels()) == len(values)
    assert len(model.trends()) == len(values)
    assert len(model.seasonal_components()) == len(values)
    assert model.fitted_values()[0] == pytest.approx(values[0])
    assert model.residuals()[0] == pytest.approx(0.0)
    assert max(model.seasonal_components()) > min(model.seasonal_components())


def test_ets_diagnostics_require_fit():
    model = ETSForecaster()

    with pytest.raises(RuntimeError, match="must be fitted"):
        model.fitted_values()
