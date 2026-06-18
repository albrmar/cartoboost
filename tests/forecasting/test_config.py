import pytest
from cartoboost.forecasting.config import ForecastingConfig


def test_forecasting_config_parses_models_and_constructs_default_backed_models(
    install_fake_native,
):
    native = install_fake_native("SeasonalNaiveForecaster")
    config = ForecastingConfig.from_toml(
        """
        horizon = 3
        freq = "H"
        target_column = "pickup_demand"
        time_column = "pickup_hour"
        panel_columns = ["PULocationID"]

        [feature_config]
        lags = [1, 24]

        [[models]]
        name = "seasonal_naive"
        optional_dependencies = []

        [models.params]
        season_length = 2
        """
    )

    models = config.construct_models()
    model = models["seasonal_naive"]

    assert config.horizon == 3
    assert config.feature_config == {"lags": [1, 24]}
    assert model.fit([10, 11, 12, 13]).predict(1) == {"args": (1,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {"season_length": 2, "prediction_interval_levels": ()},
    )
    assert native.calls[1][1].rows[-1] == ("__single__", "1970-01-04T00:00:00", 13.0)


def test_forecasting_config_constructs_independent_kalman_model(install_fake_native):
    native = install_fake_native("KalmanForecaster")
    config = ForecastingConfig.from_toml(
        """
        horizon = 2

        [[models]]
        name = "kalman"

        [models.params]
        level_process_variance = 0.2
        trend_process_variance = 0.03
        observation_variance = 0.7
        """
    )

    model = config.construct_models()["kalman"]

    assert model.fit([1, 2, 4]).predict(1) == {"args": (1,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "level_process_variance": 0.2,
            "trend_process_variance": 0.03,
            "observation_variance": 0.7,
        },
    )


def test_forecasting_config_constructs_independent_kriging_model(install_fake_native):
    native = install_fake_native("KrigingForecaster")
    config = ForecastingConfig.from_toml(
        """
        horizon = 1

        [[models]]
        name = "kriging"

        [models.params.coordinates]
        PULocationID_142 = [0.0, 0.0]
        PULocationID_236 = [1.0, 0.0]

        [models.params]
        range = 3.0
        nugget = 0.01
        """
    )

    model = config.construct_models()["kriging"]

    assert model.fit({"PULocationID_142": [10, 11], "PULocationID_236": [20, 21]}).predict(1) == {
        "args": (1,),
        "kwargs": {},
    }
    assert native.calls[0] == (
        "init",
        {
            "coordinates": [("PULocationID_142", 0.0, 0.0), ("PULocationID_236", 1.0, 0.0)],
            "range": 3.0,
            "nugget": 0.01,
        },
    )


def test_forecasting_config_rejects_unknown_root_fields_by_default():
    with pytest.raises(ValueError, match="unknown forecasting config field"):
        ForecastingConfig.from_toml(
            """
            horizon = 2
            surprise = true
            """
        )


def test_forecasting_config_rejects_unknown_model_fields_by_default():
    with pytest.raises(ValueError, match="unknown model config field"):
        ForecastingConfig.from_toml(
            """
            horizon = 2

            [[models]]
            name = "naive"
            hidden_state = "not portable"
            """
        )


def test_forecasting_config_allows_unknown_fields_when_requested():
    config = ForecastingConfig.from_toml(
        """
        allow_unknown = true
        horizon = 2
        owner = "taxi-team"

        [[models]]
        name = "naive"
        local_note = "accepted as metadata"
        """
    )

    assert config.metadata["unknown_fields"] == {"owner": "taxi-team"}
    assert config.models[0].metadata["unknown_fields"] == {"local_note": "accepted as metadata"}


def test_forecasting_config_requires_positive_horizon():
    with pytest.raises(ValueError, match="positive integer"):
        ForecastingConfig.from_toml("horizon = 0")
