import pytest
from cartoboost.forecasting import KalmanForecaster as PublicKalmanForecaster
from cartoboost.forecasting import KrigingForecaster as PublicKrigingForecaster
from cartoboost.forecasting.local import KalmanForecaster, KrigingForecaster


def test_models_are_public_forecasting_imports():
    assert PublicKalmanForecaster is KalmanForecaster
    assert PublicKrigingForecaster is KrigingForecaster


def test_kalman_converts_panel_and_delegates_to_native(install_fake_native):
    native = install_fake_native("KalmanForecaster")

    result = (
        KalmanForecaster(
            level_process_variance=0.1,
            trend_process_variance=0.01,
            observation_variance=0.5,
        )
        .fit({"pickup_1": [10, 12, 14], "pickup_2": [30, 29, 28]})
        .predict(2)
    )

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "level_process_variance": 0.1,
            "trend_process_variance": 0.01,
            "observation_variance": 0.5,
        },
    )
    assert native.calls[1][1].rows[0] == (
        "pickup_1",
        "1970-01-01T00:00:00",
        10.0,
    )


def test_kalman_fit_predict_uses_rust_binding():
    model = KalmanForecaster(
        level_process_variance=0.01,
        trend_process_variance=0.001,
        observation_variance=0.1,
    )

    model.fit([12.0, 14.0, 16.0, 18.0])
    result = model.predict(2)

    predictions = result.predictions()
    assert [row[3] for row in predictions] == ["kalman", "kalman"]
    assert [row[2] for row in predictions] == [1, 2]
    assert predictions[1][-1] > predictions[0][-1]


def test_kriging_normalizes_mapping_coordinates_and_delegates_to_native(install_fake_native):
    native = install_fake_native("KrigingForecaster")

    result = (
        KrigingForecaster(
            coordinates={"pickup_1": (0.0, 0.0), "pickup_2": (1.0, 0.0)},
            range=2.0,
            nugget=0.05,
        )
        .fit({"pickup_1": [10, 11], "pickup_2": [20, 21]})
        .predict(1)
    )

    assert result == {"args": (1,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {
            "coordinates": [("pickup_1", 0.0, 0.0), ("pickup_2", 1.0, 0.0)],
            "range": 2.0,
            "nugget": 0.05,
            "sill": 1.0,
            "variogram_model": "exponential",
            "drift": "ordinary",
            "anisotropy_angle_degrees": 0.0,
            "anisotropy_scaling": 1.0,
            "max_neighbors": None,
            "min_neighbors": 1,
            "max_distance": None,
        },
    )


def test_kriging_fit_predict_uses_rust_binding():
    model = KrigingForecaster(
        coordinates={
            "PULocationID_142": (0.0, 0.0),
            "PULocationID_236": (10.0, 0.0),
        },
        range=1.0,
        nugget=1.0e-9,
    )

    model.fit({"PULocationID_142": [10.0, 12.0], "PULocationID_236": [40.0, 42.0]})
    result = model.predict(1)

    predictions = result.predictions()
    assert [row[3] for row in predictions] == ["kriging", "kriging"]
    assert [row[2] for row in predictions] == [1, 1]
    assert abs(predictions[0][-1] - 12.0) < 1.0e-4


def test_kriging_accepts_coordinate_triples(install_fake_native):
    native = install_fake_native("KrigingForecaster")

    KrigingForecaster(
        coordinates=[("pickup_1", 0, 0), ("pickup_2", 2, 1)],
        range=1.5,
        nugget=0.0,
    ).fit({"pickup_1": [10, 11], "pickup_2": [20, 21]})

    assert native.calls[0] == (
        "init",
        {
            "coordinates": [("pickup_1", 0.0, 0.0), ("pickup_2", 2.0, 1.0)],
            "range": 1.5,
            "nugget": 0.0,
            "sill": 1.0,
            "variogram_model": "exponential",
            "drift": "ordinary",
            "anisotropy_angle_degrees": 0.0,
            "anisotropy_scaling": 1.0,
            "max_neighbors": None,
            "min_neighbors": 1,
            "max_distance": None,
        },
    )


def test_kriging_passes_extended_native_parameters(install_fake_native):
    native = install_fake_native("KrigingForecaster")

    KrigingForecaster(
        coordinates=[("pickup_1", 0, 0), ("pickup_2", 2, 1), ("pickup_3", 3, 1)],
        range=1.5,
        nugget=0.01,
        sill=2.0,
        variogram_model="spherical",
        drift="linear",
        anisotropy_angle_degrees=30.0,
        anisotropy_scaling=0.5,
        max_neighbors=2,
        min_neighbors=2,
        max_distance=5.0,
    ).fit({"pickup_1": [10, 11], "pickup_2": [20, 21], "pickup_3": [30, 31]})

    assert native.calls[0] == (
        "init",
        {
            "coordinates": [
                ("pickup_1", 0.0, 0.0),
                ("pickup_2", 2.0, 1.0),
                ("pickup_3", 3.0, 1.0),
            ],
            "range": 1.5,
            "nugget": 0.01,
            "sill": 2.0,
            "variogram_model": "spherical",
            "drift": "linear",
            "anisotropy_angle_degrees": 30.0,
            "anisotropy_scaling": 0.5,
            "max_neighbors": 2,
            "min_neighbors": 2,
            "max_distance": 5.0,
        },
    )


def test_kriging_rejects_bad_mapping_coordinate_shape(install_fake_native):
    install_fake_native("KrigingForecaster")

    with pytest.raises(ValueError, match="series_id to \\(x, y\\)"):
        KrigingForecaster(coordinates={"pickup_1": (0.0, 0.0, 1.0)})
