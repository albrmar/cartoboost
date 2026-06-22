from pathlib import Path

import pytest
from cartoboost.plotting import (
    plot_backtest_metrics,
    plot_changepoint_effects,
    plot_cutoff_predictions,
    plot_forecast,
    plot_forecast_components,
    plot_horizon_metrics,
    plot_interval_calibration,
    plot_metric_comparison,
    plot_predicted_actual,
    plot_residual_diagnostics,
    plot_route_segments,
    plot_seasonality_curve,
    plot_spatial_points,
    save_figure,
    write_plot_report,
    write_pydeck_point_map,
)


def assert_writes_png(tmp_path: Path, figure, name: str) -> None:
    path = save_figure(figure, tmp_path / name, close=True)
    assert path.exists()
    assert path.stat().st_size > 0


def test_predicted_actual_and_residual_diagnostics_write_figures(tmp_path: Path) -> None:
    actual = [12.0, 16.0, 22.0, 27.0, 33.0]
    predicted = [11.0, 17.0, 20.0, 29.0, 31.0]

    assert_writes_png(tmp_path, plot_predicted_actual(actual, predicted), "predicted_actual.png")
    assert_writes_png(tmp_path, plot_residual_diagnostics(actual, predicted), "residuals.png")


def test_metric_and_horizon_comparison_write_figures(tmp_path: Path) -> None:
    metric_rows = [
        {"model": "cartoboost", "rmse": 4.1},
        {"model": "lightgbm", "rmse": 4.8},
        {"model": "mean", "rmse": 9.4},
    ]
    horizon_rows = [
        {"model": "cartoboost", "horizon": 1, "rmse": 2.1},
        {"model": "cartoboost", "horizon": 2, "rmse": 2.8},
        {"model": "lightgbm", "horizon": 1, "rmse": 2.3},
        {"model": "lightgbm", "horizon": 2, "rmse": 3.0},
    ]

    assert_writes_png(tmp_path, plot_metric_comparison(metric_rows), "metrics.png")
    assert_writes_png(tmp_path, plot_horizon_metrics(horizon_rows), "horizon.png")


def test_backtest_and_interval_calibration_plots_write_figures(tmp_path: Path) -> None:
    backtest_rows = [
        {"model": "cartoboost", "fold": 1, "rmse": 2.4},
        {"model": "cartoboost", "fold": 2, "rmse": 2.7},
        {"model": "seasonal_naive", "fold": 1, "rmse": 3.2},
        {"model": "seasonal_naive", "fold": 2, "rmse": 3.5},
    ]
    interval_rows = [
        {"nominal_coverage": 0.5, "coverage": 0.48, "mean_width": 4.2},
        {"nominal_coverage": 0.8, "coverage": 0.77, "mean_width": 7.9},
        {"nominal_coverage": 0.9, "coverage": 0.88, "mean_width": 9.4},
    ]

    assert_writes_png(tmp_path, plot_backtest_metrics(backtest_rows), "backtest.png")
    assert_writes_png(tmp_path, plot_interval_calibration(interval_rows), "intervals.png")


def test_component_diagnostics_write_figures(tmp_path: Path) -> None:
    seasonality_rows = [
        {"component": "weekly", "phase": 0, "value": -2.0, "lower": -3.0, "upper": -1.0},
        {"component": "weekly", "phase": 1, "value": 1.5, "lower": 0.5, "upper": 2.5},
        {"component": "weekly", "phase": 2, "value": 2.0, "lower": 1.0, "upper": 3.0},
    ]
    changepoint_rows = [
        {"timestamp": "2026-01-03", "delta": 2.5},
        {"timestamp": "2026-01-10", "delta": -1.2},
    ]
    cutoff_rows = [
        {"cutoff": "fold_1", "timestamp": "2026-01-03", "actual": 45.0, "prediction": 44.0},
        {"cutoff": "fold_1", "timestamp": "2026-01-04", "actual": 48.0, "prediction": 47.0},
        {"cutoff": "fold_2", "timestamp": "2026-01-04", "actual": 48.0, "prediction": 49.0},
        {"cutoff": "fold_2", "timestamp": "2026-01-05", "actual": 51.0, "prediction": 50.0},
    ]

    assert_writes_png(
        tmp_path,
        plot_seasonality_curve(
            seasonality_rows,
            lower_col="lower",
            upper_col="upper",
            label_col="component",
        ),
        "seasonality.png",
    )
    assert_writes_png(tmp_path, plot_changepoint_effects(changepoint_rows), "changepoints.png")
    assert_writes_png(tmp_path, plot_cutoff_predictions(cutoff_rows), "cutoffs.png")


def test_plot_report_writes_named_diagnostic_bundle(tmp_path: Path) -> None:
    written = write_plot_report(
        tmp_path / "report",
        predicted_actual=([12.0, 15.0, 18.0], [11.5, 15.8, 17.4]),
        metric_rows=[
            {"model": "cartoboost", "rmse": 2.2},
            {"model": "mean", "rmse": 6.8},
        ],
        seasonality_rows=[
            {"phase": 0, "value": -1.0},
            {"phase": 1, "value": 1.0},
        ],
        prefix="taxi",
    )

    assert set(written) == {
        "metric_comparison",
        "predicted_actual",
        "residual_diagnostics",
        "seasonality_curve",
    }
    for path in written.values():
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_map_visualization_helpers_have_clear_optional_extra_errors(tmp_path: Path) -> None:
    with pytest.raises(ImportError, match=r"cartoboost\[visualization\]"):
        plot_spatial_points([{"latitude": 40.7, "longitude": -73.9, "value": 12.0}])

    with pytest.raises(ImportError, match=r"cartoboost\[visualization\]"):
        plot_route_segments(
            [
                {
                    "pickup_latitude": 40.7,
                    "pickup_longitude": -73.9,
                    "dropoff_latitude": 40.75,
                    "dropoff_longitude": -73.98,
                }
            ]
        )

    with pytest.raises(ImportError, match=r"cartoboost\[visualization\]"):
        write_pydeck_point_map(
            [{"latitude": 40.7, "longitude": -73.9}],
            tmp_path / "points.html",
        )


def test_forecast_plot_supports_history_actuals_and_intervals(tmp_path: Path) -> None:
    history = [
        {"series_id": "pickup_1", "timestamp": "2026-01-01", "actual": 41.0},
        {"series_id": "pickup_1", "timestamp": "2026-01-02", "actual": 45.0},
    ]
    forecast = [
        {
            "series_id": "pickup_1",
            "timestamp": "2026-01-03",
            "prediction": 47.0,
            "actual": 48.0,
            "lower": 43.0,
            "upper": 51.0,
        },
        {
            "series_id": "pickup_1",
            "timestamp": "2026-01-04",
            "prediction": 49.0,
            "actual": 50.0,
            "lower": 44.0,
            "upper": 54.0,
        },
    ]

    figure = plot_forecast(
        forecast,
        history=history,
        series_id="pickup_1",
        changepoints=["2026-01-03"],
    )
    assert_writes_png(tmp_path, figure, "forecast.png")


def test_component_plot_supports_trend_seasonality_and_changepoints(tmp_path: Path) -> None:
    rows = [
        {
            "series_id": "pickup_1",
            "timestamp": "2026-01-01",
            "trend": 40.0,
            "weekly": -2.0,
            "event": 0.0,
        },
        {
            "series_id": "pickup_1",
            "timestamp": "2026-01-02",
            "trend": 42.0,
            "weekly": 1.5,
            "event": 3.0,
        },
        {
            "series_id": "pickup_1",
            "timestamp": "2026-01-03",
            "trend": 44.0,
            "weekly": 2.0,
            "event": 0.0,
        },
    ]

    figure = plot_forecast_components(
        rows,
        component_cols=["trend", "weekly", "event"],
        series_id="pickup_1",
        changepoints=["2026-01-02"],
    )
    assert_writes_png(tmp_path, figure, "components.png")


def test_plotting_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        plot_predicted_actual([1.0], [1.0, 2.0])

    with pytest.raises(ValueError, match="lower values"):
        plot_forecast([{"timestamp": "2026-01-01", "prediction": 1.0, "lower": 2.0, "upper": 1.0}])

    with pytest.raises(ValueError, match="model"):
        plot_metric_comparison([{"name": "cartoboost", "rmse": 1.0}])
