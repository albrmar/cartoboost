"""Plotting helpers for CartoBoost predictions, metrics, and forecast artifacts.

The functions in this module render already-computed model outputs. They do
not fit models, score benchmarks, or change forecasting behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PALETTE = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#be123c",
    "#4b5563",
)


def plot_predicted_actual(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    title: str = "Predicted vs actual",
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    ax: Any | None = None,
    point_color: str = DEFAULT_PALETTE[0],
    reference_color: str = "#111827",
) -> Any:
    """Render a predicted-vs-actual scatter plot with a parity reference line."""

    pyplot = _require_pyplot()
    actual, predicted = _paired_numeric_arrays(y_true, y_pred, "y_true", "y_pred")
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(6.6, 5.2))

    axis.scatter(actual, predicted, s=34, alpha=0.78, color=point_color, edgecolor="white")
    lo = float(min(np.min(actual), np.min(predicted)))
    hi = float(max(np.max(actual), np.max(predicted)))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    axis.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color=reference_color)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.22)
    _tight_layout(figure)
    return figure


def plot_residual_diagnostics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    title: str = "Residual diagnostics",
    residual_label: str = "Residual",
    fitted_label: str = "Predicted",
    figure_size: tuple[float, float] = (10.0, 4.6),
) -> Any:
    """Render residual-vs-fitted and residual distribution diagnostics."""

    pyplot = _require_pyplot()
    actual, predicted = _paired_numeric_arrays(y_true, y_pred, "y_true", "y_pred")
    residuals = actual - predicted

    figure, axes = pyplot.subplots(1, 2, figsize=figure_size)
    axes[0].scatter(predicted, residuals, s=34, alpha=0.78, color=DEFAULT_PALETTE[1])
    axes[0].axhline(0.0, color="#111827", linewidth=1.1, linestyle="--")
    axes[0].set_xlabel(fitted_label)
    axes[0].set_ylabel(residual_label)
    axes[0].set_title("Residuals by prediction")
    axes[0].grid(alpha=0.22)

    bins = min(24, max(6, int(np.sqrt(residuals.size))))
    axes[1].hist(residuals, bins=bins, color=DEFAULT_PALETTE[2], alpha=0.82, edgecolor="white")
    axes[1].axvline(0.0, color="#111827", linewidth=1.1, linestyle="--")
    axes[1].set_xlabel(residual_label)
    axes[1].set_ylabel("Rows")
    axes[1].set_title("Residual distribution")
    axes[1].grid(axis="y", alpha=0.22)

    figure.suptitle(title)
    _tight_layout(figure)
    return figure


def plot_metric_comparison(
    rows: Any,
    *,
    metric: str = "rmse",
    label_col: str = "model",
    title: str | None = None,
    ylabel: str | None = None,
    sort: bool = True,
    ax: Any | None = None,
) -> Any:
    """Render a sorted bar chart from metric rows or a dataframe-like object."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    values: list[tuple[str, float]] = []
    for row in parsed:
        if label_col not in row or metric not in row:
            raise ValueError(f"each row must contain '{label_col}' and '{metric}'")
        values.append((str(row[label_col]), _finite_float(row[metric], metric)))
    if not values:
        raise ValueError("rows must contain at least one metric row")
    if sort:
        values.sort(key=lambda item: item[1])

    labels = [label for label, _ in values]
    metric_values = [value for _, value in values]
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(max(7.0, len(labels) * 0.76), 4.8))
    colors = [DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)] for index in range(len(labels))]
    axis.bar(labels, metric_values, color=colors)
    axis.set_title(title or f"{metric.upper()} by model")
    axis.set_ylabel(ylabel or metric.upper())
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.22)
    _tight_layout(figure)
    return figure


def plot_horizon_metrics(
    rows: Any,
    *,
    horizon_col: str = "horizon",
    metric_col: str = "rmse",
    model_col: str = "model",
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Render metric trajectories by forecast horizon for one or more models."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    series: dict[str, list[tuple[float, float]]] = {}
    for row in parsed:
        missing = {horizon_col, metric_col, model_col}.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"each row must contain column(s): {names}")
        model = str(row[model_col])
        horizon = _finite_float(row[horizon_col], horizon_col)
        metric_value = _finite_float(row[metric_col], metric_col)
        series.setdefault(model, []).append((horizon, metric_value))
    if not series:
        raise ValueError("rows must contain at least one horizon metric row")

    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(7.6, 4.8))
    for index, (model, points) in enumerate(sorted(series.items())):
        points.sort(key=lambda item: item[0])
        axis.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            marker="o",
            linewidth=1.9,
            color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
            label=model,
        )
    axis.set_title(title or f"{metric_col.upper()} by forecast horizon")
    axis.set_xlabel("Horizon")
    axis.set_ylabel(metric_col.upper())
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    _tight_layout(figure)
    return figure


def plot_backtest_metrics(
    rows: Any,
    *,
    fold_col: str = "fold",
    metric_col: str = "rmse",
    model_col: str = "model",
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Render model metric trajectories over rolling-origin or blocked folds."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    series: dict[str, list[tuple[float, float]]] = {}
    for row in parsed:
        missing = {fold_col, metric_col, model_col}.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"each row must contain column(s): {names}")
        model = str(row[model_col])
        fold = _finite_float(row[fold_col], fold_col)
        metric_value = _finite_float(row[metric_col], metric_col)
        series.setdefault(model, []).append((fold, metric_value))
    if not series:
        raise ValueError("rows must contain at least one backtest metric row")

    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(7.8, 4.8))
    for index, (model, points) in enumerate(sorted(series.items())):
        points.sort(key=lambda item: item[0])
        axis.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            marker="o",
            linewidth=1.9,
            color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
            label=model,
        )
    axis.set_title(title or f"{metric_col.upper()} by validation fold")
    axis.set_xlabel(fold_col)
    axis.set_ylabel(metric_col.upper())
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    _tight_layout(figure)
    return figure


def plot_interval_calibration(
    rows: Any,
    *,
    nominal_col: str = "nominal_coverage",
    observed_col: str = "coverage",
    width_col: str | None = "mean_width",
    title: str = "Prediction interval calibration",
) -> Any:
    """Render nominal-vs-observed interval coverage and optional mean width."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one interval metric row")
    points: list[tuple[float, float, float | None]] = []
    for row in parsed:
        if nominal_col not in row or observed_col not in row:
            raise ValueError(f"each row must contain '{nominal_col}' and '{observed_col}'")
        nominal = _finite_float(row[nominal_col], nominal_col)
        observed = _finite_float(row[observed_col], observed_col)
        if not 0.0 <= nominal <= 1.0 or not 0.0 <= observed <= 1.0:
            raise ValueError("coverage values must be between 0 and 1")
        width = _finite_float(row[width_col], width_col) if width_col and width_col in row else None
        points.append((nominal, observed, width))
    points.sort(key=lambda item: item[0])

    has_width = any(width is not None for _, _, width in points)
    figure, axes = pyplot.subplots(
        1, 2 if has_width else 1, figsize=(10.0 if has_width else 5.6, 4.8)
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    calibration_axis = axes[0]
    nominal_values = [item[0] for item in points]
    observed_values = [item[1] for item in points]
    calibration_axis.plot(
        nominal_values,
        observed_values,
        marker="o",
        linewidth=1.9,
        color=DEFAULT_PALETTE[0],
        label="Observed",
    )
    calibration_axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.1, color="#111827")
    calibration_axis.set_xlim(0.0, 1.0)
    calibration_axis.set_ylim(0.0, 1.0)
    calibration_axis.set_xlabel("Nominal coverage")
    calibration_axis.set_ylabel("Observed coverage")
    calibration_axis.set_title("Coverage calibration")
    calibration_axis.grid(alpha=0.22)

    if has_width:
        width_axis = axes[1]
        width_axis.bar(
            [f"{value:.0%}" for value in nominal_values],
            [0.0 if width is None else width for _, _, width in points],
            color=DEFAULT_PALETTE[2],
        )
        width_axis.set_xlabel("Nominal coverage")
        width_axis.set_ylabel(width_col or "Mean width")
        width_axis.set_title("Interval width")
        width_axis.grid(axis="y", alpha=0.22)

    figure.suptitle(title)
    _tight_layout(figure)
    return figure


def plot_seasonality_curve(
    rows: Any,
    *,
    x_col: str = "phase",
    value_col: str = "value",
    lower_col: str | None = None,
    upper_col: str | None = None,
    label_col: str | None = None,
    title: str = "Seasonality curve",
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Render one or more seasonal/component curves with optional uncertainty bands."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one seasonality row")
    for row in parsed:
        missing = {x_col, value_col}.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"seasonality rows must contain column(s): {names}")

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in parsed:
        label = str(row[label_col]) if label_col and label_col in row else value_col
        groups.setdefault(label, []).append(row)

    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(7.8, 4.8))
    for index, (label, group_rows) in enumerate(sorted(groups.items())):
        sorted_rows = sorted(group_rows, key=lambda row: _finite_float(row[x_col], x_col))
        x_values = [_finite_float(row[x_col], x_col) for row in sorted_rows]
        values = [_finite_float(row[value_col], value_col) for row in sorted_rows]
        color = DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)]
        axis.plot(x_values, values, linewidth=1.9, color=color, label=label)
        if (
            lower_col
            and upper_col
            and all(lower_col in row and upper_col in row for row in sorted_rows)
        ):
            lower = [_finite_float(row[lower_col], lower_col) for row in sorted_rows]
            upper = [_finite_float(row[upper_col], upper_col) for row in sorted_rows]
            if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
                raise ValueError("seasonality lower values must be <= upper values")
            axis.fill_between(x_values, lower, upper, color=color, alpha=0.14)

    axis.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    axis.set_title(title)
    axis.set_xlabel(xlabel or x_col)
    axis.set_ylabel(ylabel or value_col)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    _tight_layout(figure)
    return figure


def plot_changepoint_effects(
    rows: Any,
    *,
    time_col: str = "timestamp",
    delta_col: str = "delta",
    label_col: str | None = None,
    title: str = "Changepoint effects",
    ax: Any | None = None,
) -> Any:
    """Render signed changepoint effect magnitudes."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one changepoint row")
    for row in parsed:
        missing = {time_col, delta_col}.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"changepoint rows must contain column(s): {names}")
    parsed = sorted(parsed, key=lambda row: row[time_col])

    labels = [
        str(row[label_col]) if label_col and label_col in row else str(row[time_col])
        for row in parsed
    ]
    deltas = [_finite_float(row[delta_col], delta_col) for row in parsed]
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(max(7.0, len(labels) * 0.64), 4.8))
    colors = [DEFAULT_PALETTE[2] if value >= 0.0 else DEFAULT_PALETTE[1] for value in deltas]
    axis.bar(labels, deltas, color=colors)
    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel(time_col if label_col is None else label_col)
    axis.set_ylabel(delta_col)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.22)
    _tight_layout(figure)
    return figure


def plot_cutoff_predictions(
    rows: Any,
    *,
    time_col: str = "timestamp",
    cutoff_col: str = "cutoff",
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    title: str = "Predictions by cutoff",
    ax: Any | None = None,
) -> Any:
    """Render cross-validation predictions grouped by cutoff date or fold."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one cutoff prediction row")
    required = {time_col, cutoff_col, actual_col, prediction_col}
    for row in parsed:
        missing = required.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"cutoff prediction rows must contain column(s): {names}")

    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(9.0, 4.9))
    actual_by_time: dict[Any, float] = {}
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in parsed:
        actual_by_time[row[time_col]] = _finite_float(row[actual_col], actual_col)
        groups.setdefault(str(row[cutoff_col]), []).append(row)

    actual_points = sorted(actual_by_time.items(), key=lambda item: item[0])
    axis.plot(
        [item[0] for item in actual_points],
        [item[1] for item in actual_points],
        color="#111827",
        linewidth=1.8,
        label="Actual",
    )
    for index, (cutoff, group_rows) in enumerate(sorted(groups.items())):
        sorted_rows = sorted(group_rows, key=lambda row: row[time_col])
        axis.plot(
            [row[time_col] for row in sorted_rows],
            [_finite_float(row[prediction_col], prediction_col) for row in sorted_rows],
            marker="o",
            linewidth=1.4,
            color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
            label=f"Cutoff {cutoff}",
            alpha=0.86,
        )

    axis.set_title(title)
    axis.set_xlabel(time_col)
    axis.set_ylabel(prediction_col)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    _tight_layout(figure)
    return figure


def plot_spatial_points(
    rows: Any,
    *,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    value_col: str | None = None,
    title: str = "Spatial points",
    ax: Any | None = None,
    cmap: str = "viridis",
    point_size: float = 34.0,
) -> Any:
    """Render static latitude/longitude points with the visualization extra."""

    pyplot = _require_pyplot()
    geopandas, point_cls, _ = _require_static_map_stack()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one spatial point row")
    geometries = []
    values = []
    for row in parsed:
        if latitude_col not in row or longitude_col not in row:
            raise ValueError(f"point rows must contain '{latitude_col}' and '{longitude_col}'")
        latitude = _finite_float(row[latitude_col], latitude_col)
        longitude = _finite_float(row[longitude_col], longitude_col)
        geometries.append(point_cls(longitude, latitude))
        if value_col is not None:
            if value_col not in row:
                raise ValueError(f"point rows must contain '{value_col}'")
            values.append(_finite_float(row[value_col], value_col))

    frame = geopandas.GeoDataFrame(parsed, geometry=geometries, crs="EPSG:4326")
    if value_col is not None:
        frame[value_col] = values
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(7.2, 6.0))
    frame.plot(
        ax=axis,
        column=value_col,
        cmap=cmap if value_col else None,
        markersize=point_size,
        legend=value_col is not None,
        color=None if value_col else DEFAULT_PALETTE[0],
        alpha=0.82,
    )
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.18)
    _tight_layout(figure)
    return figure


def plot_route_segments(
    rows: Any,
    *,
    pickup_latitude_col: str = "pickup_latitude",
    pickup_longitude_col: str = "pickup_longitude",
    dropoff_latitude_col: str = "dropoff_latitude",
    dropoff_longitude_col: str = "dropoff_longitude",
    value_col: str | None = None,
    title: str = "Route segments",
    ax: Any | None = None,
    cmap: str = "viridis",
    linewidth: float = 1.4,
) -> Any:
    """Render static pickup/dropoff route segments with the visualization extra."""

    pyplot = _require_pyplot()
    geopandas, point_cls, line_cls = _require_static_map_stack()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one route row")
    required = {
        pickup_latitude_col,
        pickup_longitude_col,
        dropoff_latitude_col,
        dropoff_longitude_col,
    }
    geometries = []
    values = []
    for row in parsed:
        missing = required.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"route rows must contain column(s): {names}")
        pickup = point_cls(
            _finite_float(row[pickup_longitude_col], pickup_longitude_col),
            _finite_float(row[pickup_latitude_col], pickup_latitude_col),
        )
        dropoff = point_cls(
            _finite_float(row[dropoff_longitude_col], dropoff_longitude_col),
            _finite_float(row[dropoff_latitude_col], dropoff_latitude_col),
        )
        geometries.append(line_cls([pickup, dropoff]))
        if value_col is not None:
            if value_col not in row:
                raise ValueError(f"route rows must contain '{value_col}'")
            values.append(_finite_float(row[value_col], value_col))

    frame = geopandas.GeoDataFrame(parsed, geometry=geometries, crs="EPSG:4326")
    if value_col is not None:
        frame[value_col] = values
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(7.2, 6.0))
    frame.plot(
        ax=axis,
        column=value_col,
        cmap=cmap if value_col else None,
        linewidth=linewidth,
        legend=value_col is not None,
        color=None if value_col else DEFAULT_PALETTE[0],
        alpha=0.72,
    )
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.18)
    _tight_layout(figure)
    return figure


def write_pydeck_point_map(
    rows: Any,
    output_html: str | Path,
    *,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    value_col: str | None = None,
    tooltip_cols: Sequence[str] = (),
    title: str = "CartoBoost point map",
    radius_scale: float = 35.0,
) -> Path:
    """Write an interactive PyDeck point map and return the HTML path."""

    pydeck = _require_pydeck_stack()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one spatial point row")
    _validate_coordinate_rows(parsed, latitude_col, longitude_col)
    latitude, longitude = _mean_lat_lon(parsed, latitude_col, longitude_col)
    layer = pydeck.Layer(
        "ScatterplotLayer",
        data=parsed,
        get_position=[longitude_col, latitude_col],
        get_radius=_pydeck_radius_expression(value_col, radius_scale),
        get_fill_color=_pydeck_color_expression(value_col),
        pickable=True,
        opacity=0.82,
    )
    deck = pydeck.Deck(
        layers=[layer],
        initial_view_state=pydeck.ViewState(latitude=latitude, longitude=longitude, zoom=10),
        tooltip=_pydeck_tooltip(tooltip_cols),
        map_style=None,
        description=title,
    )
    return _write_pydeck_html(deck, output_html)


def write_pydeck_route_map(
    rows: Any,
    output_html: str | Path,
    *,
    pickup_latitude_col: str = "pickup_latitude",
    pickup_longitude_col: str = "pickup_longitude",
    dropoff_latitude_col: str = "dropoff_latitude",
    dropoff_longitude_col: str = "dropoff_longitude",
    value_col: str | None = None,
    tooltip_cols: Sequence[str] = (),
    title: str = "CartoBoost route map",
) -> Path:
    """Write an interactive PyDeck pickup/dropoff arc map and return the HTML path."""

    pydeck = _require_pydeck_stack()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one route row")
    _validate_coordinate_rows(parsed, pickup_latitude_col, pickup_longitude_col)
    _validate_coordinate_rows(parsed, dropoff_latitude_col, dropoff_longitude_col)
    latitude, longitude = _mean_lat_lon(parsed, pickup_latitude_col, pickup_longitude_col)
    layer = pydeck.Layer(
        "ArcLayer",
        data=parsed,
        get_source_position=[pickup_longitude_col, pickup_latitude_col],
        get_target_position=[dropoff_longitude_col, dropoff_latitude_col],
        get_source_color=_pydeck_color_expression(value_col, positive=(37, 99, 235)),
        get_target_color=_pydeck_color_expression(value_col, positive=(220, 38, 38)),
        get_width=2,
        pickable=True,
        auto_highlight=True,
    )
    deck = pydeck.Deck(
        layers=[layer],
        initial_view_state=pydeck.ViewState(latitude=latitude, longitude=longitude, zoom=10),
        tooltip=_pydeck_tooltip(tooltip_cols),
        map_style=None,
        description=title,
    )
    return _write_pydeck_html(deck, output_html)


def plot_forecast(
    forecast: Any,
    *,
    history: Any | None = None,
    time_col: str = "timestamp",
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    lower_col: str | None = "lower",
    upper_col: str | None = "upper",
    series_id: str | None = None,
    series_id_col: str = "series_id",
    changepoints: Sequence[Any] | None = None,
    title: str = "Forecast",
    ax: Any | None = None,
) -> Any:
    """Render history, forecasts, actual holdout values, and optional intervals."""

    pyplot = _require_pyplot()
    forecast_rows = _filter_series(_rows_from_table(forecast), series_id, series_id_col)
    history_rows = (
        _filter_series(_rows_from_table(history), series_id, series_id_col) if history else []
    )
    if not forecast_rows:
        raise ValueError("forecast must contain at least one row")
    if any(time_col not in row for row in forecast_rows):
        raise ValueError(f"forecast rows must contain '{time_col}'")
    if any(prediction_col not in row for row in forecast_rows):
        raise ValueError(f"forecast rows must contain '{prediction_col}'")

    forecast_rows = sorted(forecast_rows, key=lambda row: row[time_col])
    history_rows = sorted(history_rows, key=lambda row: row[time_col]) if history_rows else []
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(9.0, 4.9))

    if history_rows:
        if any(time_col not in row or actual_col not in row for row in history_rows):
            raise ValueError(f"history rows must contain '{time_col}' and '{actual_col}'")
        axis.plot(
            [row[time_col] for row in history_rows],
            [_finite_float(row[actual_col], actual_col) for row in history_rows],
            color="#111827",
            linewidth=1.8,
            label="History",
        )

    forecast_times = [row[time_col] for row in forecast_rows]
    forecast_values = [_finite_float(row[prediction_col], prediction_col) for row in forecast_rows]
    axis.plot(
        forecast_times,
        forecast_values,
        marker="o",
        color=DEFAULT_PALETTE[0],
        linewidth=1.9,
        label="Forecast",
    )

    if any(actual_col in row and row[actual_col] is not None for row in forecast_rows):
        actual_rows = [
            row for row in forecast_rows if actual_col in row and row[actual_col] is not None
        ]
        axis.scatter(
            [row[time_col] for row in actual_rows],
            [_finite_float(row[actual_col], actual_col) for row in actual_rows],
            color=DEFAULT_PALETTE[1],
            s=34,
            label="Actual",
            zorder=3,
        )

    if (
        lower_col
        and upper_col
        and all(lower_col in row and upper_col in row for row in forecast_rows)
    ):
        lower = [_finite_float(row[lower_col], lower_col) for row in forecast_rows]
        upper = [_finite_float(row[upper_col], upper_col) for row in forecast_rows]
        if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("interval lower values must be <= upper values")
        axis.fill_between(
            forecast_times, lower, upper, color=DEFAULT_PALETTE[0], alpha=0.16, label="Interval"
        )

    if changepoints:
        for index, changepoint in enumerate(changepoints):
            axis.axvline(
                changepoint,
                color="#6b7280",
                linewidth=0.9,
                linestyle=":",
                alpha=0.72,
                label="Changepoint" if index == 0 else None,
            )

    axis.set_title(title if series_id is None else f"{title}: {series_id}")
    axis.set_xlabel(time_col)
    axis.set_ylabel(prediction_col)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    axis.tick_params(axis="x", rotation=25)
    _tight_layout(figure)
    return figure


def plot_forecast_components(
    rows: Any,
    *,
    time_col: str = "timestamp",
    component_cols: Sequence[str] | None = None,
    series_id: str | None = None,
    series_id_col: str = "series_id",
    changepoints: Sequence[Any] | None = None,
    title: str = "Forecast components",
    figure_size: tuple[float, float] | None = None,
) -> Any:
    """Render trend, seasonal, event, or other forecast component columns."""

    pyplot = _require_pyplot()
    parsed_rows = _filter_series(_rows_from_table(rows), series_id, series_id_col)
    if not parsed_rows:
        raise ValueError("rows must contain at least one component row")
    if any(time_col not in row for row in parsed_rows):
        raise ValueError(f"rows must contain '{time_col}'")
    parsed_rows = sorted(parsed_rows, key=lambda row: row[time_col])

    selected_components = list(component_cols or _infer_component_columns(parsed_rows, time_col))
    if not selected_components:
        raise ValueError("component_cols must contain at least one numeric component column")
    for row in parsed_rows:
        missing = [column for column in selected_components if column not in row]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"component rows must contain column(s): {names}")

    height = max(2.8, 2.35 * len(selected_components))
    figure, axes = pyplot.subplots(
        len(selected_components),
        1,
        figsize=figure_size or (9.0, height),
        sharex=True,
        squeeze=False,
    )
    times = [row[time_col] for row in parsed_rows]
    for index, component in enumerate(selected_components):
        axis = axes[index][0]
        values = [_finite_float(row[component], component) for row in parsed_rows]
        axis.plot(
            times,
            values,
            color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
            linewidth=1.9,
        )
        axis.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
        if changepoints and index == 0:
            for changepoint in changepoints:
                axis.axvline(changepoint, color="#6b7280", linewidth=0.9, linestyle=":", alpha=0.72)
        axis.set_ylabel(component)
        axis.grid(alpha=0.22)

    axes[0][0].set_title(title if series_id is None else f"{title}: {series_id}")
    axes[-1][0].set_xlabel(time_col)
    _tight_layout(figure)
    return figure


def save_figure(figure: Any, path: str | Path, *, dpi: int = 160, close: bool = False) -> Path:
    """Save a Matplotlib figure and return the written path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if close:
        _require_pyplot().close(figure)
    return output


def write_plot_report(
    output_dir: str | Path,
    *,
    predicted_actual: tuple[Sequence[float], Sequence[float]] | None = None,
    forecast: Any | None = None,
    history: Any | None = None,
    components: Any | None = None,
    metric_rows: Any | None = None,
    horizon_rows: Any | None = None,
    backtest_rows: Any | None = None,
    interval_rows: Any | None = None,
    seasonality_rows: Any | None = None,
    changepoint_rows: Any | None = None,
    cutoff_rows: Any | None = None,
    prefix: str = "cartoboost",
    dpi: int = 160,
) -> dict[str, str]:
    """Write a standard diagnostic plot bundle and return paths by plot name."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def write(name: str, figure: Any) -> None:
        path = save_figure(figure, output / f"{prefix}_{name}.png", dpi=dpi, close=True)
        written[name] = str(path)

    if predicted_actual is not None:
        y_true, y_pred = predicted_actual
        write("predicted_actual", plot_predicted_actual(y_true, y_pred))
        write("residual_diagnostics", plot_residual_diagnostics(y_true, y_pred))
    if forecast is not None:
        write("forecast", plot_forecast(forecast, history=history))
    if components is not None:
        write("forecast_components", plot_forecast_components(components))
    if metric_rows is not None:
        write("metric_comparison", plot_metric_comparison(metric_rows))
    if horizon_rows is not None:
        write("horizon_metrics", plot_horizon_metrics(horizon_rows))
    if backtest_rows is not None:
        write("backtest_metrics", plot_backtest_metrics(backtest_rows))
    if interval_rows is not None:
        write("interval_calibration", plot_interval_calibration(interval_rows))
    if seasonality_rows is not None:
        write("seasonality_curve", plot_seasonality_curve(seasonality_rows))
    if changepoint_rows is not None:
        write("changepoint_effects", plot_changepoint_effects(changepoint_rows))
    if cutoff_rows is not None:
        write("cutoff_predictions", plot_cutoff_predictions(cutoff_rows))
    if not written:
        raise ValueError("at least one plot input must be provided")
    return written


def _require_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as pyplot
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "CartoBoost visualization requires matplotlib. Install it with "
            "`pip install 'cartoboost[visualization]'` or include the dev dependency group."
        ) from exc
    return pyplot


def _require_static_map_stack() -> tuple[Any, Any, Any]:
    try:
        import geopandas
        from shapely.geometry import LineString, Point
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "CartoBoost map visualization requires geopandas and shapely. "
            "Install them with `pip install 'cartoboost[visualization]'`."
        ) from exc
    return geopandas, Point, LineString


def _require_pydeck_stack() -> Any:
    try:
        import pydeck
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "CartoBoost interactive map visualization requires pydeck. "
            "Install it with `pip install 'cartoboost[visualization]'`."
        ) from exc
    return pydeck


def _figure_axis(pyplot: Any, *, ax: Any | None, figsize: tuple[float, float]) -> tuple[Any, Any]:
    if ax is not None:
        return ax.figure, ax
    return pyplot.subplots(figsize=figsize)


def _tight_layout(figure: Any) -> None:
    try:
        figure.tight_layout()
    except RuntimeError:
        pass


def _paired_numeric_arrays(
    left: Sequence[float],
    right: Sequence[float],
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape")
    if left_array.ndim != 1:
        raise ValueError(f"{left_name} and {right_name} must be one-dimensional")
    if left_array.size == 0:
        raise ValueError(f"{left_name} and {right_name} must contain at least one value")
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise ValueError(f"{left_name} and {right_name} must contain only finite values")
    return left_array, right_array


def _rows_from_table(table: Any) -> list[dict[str, Any]]:
    if table is None:
        return []
    if hasattr(table, "to_dict"):
        try:
            records = table.to_dict(orient="records")
        except TypeError:
            records = table.to_dict()
        if isinstance(records, Mapping):
            keys = list(records)
            if not keys:
                return []
            length = len(records[keys[0]])
            return [{key: records[key][index] for key in keys} for index in range(length)]
        return [dict(row) for row in records]
    if isinstance(table, Mapping):
        keys = list(table)
        if not keys:
            return []
        length = len(table[keys[0]])
        return [{key: table[key][index] for key in keys} for index in range(length)]
    return [dict(row) for row in table]


def _infer_component_columns(rows: Sequence[Mapping[str, Any]], time_col: str) -> list[str]:
    reserved = {
        time_col,
        "timestamp",
        "series_id",
        "horizon",
        "actual",
        "prediction",
        "mean",
        "lower",
        "upper",
    }
    columns: list[str] = []
    for column in rows[0]:
        if column in reserved or column.startswith("lower") or column.startswith("upper"):
            continue
        try:
            [_finite_float(row[column], column) for row in rows]
        except ValueError:
            continue
        columns.append(str(column))
    return columns


def _filter_series(
    rows: Sequence[Mapping[str, Any]],
    series_id: str | None,
    series_id_col: str,
) -> list[dict[str, Any]]:
    if series_id is None:
        return [dict(row) for row in rows]
    return [dict(row) for row in rows if str(row.get(series_id_col)) == series_id]


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be numeric") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} values must be finite")
    return parsed


def _validate_coordinate_rows(
    rows: Sequence[Mapping[str, Any]],
    latitude_col: str,
    longitude_col: str,
) -> None:
    for row in rows:
        if latitude_col not in row or longitude_col not in row:
            raise ValueError(f"rows must contain '{latitude_col}' and '{longitude_col}'")
        _finite_float(row[latitude_col], latitude_col)
        _finite_float(row[longitude_col], longitude_col)


def _mean_lat_lon(
    rows: Sequence[Mapping[str, Any]],
    latitude_col: str,
    longitude_col: str,
) -> tuple[float, float]:
    latitudes = [_finite_float(row[latitude_col], latitude_col) for row in rows]
    longitudes = [_finite_float(row[longitude_col], longitude_col) for row in rows]
    return float(np.mean(latitudes)), float(np.mean(longitudes))


def _pydeck_color_expression(
    value_col: str | None,
    *,
    positive: tuple[int, int, int] = (37, 99, 235),
) -> list[int] | str:
    if value_col is None:
        return [positive[0], positive[1], positive[2], 190]
    return (
        f"{value_col} >= 0 ? [{positive[0]}, {positive[1]}, {positive[2]}, 210] "
        f": [220, 38, 38, 210]"
    )


def _pydeck_radius_expression(value_col: str | None, radius_scale: float) -> float | str:
    if value_col is None:
        return radius_scale
    return f"Math.max(1, Math.abs({value_col})) * {radius_scale}"


def _pydeck_tooltip(columns: Sequence[str]) -> Mapping[str, str] | None:
    if not columns:
        return None
    lines = [f"{column}: {{{column}}}" for column in columns]
    return {"text": "\n".join(lines)}


def _write_pydeck_html(deck: Any, output_html: str | Path) -> Path:
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(output), open_browser=False)
    return output


__all__ = [
    "DEFAULT_PALETTE",
    "plot_backtest_metrics",
    "plot_changepoint_effects",
    "plot_cutoff_predictions",
    "plot_forecast",
    "plot_forecast_components",
    "plot_horizon_metrics",
    "plot_interval_calibration",
    "plot_metric_comparison",
    "plot_predicted_actual",
    "plot_residual_diagnostics",
    "plot_route_segments",
    "plot_seasonality_curve",
    "plot_spatial_points",
    "save_figure",
    "write_pydeck_point_map",
    "write_pydeck_route_map",
    "write_plot_report",
]
