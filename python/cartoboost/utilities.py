"""General Rust-backed numerical utilities."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def series_forecast(
    model: str, values: Sequence[float], horizon: int, **params: Any
) -> list[float]:
    """Forecast a numeric sequence with a Rust-backed single-series method.

    This is a general utility independent of `cartoboost.forecasting` frames.
    Supported native model names include `naive`, `seasonal_naive`, `theta`,
    `optimized_theta`, `ets`, `arima`, `auto_arima`, `local_level_kalman`, and
    `local_linear_trend_kalman`.
    """

    native = _native()
    return list(
        native.utility_series_forecast(
            str(model),
            [float(value) for value in values],
            int(horizon),
            json.dumps(params) if params else None,
        )
    )


def naive_forecast(values: Sequence[float], horizon: int) -> list[float]:
    """Forecast by repeating the final observed value."""

    return series_forecast("naive", values, horizon)


def seasonal_naive_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    season_length: int,
) -> list[float]:
    """Forecast by repeating the last observed seasonal cycle."""

    return series_forecast(
        "seasonal_naive",
        values,
        horizon,
        season_length=int(season_length),
    )


def theta_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    theta: float = 2.0,
    alpha: float = 0.5,
) -> list[float]:
    """Forecast with the Rust theta model."""

    return series_forecast("theta", values, horizon, theta=float(theta), alpha=float(alpha))


def optimized_theta_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    theta_grid: Sequence[float] = (1.0, 2.0),
    alpha_grid: Sequence[float] = (0.2, 0.5, 0.8),
) -> list[float]:
    """Forecast with deterministic Rust optimized-theta grid selection."""

    return series_forecast(
        "optimized_theta",
        values,
        horizon,
        theta_grid=[float(value) for value in theta_grid],
        alpha_grid=[float(value) for value in alpha_grid],
    )


def ets_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    alpha: float = 0.5,
    beta: float = 0.1,
    gamma: float | None = None,
    season_length: int | None = None,
) -> list[float]:
    """Forecast with the Rust additive ETS utility."""

    return series_forecast(
        "ets",
        values,
        horizon,
        alpha=float(alpha),
        beta=float(beta),
        gamma=None if gamma is None else float(gamma),
        season_length=None if season_length is None else int(season_length),
    )


def arima_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    p: int = 1,
    d: int = 0,
    q: int = 0,
) -> list[float]:
    """Forecast with the Rust ARIMA(p,d,q) utility."""

    return series_forecast("arima", values, horizon, p=int(p), d=int(d), q=int(q))


def auto_arima_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    max_p: int = 3,
    max_d: int = 1,
    max_q: int = 2,
) -> list[float]:
    """Forecast with deterministic Rust AutoARIMA candidate selection."""

    return series_forecast(
        "auto_arima",
        values,
        horizon,
        max_p=int(max_p),
        max_d=int(max_d),
        max_q=int(max_q),
    )


def local_linear_trend_kalman_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    level_process_variance: float = 0.05,
    trend_process_variance: float = 0.005,
    observation_variance: float = 1.0,
) -> list[float]:
    """Forecast with the Rust local-linear-trend Kalman state-space utility."""

    return series_forecast(
        "local_linear_trend_kalman",
        values,
        horizon,
        level_process_variance=float(level_process_variance),
        trend_process_variance=float(trend_process_variance),
        observation_variance=float(observation_variance),
    )


def local_level_kalman_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    level_process_variance: float = 0.05,
    observation_variance: float = 1.0,
) -> list[float]:
    """Forecast with the Rust local-level Kalman utility."""

    return series_forecast(
        "local_level_kalman",
        values,
        horizon,
        level_process_variance=float(level_process_variance),
        observation_variance=float(observation_variance),
    )


def local_level_kalman_filter(
    values: Sequence[float],
    *,
    level_process_variance: float = 0.05,
    observation_variance: float = 1.0,
    horizon: int = 0,
) -> dict[str, Any]:
    """Run the Rust local-level Kalman filter on a numeric series."""

    native = _native()
    payload = native.utility_local_level_kalman_filter(
        [float(value) for value in values],
        float(level_process_variance),
        float(observation_variance),
        int(horizon),
    )
    return json.loads(payload)


def intermittent_demand_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    method: str = "croston",
    alpha: float = 0.1,
    beta: float = 0.1,
) -> list[float]:
    """Forecast non-negative intermittent demand with Croston, SBA, or TSB."""

    native = _native()
    return list(
        native.utility_intermittent_demand_forecast(
            [float(value) for value in values],
            int(horizon),
            str(method),
            float(alpha),
            float(beta),
        )
    )


def croston_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    alpha: float = 0.1,
) -> list[float]:
    """Forecast intermittent demand with Croston's method."""

    return intermittent_demand_forecast(values, horizon, method="croston", alpha=alpha)


def sba_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    alpha: float = 0.1,
) -> list[float]:
    """Forecast intermittent demand with the Syntetos-Boylan approximation."""

    return intermittent_demand_forecast(values, horizon, method="sba", alpha=alpha)


def tsb_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> list[float]:
    """Forecast intermittent demand with the Teunter-Syntetos-Babai method."""

    return intermittent_demand_forecast(
        values,
        horizon,
        method="tsb",
        alpha=alpha,
        beta=beta,
    )


def kalman_filter(
    values: Sequence[float],
    *,
    level_process_variance: float = 0.05,
    trend_process_variance: float = 0.005,
    observation_variance: float = 1.0,
    horizon: int = 0,
) -> dict[str, Any]:
    """Run the Rust local-linear Kalman filter on a numeric series.

    This is a general utility, independent of `cartoboost.forecasting`.
    """

    native = _native()
    payload = native.utility_kalman_filter(
        [float(value) for value in values],
        float(level_process_variance),
        float(trend_process_variance),
        float(observation_variance),
        int(horizon),
    )
    return json.loads(payload)


def ordinary_kriging_predict(
    observations: Sequence[tuple[float, float, float]],
    targets: Sequence[tuple[float, float]],
    *,
    range: float = 1.0,
    nugget: float = 1.0e-6,
) -> list[dict[str, Any]]:
    """Predict target coordinates with Rust ordinary kriging.

    `observations` are `(x, y, value)` triples. `targets` are `(x, y)` pairs.
    This utility is independent of `cartoboost.forecasting`.
    """

    native = _native()
    rows = native.utility_ordinary_kriging_predict(
        [(float(x), float(y), float(value)) for x, y, value in observations],
        [(float(x), float(y)) for x, y in targets],
        float(range),
        float(nugget),
    )
    return [
        {"x": x, "y": y, "mean": mean, "weights": list(weights)} for x, y, mean, weights in rows
    ]


def _native() -> Any:
    try:
        from cartoboost import _native as native
    except ImportError as exc:
        raise NotImplementedError("cartoboost._native is required for utilities") from exc
    return native


__all__ = [
    "arima_forecast",
    "auto_arima_forecast",
    "ets_forecast",
    "kalman_filter",
    "croston_forecast",
    "local_level_kalman_forecast",
    "local_level_kalman_filter",
    "local_linear_trend_kalman_forecast",
    "intermittent_demand_forecast",
    "naive_forecast",
    "optimized_theta_forecast",
    "ordinary_kriging_predict",
    "seasonal_naive_forecast",
    "sba_forecast",
    "series_forecast",
    "theta_forecast",
    "tsb_forecast",
]
