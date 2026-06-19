from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .base import BaseForecaster
from .global_models import CartoBoostLagForecaster
from .schema import ForecastFrame


@dataclass(frozen=True)
class AutoForecasterConfig:
    """Deterministic routing configuration for CartoBoost forecasting."""

    seed: int = 42
    season_length: int | None = None
    quantiles: tuple[float, ...] = ()
    n_threads: int | None = None
    no_hyperopt: bool = True


class AutoForecaster(BaseForecaster):
    """Deterministic out-of-the-box CartoBoost forecaster.

    The Python class is intentionally thin: model behavior is delegated to the
    Rust-backed forecaster selected by the fixed routing rules. This facade does
    not run hyperparameter search or benchmark-specific tuning.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        season_length: int | None = None,
        quantiles: Sequence[float] | None = None,
        n_threads: int | None = None,
        **cartoboost_params: Any,
    ) -> None:
        self.config = AutoForecasterConfig(
            seed=int(seed),
            season_length=season_length,
            quantiles=tuple(float(q) for q in (quantiles or ())),
            n_threads=n_threads,
        )
        self.cartoboost_params = dict(cartoboost_params)
        self._model: Any | None = None

    def fit(self, frame: ForecastFrame, *_args: Any, **_kwargs: Any) -> AutoForecaster:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("AutoForecaster.fit requires a ForecastFrame")
        model = CartoBoostLagForecaster(
            lags=self._default_lags(),
            rolling_windows=self._default_windows(),
            calendar_features=True,
            **self.cartoboost_params,
        )
        model.fit(frame)
        self._model = model
        self._mark_fitted()
        return self

    def predict(self, horizon: int, *_args: Any, **_kwargs: Any) -> Any:
        self._check_is_fitted()
        horizon = self.validate_horizon(horizon)
        return self._model.predict(horizon)

    def get_metadata(self) -> dict[str, Any]:
        self._check_is_fitted()
        metadata = dict(self._model.metadata_)
        metadata["auto_forecaster"] = {
            "seed": self.config.seed,
            "season_length": self.config.season_length,
            "quantiles": list(self.config.quantiles),
            "no_hyperopt": self.config.no_hyperopt,
            "selected_model": self._model.__class__.__name__,
        }
        return metadata

    @property
    def metadata_(self) -> dict[str, Any]:
        return self.get_metadata()

    def _default_lags(self) -> list[int]:
        season = self.config.season_length
        lags = [1, 2, 3, 7, 14, 28]
        if season is not None and season > 0:
            lags.append(int(season))
        return sorted(set(lags))

    def _default_windows(self) -> list[int]:
        windows = [7, 14, 28]
        season = self.config.season_length
        if season is not None and season > 1:
            windows.append(int(season))
        return sorted(set(windows))


def evaluate_m4(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Use scripts/forecasting_m4.py --committed --no-hyperopt for M4 scoring")


def evaluate_m5(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise RuntimeError(
        "Use scripts/forecasting_m5.py --committed --official-wrmsse --no-hyperopt for M5 scoring"
    )


def evaluate_m6_proxy(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Use scripts/forecasting_library_benchmark.py --source m6 for proxy scoring")


def evaluate_m6_official_style(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise RuntimeError(
        "Use scripts/forecasting_m6.py --committed --official-style --no-hyperopt for M6 scoring"
    )
