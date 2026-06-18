from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cartoboost import __version__
from cartoboost.forecasting import (
    CartoBoostLagForecaster,
    ForecastFrame,
    ForecastMetricSet,
    NaiveForecaster,
    OptimizedThetaForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
    WeightedEnsembleForecaster,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic forecasting benchmark fixtures."
    )
    parser.add_argument("--output", default="artifacts/forecasting_benchmark.json")
    args = parser.parse_args()

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cartoboost_version": __version__,
        "datasets": [],
        "models": [
            "naive",
            "seasonal_naive",
            "theta",
            "optimized_theta",
            "cartoboost_lag",
            "weighted_ensemble",
        ],
        "metrics": {},
    }
    for name, frame in _fixtures().items():
        payload["datasets"].append(name)
        payload["metrics"][name] = _score_fixture(frame)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _fixtures() -> dict[str, ForecastFrame]:
    return {
        "trend_only": _single_fixture(lambda i: 20.0 + 0.8 * i),
        "weekly_seasonal": _single_fixture(lambda i: 30.0 + 0.2 * i + 5.0 * (i % 7 == 0)),
        "intermittent_sparse": _single_fixture(lambda i: 0.0 if i % 5 else 12.0 + 0.1 * i),
        "known_future_covariate": _single_fixture(lambda i: 25.0 + 2.0 * np.sin(i / 3.0)),
        "noisy_geotemporal": _single_fixture(lambda i: 18.0 + 0.4 * i + ((i * 17) % 9) * 0.3),
        "panel_lanes": _panel_fixture(),
    }


def _single_fixture(fn: Any) -> ForecastFrame:
    rows = [
        {
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "loads": float(fn(i)),
        }
        for i in range(70)
    ]
    return ForecastFrame.from_pandas(
        pd.DataFrame(rows),
        timestamp_col="date",
        target_col="loads",
        freq="D",
    )


def _panel_fixture() -> ForecastFrame:
    rows = []
    for lane, base in [("JFK->LGA", 20.0), ("LGA->EWR", 35.0), ("EWR->JFK", 28.0)]:
        for i in range(70):
            rows.append(
                {
                    "lane_id": lane,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "loads": base + 0.25 * i + (i % 7) * 0.7,
                }
            )
    return ForecastFrame.from_pandas(
        pd.DataFrame(rows),
        timestamp_col="date",
        target_col="loads",
        series_id_col="lane_id",
        freq="D",
    )


def _score_fixture(frame: ForecastFrame) -> dict[str, dict[str, float]]:
    horizon = 14
    data = frame.to_pandas()
    cutoff = data[frame.timestamp_col].drop_duplicates().sort_values().iloc[-horizon]
    train = data[data[frame.timestamp_col] < cutoff]
    test = data[data[frame.timestamp_col] >= cutoff]
    train_frame = ForecastFrame.from_pandas(
        train,
        timestamp_col=frame.timestamp_col,
        target_col=frame.target_col,
        series_id_col=frame.series_id_col,
        freq=frame.freq,
    )
    actual = test.copy()
    actual["series_id"] = (
        "__single__" if frame.series_id_col is None else actual[frame.series_id_col]
    )
    actual = actual.rename(columns={frame.timestamp_col: "timestamp", frame.target_col: "actual"})
    actual["horizon"] = actual.groupby("series_id").cumcount() + 1
    models = _models(frame)
    scores = {}
    for name, model in models.items():
        try:
            model.fit(train_frame)
            forecast = model.predict(horizon).to_pandas()
            merged = actual[["series_id", "timestamp", "horizon", "actual"]].merge(
                forecast[["series_id", "timestamp", "horizon", "mean"]],
                on=["series_id", "timestamp", "horizon"],
                how="inner",
            )
            metrics = ForecastMetricSet(seasonal_period=7).evaluate(
                merged["actual"],
                merged["mean"],
                horizon=merged["horizon"],
                series_id=merged["series_id"],
                y_train=train[frame.target_col],
            )
            scores[name] = {
                metric: float(metrics[metric])
                for metric in ("mae", "rmse", "mase", "wape", "smape", "bias")
                if metric in metrics and np.isfinite(metrics[metric])
            }
            scores[name]["coverage_80"] = 0.0
            scores[name]["coverage_95"] = 0.0
        except Exception as exc:
            scores[name] = {"error": str(exc)}
    return scores


def _models(frame: ForecastFrame) -> dict[str, Any]:
    lag = CartoBoostLagForecaster(
        lags=[1, 7, 14],
        rolling_windows=[7],
        regressor_params={
            "n_estimators": 8,
            "learning_rate": 0.2,
            "max_depth": 2,
            "min_samples_leaf": 1,
            "splitters": ["axis"],
        },
    )
    return {
        "naive": NaiveForecaster(),
        "seasonal_naive": SeasonalNaiveForecaster(season_length=7),
        "theta": ThetaForecaster(season_length=7),
        "optimized_theta": OptimizedThetaForecaster(season_length=7),
        "cartoboost_lag": lag,
        "weighted_ensemble": WeightedEnsembleForecaster(
            models={"theta": ThetaForecaster(season_length=7), "naive": NaiveForecaster()},
            weights={"theta": 0.7, "naive": 0.3},
        ),
    }


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        dataset: {
            model: metrics.get("mae")
            for model, metrics in dataset_metrics.items()
            if "mae" in metrics
        }
        for dataset, dataset_metrics in payload["metrics"].items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
