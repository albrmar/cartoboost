#!/usr/bin/env python3
"""Run a deterministic embedding-table benchmark for structured vs neural-hybrid.

Usage:
  python scripts/run_neural_embedding_benchmark.py --output target/validation/neural_benchmark.json
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from cartoboost import (
    CartoBoostRegressor,
    NeuralEmbeddingRegressor,
)

DEFAULT_OUTPUT = ROOT / "target" / "validation" / "neural_embedding_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=2_000)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--n-cells", type=int, default=128)
    parser.add_argument("--n-neural-dim", type=int, default=16)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-sklearn", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    mae: float
    fit_ms: float
    predict_ms: float


def ms(start: float) -> float:
    return float((time.perf_counter() - start) * 1000.0)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_absolute_error(actual, predicted))


def synthetic_dataset(
    *,
    n_rows: int,
    n_features: int,
    n_cells: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_rows, n_features))
    ids = rng.integers(1, n_cells + 1, size=n_rows, dtype=np.uint64)

    cell_effect = np.sin(ids.astype(np.float64) / 11.0) + (ids % 7) * 0.05
    hour = rng.integers(0, 24, size=n_rows)
    hourly_effect = np.where((hour >= 17) & (hour <= 19), 1.2, -0.3)

    y = (
        2.5 * x[:, 0]
        - 1.2 * x[:, 1]
        + 0.35 * x[:, 2]
        + 0.25 * x[:, 3] ** 2
        + 0.6 * cell_effect
        + 0.2 * hourly_effect
        + rng.normal(0.0, 0.4, size=n_rows)
    )
    return x, ids.astype(np.float64), y


def benchmark_cartoboost(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kwargs: dict[str, Any],
    train_frac: float,
) -> BenchmarkResult:
    split = x.shape[0] - int(x.shape[0] * train_frac)
    x_train, x_test = x[:-split], x[-split:]
    y_train, y_test = y[:-split], y[-split:]

    model = CartoBoostRegressor(**kwargs)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_ms = ms(start)

    start = time.perf_counter()
    pred = model.predict(x_test)
    predict_ms = ms(start)

    return BenchmarkResult(
        name="cartoboost",
        mae=mae(y_test, pred),
        fit_ms=fit_ms,
        predict_ms=predict_ms,
    )


def benchmark_sklearn(x: np.ndarray, y: np.ndarray, *, train_frac: float) -> BenchmarkResult:
    split = x.shape[0] - int(x.shape[0] * train_frac)
    x_train, x_test = x[:-split], x[-split:]
    y_train, y_test = y[:-split], y[-split:]

    model = GradientBoostingRegressor(
        n_estimators=80,
        max_depth=4,
        random_state=42,
    )
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_ms = ms(start)

    start = time.perf_counter()
    pred = model.predict(x_test)
    predict_ms = ms(start)

    return BenchmarkResult(
        name="sklearn_gbr",
        mae=mae(y_test, pred),
        fit_ms=fit_ms,
        predict_ms=predict_ms,
    )


def benchmark_neural(
    x: np.ndarray,
    ids: np.ndarray,
    y: np.ndarray,
    *,
    dim: int,
    train_frac: float,
) -> BenchmarkResult:
    split = x.shape[0] - int(x.shape[0] * train_frac)
    x_train, x_test = x[:-split], x[-split:]
    id_train, id_test = ids[:-split], ids[-split:]
    y_train, y_test = y[:-split], y[-split:]

    model = NeuralEmbeddingRegressor(
        dim=dim,
        id_column=None,
        drop_id_column=False,
    )
    start = time.perf_counter()
    model.fit(x_train, y_train, ids=id_train)
    fit_ms = ms(start)

    start = time.perf_counter()
    pred = model.predict(x_test, ids=id_test)
    predict_ms = ms(start)

    return BenchmarkResult(
        name="neural_embedding_hybrid",
        mae=mae(y_test, pred),
        fit_ms=fit_ms,
        predict_ms=predict_ms,
    )


def main() -> None:
    args = parse_args()

    x, ids, y = synthetic_dataset(
        n_rows=args.n_rows,
        n_features=args.n_features,
        n_cells=args.n_cells,
        seed=args.seed,
    )

    carto = benchmark_cartoboost(
        x,
        y,
        train_frac=args.train_frac,
        kwargs={
            "n_estimators": 50,
            "learning_rate": 0.07,
            "max_depth": 5,
            "min_samples_leaf": 1,
            "min_gain": 0.0,
            "splitters": ["axis_histogram:256"],
        },
    )

    neural = benchmark_neural(x, ids, y, dim=args.n_neural_dim, train_frac=args.train_frac)

    results = [carto, neural]

    if args.include_sklearn:
        results.append(benchmark_sklearn(x, y, train_frac=args.train_frac))

    payload = {
        "seed": args.seed,
        "n_rows": args.n_rows,
        "n_features": args.n_features,
        "n_cells": args.n_cells,
        "neural_dim": args.n_neural_dim,
        "train_frac": args.train_frac,
        "results": {
            result.name: {
                "mae": result.mae,
                "fit_ms": result.fit_ms,
                "predict_ms": result.predict_ms,
            }
            for result in results
        },
    }
    payload["hybrid_vs_baseline_improvement"] = (
        results[0].mae - results[1].mae if results[0].name == "cartoboost" else 0.0
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for name, metrics in payload["results"].items():
        print(
            f"{name:22} | MAE {metrics['mae']:.4f} "
            f"| fit {metrics['fit_ms']:.1f} ms "
            f"| predict {metrics['predict_ms']:.1f} ms"
        )

    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
