#!/usr/bin/env python3
"""Run synthetic splitter acceptance checks and write JSON/Markdown artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from geoboost import GeoBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "target" / "validation" / "splitter_tests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    splitters: list[str],
    n_estimators: int = 1,
    learning_rate: float = 1.0,
    max_depth: int = 1,
    min_samples_leaf: int = 1,
    leaf_predictor: str = "constant",
    linear_leaf_features: list[str] | None = None,
    fuzzy: bool = False,
    fuzzy_bandwidth: float = 0.0,
) -> GeoBoostRegressor:
    model = GeoBoostRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_gain=0.0,
        splitters=splitters,
        leaf_predictor=leaf_predictor,
        linear_leaf_features=linear_leaf_features,
        fuzzy=fuzzy,
        fuzzy_bandwidth=fuzzy_bandwidth,
        l2_regularization=0.0,
        backend="rust",
    )
    model.fit(x, y)
    return model


def grid_2d(size: int = 31, low: float = -3.0, high: float = 3.0) -> np.ndarray:
    values = np.linspace(low, high, size)
    xx, yy = np.meshgrid(values, values)
    return np.column_stack([xx.ravel(), yy.ravel()])


def axis_metrics() -> dict[str, Any]:
    x = np.linspace(-3.0, 3.0, 49).reshape(-1, 1)
    y = np.where(x[:, 0] > 0.0, 10.0, -10.0)
    axis = fit_model(x, y, splitters=["axis"], min_samples_leaf=2)
    pred = axis.predict(x)
    return {
        "models": {"axis": {"train_rmse": rmse(y, pred)}},
        "inspection_metrics": {
            "left_prediction": float(axis.predict([[-2.0]])[0]),
            "right_prediction": float(axis.predict([[2.0]])[0]),
        },
        "future_checks": [
            "A depth-1 axis stump must solve a clean threshold fixture.",
            "Left and right inspection predictions should remain well separated.",
        ],
    }


def diagonal_metrics() -> dict[str, Any]:
    x = grid_2d()
    y = np.where(x[:, 0] + x[:, 1] > 0.0, 10.0, -10.0)
    axis = fit_model(x, y, splitters=["axis"], min_samples_leaf=4)
    diagonal = fit_model(x, y, splitters=["diagonal_2d"], min_samples_leaf=4)
    axis_pred = axis.predict(x)
    diagonal_pred = diagonal.predict(x)
    return {
        "models": {
            "axis": {"train_rmse": rmse(y, axis_pred)},
            "diagonal_2d": {"train_rmse": rmse(y, diagonal_pred)},
        },
        "inspection_metrics": {
            "axis_boundary_mae": mae(y, axis_pred),
            "diagonal_boundary_mae": mae(y, diagonal_pred),
            "negative_corner_prediction": float(diagonal.predict([[-2.0, -2.0]])[0]),
            "positive_corner_prediction": float(diagonal.predict([[2.0, 2.0]])[0]),
        },
        "future_checks": [
            "Diagonal splitters should beat a one-stump axis baseline on x + y boundaries.",
            "Corner probes catch swapped branch polarity.",
        ],
    }


def gaussian_metrics() -> dict[str, Any]:
    x = grid_2d(size=35)
    y = np.where(np.hypot(x[:, 0], x[:, 1]) <= 1.5, 10.0, -10.0)
    axis = fit_model(x, y, splitters=["axis"], min_samples_leaf=4)
    gaussian = fit_model(x, y, splitters=["gaussian_2d"], min_samples_leaf=4)
    axis_pred = axis.predict(x)
    gaussian_pred = gaussian.predict(x)
    return {
        "models": {
            "axis": {"train_rmse": rmse(y, axis_pred)},
            "gaussian_2d": {"train_rmse": rmse(y, gaussian_pred)},
        },
        "inspection_metrics": {
            "axis_radial_mae": mae(y, axis_pred),
            "gaussian_radial_mae": mae(y, gaussian_pred),
            "center_prediction": float(gaussian.predict([[0.0, 0.0]])[0]),
            "outer_prediction": float(gaussian.predict([[3.0, 3.0]])[0]),
        },
        "future_checks": [
            "Gaussian splitters should beat axis stumps on radial fixtures.",
            "Center and outer probes verify inside/outside routing.",
        ],
    }


def periodic_metrics() -> dict[str, Any]:
    hours = np.array([[float(hour)] for hour in range(24) for _ in range(3)])
    y = np.array([15.0 if hour[0] >= 22.0 or hour[0] <= 2.0 else -5.0 for hour in hours])
    axis = fit_model(hours, y, splitters=["axis"], min_samples_leaf=3)
    periodic = fit_model(hours, y, splitters=["periodic_time"], min_samples_leaf=3)
    axis_pred = axis.predict(hours)
    periodic_pred = periodic.predict(hours)
    return {
        "models": {
            "axis": {"train_rmse": rmse(y, axis_pred)},
            "periodic_time": {"train_rmse": rmse(y, periodic_pred)},
        },
        "inspection_metrics": {
            "axis_wrap_gap": abs(float(axis.predict([[23.0]])[0] - axis.predict([[1.0]])[0])),
            "periodic_wrap_gap": abs(
                float(periodic.predict([[23.0]])[0] - periodic.predict([[1.0]])[0])
            ),
            "periodic_edge_mid_gap": float(
                np.mean(periodic.predict([[23.0], [1.0]])) - periodic.predict([[12.0]])[0]
            ),
        },
        "future_checks": [
            "Periodic splitters must handle wraparound intervals such as 22..2.",
            "Equivalent edge-hour predictions should remain close across the period boundary.",
        ],
    }


def fuzzy_metrics() -> dict[str, Any]:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    hard = fit_model(x, y, splitters=["axis"])
    fuzzy = fit_model(x, y, splitters=["axis"], fuzzy=True, fuzzy_bandwidth=1.0)
    probes = np.linspace(1.0, 2.0, 101).reshape(-1, 1)
    hard_pred = hard.predict(probes)
    fuzzy_pred = fuzzy.predict(probes)
    return {
        "models": {
            "hard_axis": {"probe_rmse": rmse(np.linspace(0.0, 10.0, 101), hard_pred)},
            "fuzzy_axis": {"probe_rmse": rmse(np.linspace(0.0, 10.0, 101), fuzzy_pred)},
        },
        "inspection_metrics": {
            "hard_boundary_jump": abs(float(hard.predict([[1.49]])[0] - hard.predict([[1.51]])[0])),
            "fuzzy_boundary_jump": abs(
                float(fuzzy.predict([[1.49]])[0] - fuzzy.predict([[1.51]])[0])
            ),
            "fuzzy_midpoint_prediction": float(fuzzy.predict([[1.5]])[0]),
        },
        "future_checks": [
            "Fuzzy routing should smooth local boundary jumps.",
            "The exact threshold midpoint should blend both child leaves.",
        ],
    }


def linear_metrics() -> dict[str, Any]:
    x = np.linspace(0.0, 5.0, 30).reshape(-1, 1)
    y = 2.0 * x[:, 0] + 3.0
    constant = fit_model(x, y, splitters=["axis"], max_depth=1, min_samples_leaf=16)
    linear = fit_model(
        x,
        y,
        splitters=["axis"],
        max_depth=1,
        min_samples_leaf=16,
        leaf_predictor="linear",
        linear_leaf_features=["0"],
    )
    constant_pred = constant.predict(x)
    linear_pred = linear.predict(x)
    return {
        "models": {
            "constant_leaf": {"train_rmse": rmse(y, constant_pred)},
            "linear_leaf": {"train_rmse": rmse(y, linear_pred)},
        },
        "inspection_metrics": {
            "linear_low_prediction": float(linear.predict([[0.0]])[0]),
            "linear_high_prediction": float(linear.predict([[5.0]])[0]),
        },
        "future_checks": [
            "A linear leaf should solve a noiseless linear residual fixture.",
            "Endpoint predictions should preserve the learned local slope.",
        ],
    }


def sparse_metrics() -> dict[str, Any]:
    x = np.array([[7.0], [7.0], [3.0], [4.0], [9.0], [9.0], [8.0], [8.0]])
    y = np.array([30.0, 30.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
    axis = fit_model(x, y, splitters=["axis"], min_samples_leaf=2)
    sparse = fit_model(x, y, splitters=["sparse_set"], min_samples_leaf=2)
    axis_pred = axis.predict(x)
    sparse_pred = sparse.predict(x)
    return {
        "models": {
            "axis": {"train_rmse": rmse(y, axis_pred)},
            "sparse_set": {"train_rmse": rmse(y, sparse_pred)},
        },
        "inspection_metrics": {
            "toll_cell_prediction": float(sparse.predict([[7.0]])[0]),
            "cold_cell_prediction": float(sparse.predict([[3.0]])[0]),
            "observed_cell_count": float(len(set(x[:, 0]))),
            "dense_one_hot_columns_avoided": float(len(set(x[:, 0])) - x.shape[1]),
        },
        "future_checks": [
            "Sparse set routing should isolate high-cardinality membership effects.",
            "The fixture stores IDs directly and avoids dense one-hot expansion.",
        ],
    }


def learning_rate_metrics() -> dict[str, Any]:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    model = fit_model(x, y, splitters=["axis"], learning_rate=0.25)
    pred = model.predict(x)
    expected = np.array([3.75, 3.75, 6.25, 6.25])
    return {
        "models": {"axis_shrinkage": {"train_rmse": rmse(expected, pred)}},
        "inspection_metrics": {
            "expected_left_prediction": float(expected[0]),
            "actual_left_prediction": float(pred[0]),
            "expected_right_prediction": float(expected[-1]),
            "actual_right_prediction": float(pred[-1]),
        },
        "future_checks": [
            "The fitted tree must target the L2 negative gradient and apply learning_rate once.",
            "This catches accidentally fitting raw targets or double-applying shrinkage.",
        ],
    }


def collect_metrics() -> dict[str, Any]:
    return {
        "axis_threshold": axis_metrics(),
        "diagonal_2d": diagonal_metrics(),
        "gaussian_2d": gaussian_metrics(),
        "periodic_wraparound": periodic_metrics(),
        "fuzzy_axis": fuzzy_metrics(),
        "linear_leaf": linear_metrics(),
        "sparse_set": sparse_metrics(),
        "learning_rate_gradient_shrinkage": learning_rate_metrics(),
    }


def render_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# GeoBoost splitter acceptance metrics",
        "",
        "Synthetic checks are generated from deterministic fixtures. They prove exact behavior for",
        "fixed examples and qualitative splitter superiority on designed structures; they do not",
        "claim universal production superiority.",
        "",
        "| phase | model | metric | value |",
        "| --- | --- | --- | ---: |",
    ]
    for phase, phase_metrics in metrics.items():
        for model_name, model_metrics in phase_metrics["models"].items():
            for metric, value in model_metrics.items():
                lines.append(f"| {phase} | {model_name} | {metric} | {value:.8f} |")
    lines.extend(["", "## Inspection Metrics", ""])
    for phase, phase_metrics in metrics.items():
        lines.append(f"### {phase}")
        for metric, value in phase_metrics["inspection_metrics"].items():
            lines.append(f"- `{metric}`: {value:.8f}")
        lines.append("")
    return "\n".join(lines)


def assert_finite(metrics: dict[str, Any]) -> None:
    for phase_metrics in metrics.values():
        for group in ("models", "inspection_metrics"):
            values = phase_metrics[group]
            if group == "models":
                values = {
                    f"{model}.{metric}": value
                    for model, model_metrics in values.items()
                    for metric, value in model_metrics.items()
                }
            for name, value in values.items():
                if not math.isfinite(float(value)):
                    raise ValueError(f"non-finite metric {name}: {value}")


def main() -> None:
    args = parse_args()
    metrics = collect_metrics()
    assert_finite(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "acceptance_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "acceptance_metrics.md").write_text(
        render_markdown(metrics) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
