#!/usr/bin/env python3
"""Run deterministic model benchmarks across dense, neural, and graph workloads."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "model_benchmarks"
GRAPH_MODEL_FAMILIES = {
    "cartoboost_graph": "graphsage",
    "cartoboost_graph_node2vec": "node2vec",
    "cartoboost_graph_graphsage": "graphsage",
    "cartoboost_graph_hetero_graphsage": "hetero_graphsage",
    "cartoboost_graph_hinsage": "hinsage",
}


@dataclass(frozen=True)
class Workload:
    name: str
    display_name: str
    description: str
    features: np.ndarray
    target: np.ndarray
    split_group: np.ndarray | None = None
    embedding_ids: np.ndarray | None = None
    graph_source: np.ndarray | None = None
    graph_target: np.ndarray | None = None
    graph_edges: list[tuple[int, int]] | None = None
    graph_node_features: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rows", type=int, default=2_400)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--datasets",
        default="normal,neural,graph",
        help="Comma-separated workloads from: normal, neural, graph",
    )
    parser.add_argument(
        "--models",
        default=(
            "mean,cartoboost,cartoboost_neural,cartoboost_graph_node2vec,"
            "cartoboost_graph_graphsage,cartoboost_graph_hetero_graphsage,"
            "cartoboost_graph_hinsage,xgboost,lightgbm"
        ),
        help=(
            "Comma-separated models from: mean, cartoboost, cartoboost_neural, "
            "cartoboost_graph, cartoboost_graph_node2vec, cartoboost_graph_graphsage, "
            "cartoboost_graph_hetero_graphsage, cartoboost_graph_hinsage, xgboost, lightgbm"
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument("--neural-dim", type=int, default=12)
    parser.add_argument("--graph-dim", type=int, default=8)
    parser.add_argument("--graph-epochs", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def dense_normal_workload(*, n_rows: int, seed: int) -> Workload:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 10))
    y = (
        1.7 * x[:, 0]
        - 0.9 * x[:, 1]
        + 0.45 * x[:, 2] ** 2
        + np.sin(x[:, 3] * 1.8)
        + 0.25 * x[:, 4] * x[:, 5]
        + rng.normal(0.0, 0.28, size=n_rows)
    )
    return Workload(
        name="normal",
        display_name="Normal dense",
        description="IID dense numeric regression with nonlinear feature interactions.",
        features=x.astype(np.float64),
        target=y.astype(np.float64),
    )


def neural_workload(*, n_rows: int, seed: int) -> Workload:
    rng = np.random.default_rng(seed + 17)
    n_cells = max(64, int(math.sqrt(n_rows) * 4))
    x = rng.normal(size=(n_rows, 8))
    ids = rng.integers(0, n_cells, size=n_rows, dtype=np.uint64)
    cell_float = ids.astype(np.float64)
    cell_effect = np.sin(cell_float / 5.0) + 0.5 * np.cos(cell_float / 13.0)
    y = (
        1.25 * x[:, 0]
        - 0.7 * x[:, 1]
        + 0.35 * x[:, 2] ** 2
        + 0.85 * cell_effect
        + rng.normal(0.0, 0.30, size=n_rows)
    )
    features = np.column_stack([x, cell_float])
    return Workload(
        name="neural",
        display_name="Neural ID",
        description=(
            "Dense regression with repeated cell IDs whose residual signal is learnable by "
            "embedding features."
        ),
        features=features.astype(np.float64),
        target=y.astype(np.float64),
        split_group=ids.astype(np.int64),
        embedding_ids=ids,
    )


def graph_workload(*, n_rows: int, seed: int) -> Workload:
    rng = np.random.default_rng(seed + 31)
    n_nodes = max(48, int(math.sqrt(n_rows) * 3))
    source = rng.integers(0, n_nodes, size=n_rows)
    target = (source + rng.integers(1, max(3, n_nodes // 6), size=n_rows)) % n_nodes
    x = rng.normal(size=(n_rows, 6))

    node_axis = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    node_features = np.column_stack(
        [
            np.sin(node_axis),
            np.cos(node_axis),
            np.linspace(-1.0, 1.0, n_nodes),
        ]
    )
    src_effect = 0.7 * node_features[source, 0] + 0.25 * node_features[source, 2]
    dst_effect = -0.4 * node_features[target, 1] + 0.15 * node_features[target, 2]
    flow_effect = np.sin((source - target).astype(float) / 4.0)
    y = (
        1.1 * x[:, 0]
        - 0.75 * x[:, 1]
        + 0.25 * x[:, 2] * x[:, 3]
        + src_effect
        + dst_effect
        + 0.55 * flow_effect
        + rng.normal(0.0, 0.26, size=n_rows)
    )
    features = np.column_stack([x, source.astype(float), target.astype(float)])
    edges = sorted({(int(src), int(dst)) for src, dst in zip(source, target, strict=True)})
    return Workload(
        name="graph",
        display_name="Graph source-target",
        description=(
            "Directed source-target regression where graph topology and node features carry "
            "predictive signal."
        ),
        features=features.astype(np.float64),
        target=y.astype(np.float64),
        split_group=source.astype(np.int64),
        graph_source=source,
        graph_target=target,
        graph_edges=edges,
        graph_node_features=node_features.astype(np.float64),
    )


def build_workloads(args: argparse.Namespace) -> list[Workload]:
    requested = [part.strip() for part in args.datasets.split(",") if part.strip()]
    builders = {
        "normal": dense_normal_workload,
        "neural": neural_workload,
        "graph": graph_workload,
    }
    unknown = sorted(set(requested) - set(builders))
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(unknown)}")
    return [builders[name](n_rows=args.n_rows, seed=args.seed) for name in requested]


def random_split(n_rows: int, *, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_rows)
    split = int(n_rows * train_frac)
    if split <= 0 or split >= n_rows:
        raise ValueError("--train-frac must keep both train and validation rows")
    return indices[:split], indices[split:]


def group_holdout_split(
    groups: np.ndarray,
    *,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    split = max(1, min(len(unique) - 1, int(len(unique) * train_frac)))
    train_groups = set(int(value) for value in unique[:split])
    train_mask = np.asarray([int(value) in train_groups for value in groups], dtype=bool)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(~train_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return random_split(len(groups), train_frac=train_frac, seed=seed)
    return train_idx, test_idx


def split_workload(
    workload: Workload,
    *,
    train_frac: float,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    splits = {"random": random_split(len(workload.target), train_frac=train_frac, seed=seed)}
    if workload.split_group is not None:
        splits["group_holdout"] = group_holdout_split(
            workload.split_group,
            train_frac=train_frac,
            seed=seed + 101,
        )
    return splits


def metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": rmse,
        "r2": float(r2_score(actual, predicted)),
    }


def timed_fit_predict(
    fit: Any,
    predict: Any,
    test_rows: int,
) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    fit()
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = np.asarray(predict(), dtype=np.float64)
    predict_seconds = time.perf_counter() - start
    rps = float(test_rows / predict_seconds) if predict_seconds > 0 else float("inf")
    return prediction, {
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "predict_rows_per_second": rps,
    }


def run_mean(train_y: np.ndarray, test_rows: int) -> tuple[np.ndarray, dict[str, float]]:
    value = 0.0

    def fit() -> None:
        nonlocal value
        value = float(np.mean(train_y))

    return timed_fit_predict(fit, lambda: np.full(test_rows, value), test_rows)


def cartoboost_model(args: argparse.Namespace) -> Any:
    from cartoboost import CartoBoostRegressor

    return CartoBoostRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=2,
        min_gain=0.0,
        splitters=["axis_histogram:256"],
        random_state=args.seed,
        n_threads=args.n_threads or None,
    )


def run_cartoboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    model = cartoboost_model(args)
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return prediction, timing, {"backend": getattr(model, "_backend_used", None)}


def run_neural(
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    if workload.embedding_ids is None:
        raise ValueError("neural benchmark requires embedding ids")
    from cartoboost import NeuralEmbeddingRegressor

    model = NeuralEmbeddingRegressor(
        dim=args.neural_dim,
        drop_id_column=False,
        id_column=None,
        random_state=args.seed,
        oof_folds=5,
        support_prior_strength=2.0,
        base_model_kwargs={
            "n_estimators": max(10, args.n_estimators // 2),
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": 2,
            "splitters": ["axis_histogram:256"],
        },
        final_model_kwargs={
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": 2,
            "splitters": ["axis_histogram:256"],
        },
    )
    train_x = workload.features[train_idx]
    test_x = workload.features[test_idx]
    train_y = workload.target[train_idx]
    train_ids = workload.embedding_ids[train_idx]
    test_ids = workload.embedding_ids[test_idx]
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y, ids=train_ids),
        lambda: model.predict(test_x, ids=test_ids),
        len(test_idx),
    )
    return (
        prediction,
        timing,
        {
            "embedding_dim": int(args.neural_dim),
            "oof_folds": 5,
            "support_prior_strength": 2.0,
            "fit_stages_ms": model.timings,
        },
    )


def graph_augmented_features(
    workload: Workload,
    train_idx: np.ndarray,
    args: argparse.Namespace,
    graph_family: str,
) -> np.ndarray:
    if (
        workload.graph_source is None
        or workload.graph_target is None
        or workload.graph_edges is None
        or workload.graph_node_features is None
    ):
        raise ValueError("graph benchmark requires source, target, edges, and node features")
    from cartoboost.graph import GraphFeatureTransformer

    train_pairs = {
        (int(workload.graph_source[index]), int(workload.graph_target[index]))
        for index in train_idx
    }
    train_edges = sorted(train_pairs) or workload.graph_edges
    from cartoboost.graph import (
        DirectionalFeature,
        DirectionalityConfig,
        GraphEmbeddingsConfig,
        GraphEncoderConfig,
        GraphEncoderFamily,
    )

    node_count = int(workload.graph_node_features.shape[0])
    if graph_family == "node2vec":
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.NODE2VEC,
            dim=int(args.graph_dim),
            walk_length=16,
            walks_per_node=4,
            window_size=4,
            epochs=int(args.graph_epochs),
            negative_samples=3,
            seed=int(args.seed),
        )
        fit_edges: list[tuple[int, int]] | list[tuple[int, int, int]] = train_edges
        fit_kwargs: dict[str, Any] = {}
    elif graph_family == "hetero_graphsage":
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.HINSAGE,
            hetero=True,
            input_dim=int(workload.graph_node_features.shape[1]),
            hidden_dims=(int(args.graph_dim),),
            epochs=int(args.graph_epochs),
            seed=int(args.seed),
        )
        fit_edges = [(source, target, 0) for source, target in train_edges]
        fit_kwargs = {"relation_ids": [0]}
    elif graph_family == "hinsage":
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.HINSAGE,
            input_dim=int(workload.graph_node_features.shape[1]),
            node_type_count=1,
            edge_type_triples=((0, 0, 0),),
            hidden_dims=(int(args.graph_dim),),
            epochs=int(args.graph_epochs),
            seed=int(args.seed),
            neighbor_samples=(8,),
        )
        fit_edges = [(source, target, 0) for source, target in train_edges]
        fit_kwargs = {"node_types": [0] * node_count}
    elif graph_family == "graphsage":
        encoder = GraphEncoderConfig(
            family=GraphEncoderFamily.GRAPHSAGE,
            input_dim=int(workload.graph_node_features.shape[1]),
            hidden_dims=(int(args.graph_dim),),
            epochs=int(args.graph_epochs),
            seed=int(args.seed),
        )
        fit_edges = train_edges
        fit_kwargs = {}
    else:
        raise ValueError(f"unsupported graph family {graph_family!r}")

    transformer = GraphFeatureTransformer.from_config(
        GraphEmbeddingsConfig(
            encoder=encoder,
            directionality=DirectionalityConfig(
                compute_asymmetry_features=True,
                directional_feature_prefix="graph",
                directional_features=(
                    DirectionalFeature.SOURCE_TARGET_AFFINITY,
                    DirectionalFeature.FLOW_IMBALANCE_RATIO,
                ),
            ),
        )
    )
    bundle = transformer.fit_transform(
        node_features=workload.graph_node_features,
        edges=fit_edges,
        node_count=node_count,
        directed=True,
        **fit_kwargs,
    )
    node_embeddings = bundle.embeddings.astype(np.float64)
    source_emb = node_embeddings[workload.graph_source]
    target_emb = node_embeddings[workload.graph_target]
    return np.hstack([workload.features, source_emb, target_emb])


def run_graph(
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    graph_family: str,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    augmented = graph_augmented_features(workload, train_idx, args, graph_family)
    prediction, timing, config = run_cartoboost(
        augmented[train_idx],
        workload.target[train_idx],
        augmented[test_idx],
        args,
    )
    config.update(
        {
            "graph_family": graph_family,
            "graph_dim": int(args.graph_dim),
            "graph_epochs": int(args.graph_epochs),
        }
    )
    return prediction, timing, config


def run_xgboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    xgboost = optional_import("xgboost")
    if xgboost is None:
        raise ImportError("xgboost is not installed")
    model = xgboost.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=args.n_threads or 0,
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return prediction, timing, {"tree_method": "hist"}


def run_lightgbm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    lightgbm = optional_import("lightgbm")
    if lightgbm is None:
        raise ImportError("lightgbm is not installed")
    model = lightgbm.LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=2**args.max_depth,
        random_state=args.seed,
        n_jobs=args.n_threads or -1,
        verbose=-1,
    )
    prediction, timing = timed_fit_predict(
        lambda: model.fit(train_x, train_y),
        lambda: model.predict(test_x),
        len(test_x),
    )
    return prediction, timing, {"num_leaves": int(2**args.max_depth)}


def run_one_model(
    model_name: str,
    workload: Workload,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_x = workload.features[train_idx]
    test_x = workload.features[test_idx]
    train_y = workload.target[train_idx]
    test_y = workload.target[test_idx]
    try:
        if model_name == "mean":
            prediction, timing = run_mean(train_y, len(test_idx))
            config: dict[str, Any] = {}
        elif model_name == "cartoboost":
            prediction, timing, config = run_cartoboost(train_x, train_y, test_x, args)
        elif model_name == "cartoboost_neural":
            if workload.embedding_ids is None:
                return {"status": "skipped", "reason": "workload has no embedding ids"}
            prediction, timing, config = run_neural(workload, train_idx, test_idx, args)
        elif model_name in GRAPH_MODEL_FAMILIES:
            if workload.graph_edges is None:
                return {"status": "skipped", "reason": "workload has no graph topology"}
            prediction, timing, config = run_graph(
                workload,
                train_idx,
                test_idx,
                args,
                GRAPH_MODEL_FAMILIES[model_name],
            )
        elif model_name == "xgboost":
            prediction, timing, config = run_xgboost(train_x, train_y, test_x, args)
        elif model_name == "lightgbm":
            prediction, timing, config = run_lightgbm(train_x, train_y, test_x, args)
        else:
            raise ValueError(f"unknown model {model_name!r}")
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": str(exc)}

    return {
        "status": "ok",
        "metrics": metric_summary(test_y, prediction),
        "timing": timing,
        "config": config,
    }


def run_suite(workloads: list[Workload], args: argparse.Namespace) -> dict[str, Any]:
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    valid = {
        "mean",
        "cartoboost",
        "cartoboost_neural",
        "xgboost",
        "lightgbm",
        *GRAPH_MODEL_FAMILIES,
    }
    unknown = sorted(set(models) - valid)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")

    payload: dict[str, Any] = {
        "artifact_version": 1,
        "seed": int(args.seed),
        "n_rows": int(args.n_rows),
        "train_frac": float(args.train_frac),
        "models_requested": models,
        "model_config": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "neural_dim": int(args.neural_dim),
            "graph_dim": int(args.graph_dim),
            "graph_epochs": int(args.graph_epochs),
            "graph_model_families": dict(GRAPH_MODEL_FAMILIES),
        },
        "workloads": {},
    }
    for workload in workloads:
        workload_report: dict[str, Any] = {
            "display_name": workload.display_name,
            "description": workload.description,
            "row_count": int(len(workload.target)),
            "feature_count": int(workload.features.shape[1]),
            "splits": {},
        }
        for split_name, (train_idx, test_idx) in split_workload(
            workload,
            train_frac=args.train_frac,
            seed=args.seed,
        ).items():
            split_report: dict[str, Any] = {
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "models": {},
            }
            for model_name in models:
                split_report["models"][model_name] = run_one_model(
                    model_name,
                    workload,
                    train_idx,
                    test_idx,
                    args,
                )
            workload_report["splits"][split_name] = split_report
        payload["workloads"][workload.name] = workload_report
    return payload


def ok_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload_name, workload in payload["workloads"].items():
        for split_name, split in workload["splits"].items():
            for model_name, result in split["models"].items():
                if result["status"] != "ok":
                    continue
                rows.append(
                    {
                        "workload": workload_name,
                        "split": split_name,
                        "model": model_name,
                        **result["metrics"],
                        **result["timing"],
                    }
                )
    return rows


def write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Model Benchmark Suite",
        "",
        (
            "This generated report compares CartoBoost against optional XGBoost and LightGBM "
            "baselines on deterministic synthetic workloads."
        ),
        "",
        "## Command",
        "",
        "`uv run --group dev --group bench python scripts/run_model_benchmark_suite.py`",
        "",
        "## Configuration",
        "",
        f"- Seed: `{payload['seed']}`",
        f"- Rows per workload: `{payload['n_rows']}`",
        f"- Train fraction: `{payload['train_frac']}`",
        f"- Models requested: `{', '.join(payload['models_requested'])}`",
        "",
        "## Results",
        "",
    ]
    for workload in payload["workloads"].values():
        lines.extend([f"### {workload['display_name']}", "", workload["description"], ""])
        for split_name, split in workload["splits"].items():
            lines.extend(
                [
                    f"#### {split_name}",
                    "",
                    f"Train rows: `{split['train_rows']}`; test rows: `{split['test_rows']}`.",
                    "",
                    "| Model | Status | MAE | RMSE | R2 | Train s | Predict rows/s |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for model_name, result in split["models"].items():
                if result["status"] == "ok":
                    metrics = result["metrics"]
                    timing = result["timing"]
                    lines.append(
                        f"| {model_name} | ok | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | "
                        f"{metrics['r2']:.4f} | {timing['train_seconds']:.4f} | "
                        f"{timing['predict_rows_per_second']:.0f} |"
                    )
                else:
                    lines.append(f"| {model_name} | skipped: {result['reason']} |  |  |  |  |  |")
            lines.append("")
    lines.extend(
        [
            "## Plots",
            "",
            "![MAE by workload and split](mae_by_model.png)",
            "",
            "![Training time by workload and split](train_time_by_model.png)",
            "",
            "![Prediction throughput by workload and split](prediction_throughput_by_model.png)",
            "",
            "## Interpretation Notes",
            "",
            "- The normal workload checks dense numeric behavior without ID or graph augmentation.",
            (
                "- The neural workload includes repeated IDs and a group holdout split, so "
                "`cartoboost_neural` should be read as an embedding augmentation check rather "
                "than a replacement for external neural networks."
            ),
            (
                "- The graph workload benchmarks node2vec, GraphSAGE, HeteroGraphSAGE, and "
                "HinSAGE feature augmentation from train topology before fitting CartoBoost "
                "on augmented source-target rows."
            ),
            (
                "- XGBoost and LightGBM rows are skipped when their optional benchmark "
                "dependencies are not installed."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric(rows: list[dict[str, Any]], metric: str, title: str, output_path: Path) -> None:
    if not rows:
        return
    labels = [f"{row['workload']}\n{row['split']}\n{row['model']}" for row in rows]
    values = [float(row[metric]) for row in rows]
    width = max(10, min(28, len(rows) * 0.55))
    fig, ax = plt.subplots(figsize=(width, 7))
    positions = np.arange(len(rows))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(rows)))
    ax.bar(positions, values, color=colors)
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, args.output_dir / "results.md")
    if not args.no_plots:
        rows = ok_rows(payload)
        plot_metric(
            rows, "mae", "MAE by model, workload, and split", args.output_dir / "mae_by_model.png"
        )
        plot_metric(
            rows,
            "train_seconds",
            "Training time by model, workload, and split",
            args.output_dir / "train_time_by_model.png",
        )
        plot_metric(
            rows,
            "predict_rows_per_second",
            "Prediction throughput by model, workload, and split",
            args.output_dir / "prediction_throughput_by_model.png",
        )


def main() -> None:
    args = parse_args()
    workloads = build_workloads(args)
    payload = run_suite(workloads, args)
    write_outputs(payload, args)
    print(f"Benchmark results written to {args.output_dir}")


if __name__ == "__main__":
    main()
