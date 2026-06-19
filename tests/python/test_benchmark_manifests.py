from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

aggregate_module = importlib.import_module("benchmarks.runners.aggregate_results")
manifest_module = importlib.import_module("benchmarks.runners.manifest")
significance_module = importlib.import_module("benchmarks.runners.significance")

aggregate = aggregate_module.aggregate
read_jsonl = aggregate_module.read_jsonl
load_all_tracks = manifest_module.load_all_tracks
validate_configs = manifest_module.validate_configs
average_ranks = significance_module.average_ranks
paired_bootstrap_ci = significance_module.paired_bootstrap_ci


def test_public_benchmark_manifests_are_valid() -> None:
    validate_configs()
    specs = load_all_tracks()

    assert {spec.name for spec in specs} == {"forecasting", "graph", "spatial", "tabular"}


def test_aggregate_results_reports_confidence_intervals(tmp_path: Path) -> None:
    rows = [
        {"task_id": "fare", "model_family": "cartoboost", "metric": "rmse", "value": 1.0},
        {"task_id": "fare", "model_family": "cartoboost", "metric": "rmse", "value": 2.0},
        {"task_id": "fare", "model_family": "gbdt", "metric": "rmse", "value": 3.0},
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = aggregate(read_jsonl(path))

    cartoboost = [
        row
        for row in summary["metrics"]
        if row["task_id"] == "fare" and row["model_family"] == "cartoboost"
    ][0]
    assert cartoboost["n"] == 2
    assert cartoboost["mean"] == 1.5
    assert cartoboost["ci95_low"] < cartoboost["mean"] < cartoboost["ci95_high"]


def test_paired_bootstrap_ci_uses_paired_deltas() -> None:
    observed, low, high = paired_bootstrap_ci(
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        iterations=100,
        seed=11,
    )

    assert observed == -1.0
    assert low == -1.0
    assert high == -1.0


def test_average_ranks_respects_metric_direction() -> None:
    ranks = average_ranks(
        [
            {"cartoboost": 1.0, "gbdt": 2.0},
            {"cartoboost": 3.0, "gbdt": 2.0},
        ],
        lower_is_better=True,
    )

    assert ranks == {"cartoboost": 1.5, "gbdt": 1.5}
