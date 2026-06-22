from __future__ import annotations

import importlib
import json
import re
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
load_config = manifest_module.load_config
validate_configs = manifest_module.validate_configs
average_ranks = significance_module.average_ranks
paired_bootstrap_ci = significance_module.paired_bootstrap_ci


def test_public_benchmark_manifests_are_valid() -> None:
    validate_configs()
    specs = load_all_tracks()

    assert {spec.name for spec in specs} == {"forecasting", "graph", "spatial", "tabular"}


def test_non_forecast_required_baselines_are_concrete() -> None:
    baselines = load_config("required_baselines")

    assert baselines["tabular"] == [
        "cartoboost",
        "lightgbm",
        "xgboost",
        "mean",
        "deep_tabular_baseline",
    ]
    assert baselines["spatial"] == [
        "cartoboost",
        "cartoboost_neural",
        "cartoboost_graph",
        "lightgbm",
        "xgboost",
        "mean",
    ]
    assert baselines["graph"] == [
        "cartoboost",
        "cartoboost_graph",
        "node2vec_baseline",
        "graphsage_baseline",
        "tabularized_graph_baseline",
        "mean",
    ]


def test_non_forecast_dataset_identities_are_frozen() -> None:
    specs = {spec.name: spec for spec in load_all_tracks()}
    for track in ["tabular", "spatial", "graph"]:
        for dataset in specs[track].datasets["datasets"]:
            assert dataset["hash"].startswith("sha256:")
            assert dataset["hash"] != "to_be_frozen"
            assert dataset["source_url"]
            assert dataset["source_identity"]

    assert specs["tabular"].datasets["datasets"][0]["id"] == "sklearn_diabetes_regression_v1"
    assert specs["graph"].datasets["datasets"][0]["id"] == "zachary_karate_club_78_edge_v1"


def test_benchmark_navigation_links_resolve() -> None:
    docs = [
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "llms.txt",
        ROOT / "llms.txt",
    ]
    benchmark_links = set()
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        benchmark_links.update(re.findall(r"\((?:\./)?(docs/benchmarks/[^)]+\.md)\)", text))
        benchmark_links.update(re.findall(r": \./(docs/benchmarks/[^\s]+\.md)", text))
        benchmark_links.update(re.findall(r"\((benchmarks/[^)]+\.md)\)", text))

    resolved = []
    for link in benchmark_links:
        path = ROOT / "docs" / link if link.startswith("benchmarks/") else ROOT / link
        resolved.append(path)
        assert path.exists(), f"missing benchmark navigation target: {link}"

    assert ROOT / "docs" / "benchmarks" / "model-suite.md" in resolved
    assert ROOT / "docs" / "benchmarks" / "nyc-taxi.md" in resolved


def test_benchmark_docs_asset_paths_exist() -> None:
    docs = [
        ROOT / "docs" / "benchmarks" / "index.md",
        ROOT / "docs" / "benchmarks" / "model-suite.md",
        ROOT / "docs" / "benchmarks" / "nyc-taxi.md",
        ROOT / "docs" / "benchmarks" / "taxi-zone.md",
    ]
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for asset in sorted(set(re.findall(r"`(docs/assets/[^`]+)`", text))):
            path = ROOT / asset
            assert path.exists(), f"{doc.relative_to(ROOT)} references missing asset: {asset}"


def test_non_forecast_public_artifacts_exist() -> None:
    paths = [
        ROOT / "docs" / "assets" / "model_benchmarks_public" / "results.json",
        ROOT / "docs" / "assets" / "model_benchmarks_public" / "results.jsonl",
        ROOT / "docs" / "assets" / "model_benchmarks_public" / "results_aggregate.json",
        ROOT / "docs" / "assets" / "model_benchmarks_public" / "results.md",
    ]
    for path in paths:
        assert path.exists(), f"missing maintained public model benchmark artifact: {path}"

    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert set(payload["workloads"]) == {"diabetes", "karate"}

    aggregate = json.loads(paths[2].read_text(encoding="utf-8"))
    keys = {
        (row["track"], row["task_id"], row["split_id"], row["model_family"], row["metric"])
        for row in aggregate["metrics"]
    }
    assert ("tabular", "diabetes", "random", "cartoboost", "rmse") in keys
    assert ("graph", "karate", "random", "cartoboost", "rmse") in keys


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


def test_aggregate_results_preserves_track_and_split_identity(tmp_path: Path) -> None:
    rows = [
        {
            "track": "spatial",
            "task_id": "fare",
            "split_id": "random",
            "model_family": "cartoboost",
            "metric": "rmse",
            "value": 1.0,
        },
        {
            "track": "spatial",
            "task_id": "fare",
            "split_id": "spatial_holdout",
            "model_family": "cartoboost",
            "metric": "rmse",
            "value": 2.0,
        },
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = aggregate(read_jsonl(path))

    grouped = {(row["track"], row["split_id"]): row for row in summary["metrics"]}
    assert grouped[("spatial", "random")]["mean"] == 1.0
    assert grouped[("spatial", "spatial_holdout")]["mean"] == 2.0


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
