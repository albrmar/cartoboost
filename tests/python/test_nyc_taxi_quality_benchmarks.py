from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from scripts.run_nyc_taxi_quality_benchmarks import (
    BenchmarkTask,
    ZoneContext,
    build_real_tasks,
    clean_tlc_frame,
    graph_augmented_split_features,
    pickup_demand_cold_zone_fraction,
    sample_tlc_frame,
)


def test_nyc_taxi_quality_benchmark_synthetic_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "nyc_taxi_benchmarks"
    script = repo_root / "scripts" / "run_nyc_taxi_quality_benchmarks.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-smoke",
            "--models",
            "mean",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=repo_root,
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert results["artifact_version"] == 1
    assert results["dataset"]["source"] == "synthetic_smoke"
    assert set(results["tasks"]) == {"duration", "fare", "pickup_demand"}

    for task in results["tasks"].values():
        assert set(task["splits"]) == {"random", "spatial_holdout"}
        for split in task["splits"].values():
            model = split["models"]["mean"]
            assert model["status"] == "ok"
            assert np.isfinite(model["metrics"]["rmse"])
            assert np.isfinite(model["metrics"]["mae"])
            assert np.isfinite(model["metrics"]["r2"])
            assert model["timing"]["train_seconds"] >= 0.0
            assert model["timing"]["predict_seconds"] >= 0.0
            assert model["timing"]["fit_predict_seconds"] >= 0.0
            assert model["timing"]["prediction_rows"] > 0.0
            assert model["timing"]["predict_rows_per_second"] > 0.0

    markdown = (output_dir / "results.md").read_text(encoding="utf-8")
    assert "NYC Taxi Model Quality Benchmarks" in markdown
    assert "Trip duration" in markdown
    assert "Pickup-zone demand" in markdown

    summary = mpimg.imread(output_dir / "metric_summary.png")
    assert summary.shape[0] >= 300
    assert summary.shape[1] >= 500
    assert float(np.std(summary[..., :3])) > 0.01

    speed_summary = mpimg.imread(output_dir / "speed_summary.png")
    assert speed_summary.shape[0] >= 300
    assert speed_summary.shape[1] >= 500
    assert float(np.std(speed_summary[..., :3])) > 0.01

    throughput = mpimg.imread(output_dir / "prediction_throughput.png")
    assert throughput.shape[0] >= 300
    assert throughput.shape[1] >= 500
    assert float(np.std(throughput[..., :3])) > 0.01


def test_nyc_taxi_quality_benchmark_skips_missing_optional_models(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "nyc_taxi_optional_skips"
    script = repo_root / "scripts" / "run_nyc_taxi_quality_benchmarks.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-smoke",
            "--models",
            "lightgbm,xgboost",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=repo_root,
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    for task in results["tasks"].values():
        for split in task["splits"].values():
            for model_name in ["lightgbm", "xgboost"]:
                model = split["models"][model_name]
                if model["status"] == "skipped":
                    assert "not installed" in model["reason"]


def test_real_pickup_demand_aggregates_full_cleaned_frame_when_rows_are_sampled():
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {
            "tpep_pickup_datetime": pandas.to_datetime(
                [
                    "2024-01-01 08:00:00",
                    "2024-01-01 08:15:00",
                    "2024-01-01 08:30:00",
                    "2024-01-02 09:00:00",
                ]
            ),
            "tpep_dropoff_datetime": pandas.to_datetime(
                [
                    "2024-01-01 08:10:00",
                    "2024-01-01 08:25:00",
                    "2024-01-01 08:40:00",
                    "2024-01-02 09:12:00",
                ]
            ),
            "passenger_count": [1.0, 1.0, 1.0, 1.0],
            "trip_distance": [1.0, 1.2, 1.1, 2.0],
            "fare_amount": [8.0, 9.0, 8.5, 12.0],
            "total_amount": [10.0, 11.0, 10.5, 14.0],
            "PULocationID": [10, 10, 10, 20],
            "DOLocationID": [20, 20, 20, 10],
        }
    )
    cleaned = clean_tlc_frame(frame)
    sampled_rows = sample_tlc_frame(cleaned, sample_size=2, seed=0)
    tasks = build_real_tasks(
        sampled_rows,
        {
            10: ZoneContext(borough_code=1, service_zone_code=4),
            20: ZoneContext(borough_code=2, service_zone_code=2),
        },
        demand_frame=cleaned,
    )

    duration = next(task for task in tasks if task.name == "duration")
    demand = next(task for task in tasks if task.name == "pickup_demand")
    assert len(duration.target) == 2
    demand_by_zone = {
        int(features[0]): float(target)
        for features, target in zip(demand.features, demand.target, strict=True)
    }
    assert demand_by_zone[10] == np.log1p(3.0)
    assert demand_by_zone[20] == np.log1p(1.0)


def test_nyc_taxi_quality_benchmark_runs_neural_and_graph_models(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "nyc_taxi_neural_graph"
    script = repo_root / "scripts" / "run_nyc_taxi_quality_benchmarks.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-smoke",
            "--tasks",
            "duration",
            "--models",
            (
                "cartoboost_neural,cartoboost_graph_node2vec,cartoboost_graph_graphsage,"
                "cartoboost_graph_hetero_graphsage,cartoboost_graph_hinsage"
            ),
            "--graph-dim",
            "2",
            "--graph-epochs",
            "1",
            "--neural-dim",
            "2",
            "--cartoboost-n-estimators",
            "4",
            "--model-workers",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=repo_root,
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    task = results["tasks"]["duration"]
    for split in task["splits"].values():
        assert split["models"]["cartoboost_neural"]["status"] == "ok"
        assert split["models"]["cartoboost_neural"]["config"]["neural_dim"] > 0
        for model_name, family in {
            "cartoboost_graph_node2vec": "node2vec",
            "cartoboost_graph_graphsage": "graphsage",
            "cartoboost_graph_hetero_graphsage": "hetero_graphsage",
            "cartoboost_graph_hinsage": "hinsage",
        }.items():
            assert split["models"][model_name]["status"] == "ok"
            assert split["models"][model_name]["config"]["graph_family"] == family
            assert split["models"][model_name]["config"]["graph_edges"] > 0


def test_pickup_demand_graph_uses_zone_context_topology_for_cold_holdout():
    task = BenchmarkTask(
        name="pickup_demand",
        display_name="Pickup-zone demand",
        description="fixture",
        features=np.asarray(
            [
                [1.0, 8.0, 0.0],
                [1.0, 9.0, 0.0],
                [2.0, 8.0, 0.0],
                [3.0, 8.0, 0.0],
                [4.0, 8.0, 0.0],
            ],
            dtype=float,
        ),
        target=np.asarray([1.0, 1.2, 1.4, 1.6, 1.8], dtype=float),
        pickup_zones=np.asarray([1, 1, 2, 3, 4], dtype=int),
        feature_names=["PULocationID", "hour", "dayofweek"],
        sparse_sets={
            "pickup_zone": [[1], [1], [2], [3], [4]],
            "pickup_borough": [[1], [1], [1], [2], [2]],
            "pickup_service_zone": [[4], [4], [4], [2], [2]],
        },
        zone_adjacency={1: [2], 2: [1, 3], 3: [2, 4], 4: [3]},
    )
    train_indices = np.asarray([0, 1, 2], dtype=int)
    test_indices = np.asarray([3, 4], dtype=int)

    class Args:
        graph_dim = 2
        graph_epochs = 1
        seed = 7

    train_augmented, test_augmented, config = graph_augmented_split_features(
        task,
        train_indices,
        test_indices,
        task.features[train_indices],
        task.features[test_indices],
        Args(),
        graph_family="graphsage",
    )

    assert config["graph_topology"] == "zone_adjacency_borough_service"
    assert config["adjacency_edges"] > 0
    assert config["borough_edges"] > 0
    assert config["service_zone_edges"] > 0
    assert config["context_hub_nodes"] > 0
    assert train_augmented.shape[1] == test_augmented.shape[1]
    assert config["graph_feature_count"] == train_augmented.shape[1] - task.features.shape[1]
    assert float(np.std(test_augmented[:, task.features.shape[1] :])) > 0.0


def test_pickup_demand_cold_zone_fraction_detects_spatial_holdout():
    task = BenchmarkTask(
        name="pickup_demand",
        display_name="Pickup-zone demand",
        description="fixture",
        features=np.asarray(
            [
                [1.0, 8.0, 0.0],
                [1.0, 9.0, 0.0],
                [2.0, 8.0, 0.0],
                [3.0, 8.0, 0.0],
            ],
            dtype=float,
        ),
        target=np.asarray([1.0, 1.2, 1.4, 1.6], dtype=float),
        pickup_zones=np.asarray([1, 1, 2, 3], dtype=int),
        feature_names=["PULocationID", "hour", "dayofweek"],
        sparse_sets={"pickup_zone": [[1], [1], [2], [3]]},
    )

    assert (
        pickup_demand_cold_zone_fraction(
            task,
            np.asarray([0, 1, 2], dtype=int),
            np.asarray([3], dtype=int),
        )
        == 1.0
    )
    assert (
        pickup_demand_cold_zone_fraction(
            task,
            np.asarray([0, 2, 3], dtype=int),
            np.asarray([1], dtype=int),
        )
        == 0.0
    )
