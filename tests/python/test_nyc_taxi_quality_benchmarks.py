from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_nyc_taxi_quality_benchmarks import (  # noqa: E402
    BenchmarkTask,
    ZoneContext,
    benchmark_needs_zone_centroids,
    build_real_tasks,
    cartoboost_schema,
    clean_tlc_frame,
    graph_augmented_split_features,
    lightgbm_comparison,
    pickup_demand_cold_zone_fraction,
    sample_tlc_frame,
)
from scripts.run_repeated_nyc_taxi_benchmarks import collect_quality  # noqa: E402


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
    jsonl_rows = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert results["artifact_version"] == 1
    assert results["dataset"]["source"] == "synthetic_smoke"
    assert len(results["dataset_hash"]) == 64
    assert results["dataset"]["dataset_hash"] == results["dataset_hash"]
    assert results["source_file_hashes"] == {}
    assert results["git_commit"] is None or len(results["git_commit"]) == 40
    assert results["benchmark_integrity"]["hpo"] == "fixed_settings_no_hpo"
    assert results["benchmark_integrity"]["split_modes"] == ["random", "spatial_holdout"]
    assert results["feature_access_policy"]["baseline_feature_access"]
    assert set(results["split_definitions"]) == {"random", "spatial_holdout"}
    assert results["model_roster"] == ["mean"]
    assert results["resource_usage"]["python"]
    assert set(results["tasks"]) == {"duration", "fare", "pickup_demand"}
    assert jsonl_rows
    assert {"task_id", "split_id", "model_family", "metric", "value"} <= set(jsonl_rows[0])
    assert {row["model_family"] for row in jsonl_rows} == {"mean"}

    aggregate_path = output_dir / "aggregate.json"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "runners" / "aggregate_results.py"),
            "--input",
            str(output_dir / "results.jsonl"),
            "--output",
            str(aggregate_path),
        ],
        check=True,
        cwd=repo_root,
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_keys = {
        (row["track"], row["task_id"], row["split_id"], row["model_family"], row["metric"])
        for row in aggregate["metrics"]
    }
    assert ("spatial", "duration", "random", "mean", "rmse") in aggregate_keys
    assert ("spatial", "duration", "spatial_holdout", "mean", "rmse") in aggregate_keys

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
                    assert (
                        "not installed" in model["reason"]
                        or "cold-zone spatial holdout" in model["reason"]
                    )


def test_nyc_taxi_quality_benchmark_reports_best_cartoboost_vs_lightgbm():
    payload = {
        "tasks": {
            "fare": {
                "splits": {
                    "random": {
                        "models": {
                            "cartoboost": {
                                "status": "ok",
                                "metrics": {"rmse": 0.15, "r2": 0.86},
                            },
                            "cartoboost_graph_node2vec": {
                                "status": "ok",
                                "metrics": {"rmse": 0.13, "r2": 0.89},
                            },
                            "lightgbm": {
                                "status": "ok",
                                "metrics": {"rmse": 0.17, "r2": 0.84},
                            },
                        },
                    },
                    "spatial_holdout": {
                        "models": {
                            "cartoboost": {
                                "status": "skipped",
                                "reason": "not applicable",
                            },
                            "lightgbm": {
                                "status": "skipped",
                                "reason": "not applicable",
                            },
                        },
                    },
                },
            },
        },
    }

    assert lightgbm_comparison(payload) == [
        {
            "task": "fare",
            "split": "random",
            "best_cartoboost_model": "cartoboost_graph_node2vec",
            "best_cartoboost_rmse": 0.13,
            "lightgbm_rmse": 0.17,
            "rmse_delta_vs_lightgbm": -0.04000000000000001,
            "best_cartoboost_r2": 0.89,
            "lightgbm_r2": 0.84,
            "r2_delta_vs_lightgbm": 0.050000000000000044,
            "winner": "cartoboost",
        }
    ]


def test_repeated_nyc_quality_summary_reports_cis_and_paired_deltas():
    results = [
        {
            "tasks": {
                "fare": {
                    "splits": {
                        "random": {
                            "models": {
                                "cartoboost": {
                                    "status": "ok",
                                    "metrics": {"rmse": 1.0, "mae": 0.7, "r2": 0.8},
                                    "timing": {
                                        "train_seconds": 2.0,
                                        "predict_seconds": 0.2,
                                        "predict_rows_per_second": 100.0,
                                    },
                                },
                                "lightgbm": {
                                    "status": "ok",
                                    "metrics": {"rmse": 1.2, "mae": 0.8, "r2": 0.7},
                                    "timing": {
                                        "train_seconds": 1.0,
                                        "predict_seconds": 0.1,
                                        "predict_rows_per_second": 200.0,
                                    },
                                },
                            },
                        }
                    },
                }
            }
        },
        {
            "tasks": {
                "fare": {
                    "splits": {
                        "random": {
                            "models": {
                                "cartoboost": {
                                    "status": "ok",
                                    "metrics": {"rmse": 1.1, "mae": 0.75, "r2": 0.78},
                                    "timing": {
                                        "train_seconds": 3.0,
                                        "predict_seconds": 0.3,
                                        "predict_rows_per_second": 90.0,
                                    },
                                },
                                "lightgbm": {
                                    "status": "ok",
                                    "metrics": {"rmse": 1.3, "mae": 0.85, "r2": 0.68},
                                    "timing": {
                                        "train_seconds": 1.1,
                                        "predict_seconds": 0.1,
                                        "predict_rows_per_second": 190.0,
                                    },
                                },
                            },
                        }
                    },
                }
            }
        },
    ]

    summary = collect_quality(results)
    cartoboost = summary["fare/random"]["models"]["cartoboost"]
    assert cartoboost["metrics"]["rmse"]["n"] == 2
    assert cartoboost["metrics"]["rmse"]["mean"] == pytest.approx(1.05)
    assert cartoboost["rmse_wins_or_ties"] == 2

    delta = summary["fare/random"]["paired_deltas"]["cartoboost_vs_lightgbm"]
    assert delta["rmse_delta_mean"] == pytest.approx(-0.2)
    assert delta["r2_delta_mean"] == pytest.approx(0.1)


def test_nyc_taxi_centroids_are_only_required_for_cartoboost_geometry():
    class Args:
        models = "lightgbm,mean"
        zone_treatment = "target_mean"
        cartoboost_splitters = "axis_histogram:512,diagonal_2d,gaussian_2d"

    assert not benchmark_needs_zone_centroids(Args())

    Args.models = "cartoboost"
    assert benchmark_needs_zone_centroids(Args())

    Args.cartoboost_splitters = "axis_histogram:512,periodic:24,sparse_set"
    assert not benchmark_needs_zone_centroids(Args())

    Args.zone_treatment = "raw"
    assert benchmark_needs_zone_centroids(Args())


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


def test_cartoboost_schema_marks_zone_geometry_spatial_without_dense_sparse_schema():
    task = BenchmarkTask(
        name="fare",
        display_name="Fare",
        description="fixture",
        features=np.asarray([[1.0, 2.0]], dtype=float),
        target=np.asarray([1.0], dtype=float),
        pickup_zones=np.asarray([1], dtype=int),
        feature_names=["PULocationID", "hour"],
        sparse_sets={"pickup_zone": [[1]]},
    )

    schema = cartoboost_schema(
        task,
        feature_names=[
            "PULocationID",
            "hour",
            "PULocationID_centroid_x",
            "PULocationID_centroid_y",
            "od_centroid_distance",
        ],
        dense_id_sets=True,
        include_sparse_sets=False,
    )

    assert schema == {
        "dense": [
            {"name": "PULocationID", "kind": "numeric"},
            {"name": "hour", "kind": "periodic", "period": 24},
            {"name": "PULocationID_centroid_x", "kind": "spatial"},
            {"name": "PULocationID_centroid_y", "kind": "spatial"},
            {"name": "od_centroid_distance", "kind": "numeric"},
        ],
        "sparse_sets": [],
    }


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
