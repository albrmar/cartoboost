from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


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
