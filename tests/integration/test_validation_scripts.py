from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def test_full_validation_script_uses_native_estimator_without_backend_toggle():
    repo_root = Path(__file__).resolve().parents[2]

    full_validation = (repo_root / "scripts" / "run_full_validation.py").read_text(encoding="utf-8")
    splitter_metrics = (repo_root / "scripts" / "run_splitter_acceptance_metrics.py").read_text(
        encoding="utf-8"
    )
    lane_metrics = (repo_root / "scripts" / "run_lane_level_acceptance_metrics.py").read_text(
        encoding="utf-8"
    )

    assert "scripts/run_splitter_acceptance_metrics.py" in full_validation
    assert "scripts/run_lane_level_acceptance_metrics.py" in full_validation
    assert "CartoBoostRegressor(" in splitter_metrics
    assert "CartoBoostRegressor(" in lane_metrics
    backend_kwarg = "backend" + "="
    assert backend_kwarg not in splitter_metrics
    assert backend_kwarg not in lane_metrics


def test_ci_installs_native_extension_before_validation_artifacts():
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    install_step = workflow.index("uv run --group dev maturin develop")
    validation_step = workflow.index("uv run --group dev python scripts/run_full_validation.py")

    assert install_step < validation_step


def test_model_benchmark_suite_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "model_benchmarks"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_model_benchmark_suite.py"),
            "--output-dir",
            str(output_dir),
            "--datasets",
            "normal",
            "--models",
            "mean,cartoboost",
            "--n-rows",
            "120",
            "--no-plots",
        ],
        cwd=repo_root,
        check=True,
    )

    results = output_dir / "results.json"
    report = output_dir / "results.md"
    assert results.exists()
    assert report.exists()
    assert "Normal dense" in report.read_text(encoding="utf-8")


def test_model_benchmark_suite_reports_best_cartoboost_vs_lightgbm():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "run_model_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_model_benchmark_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark_suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark_suite
    spec.loader.exec_module(benchmark_suite)

    payload = {
        "workloads": {
            "normal": {
                "splits": {
                    "random": {
                        "models": {
                            "cartoboost": {
                                "status": "ok",
                                "metrics": {"rmse": 0.42, "r2": 0.91},
                            },
                            "cartoboost_neural": {
                                "status": "skipped",
                                "reason": "not applicable",
                            },
                            "lightgbm": {
                                "status": "ok",
                                "metrics": {"rmse": 0.50, "r2": 0.88},
                            },
                        },
                    },
                },
            },
        },
    }

    rows = benchmark_suite.lightgbm_comparison(payload)

    assert rows == [
        {
            "workload": "normal",
            "split": "random",
            "best_cartoboost_model": "cartoboost",
            "best_cartoboost_rmse": 0.42,
            "lightgbm_rmse": 0.50,
            "rmse_delta_vs_lightgbm": -0.08000000000000002,
            "best_cartoboost_r2": 0.91,
            "lightgbm_r2": 0.88,
            "r2_delta_vs_lightgbm": 0.030000000000000027,
            "winner": "cartoboost",
        }
    ]


def test_model_benchmark_suite_graph_families_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "model_benchmarks_graph"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_model_benchmark_suite.py"),
            "--output-dir",
            str(output_dir),
            "--datasets",
            "graph",
            "--models",
            (
                "cartoboost_graph_node2vec,cartoboost_graph_graphsage,"
                "cartoboost_graph_hetero_graphsage,cartoboost_graph_hinsage"
            ),
            "--n-rows",
            "96",
            "--graph-dim",
            "2",
            "--graph-epochs",
            "1",
            "--n-estimators",
            "4",
            "--no-plots",
        ],
        cwd=repo_root,
        check=True,
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    models = results["workloads"]["graph"]["splits"]["random"]["models"]
    for model_name, family in {
        "cartoboost_graph_node2vec": "node2vec",
        "cartoboost_graph_graphsage": "graphsage",
        "cartoboost_graph_hetero_graphsage": "hetero_graphsage",
        "cartoboost_graph_hinsage": "hinsage",
    }.items():
        assert models[model_name]["status"] == "ok"
        assert models[model_name]["config"]["graph_family"] == family


def test_nyc_taxi_benchmark_graph_family_switch_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    for family in ["node2vec", "graphsage", "hetero_graphsage", "hinsage"]:
        output_dir = tmp_path / f"nyc_taxi_{family}"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_nyc_taxi_quality_benchmarks.py"),
                "--synthetic-smoke",
                "--output-dir",
                str(output_dir),
                "--tasks",
                "duration",
                "--models",
                "cartoboost_graph",
                "--graph-family",
                family,
                "--graph-dim",
                "2",
                "--graph-epochs",
                "1",
                "--cartoboost-n-estimators",
                "4",
            ],
            cwd=repo_root,
            check=True,
        )

        results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        assert results["model_config"]["graph_family"] == family
        model = results["tasks"]["duration"]["splits"]["random"]["models"]["cartoboost_graph"]
        assert model["status"] == "ok"
        assert model["config"]["graph_family"] == family
