from __future__ import annotations

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
