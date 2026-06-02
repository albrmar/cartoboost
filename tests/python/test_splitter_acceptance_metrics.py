from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


def test_splitter_acceptance_metrics_are_generated(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "splitter_tests"
    script = repo_root / "scripts" / "run_splitter_acceptance_metrics.py"

    try:
        subprocess.run(
            [sys.executable, str(script), "--output-dir", str(output_dir)],
            check=True,
            cwd=repo_root,
        )
    except ImportError as exc:
        raise AssertionError("Rust extension must be available for acceptance metrics") from exc

    metrics_path = output_dir / "acceptance_metrics.json"
    markdown_path = output_dir / "acceptance_metrics.md"
    assert metrics_path.exists()
    assert markdown_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(metrics) == {
        "axis_threshold",
        "diagonal_2d",
        "gaussian_2d",
        "periodic_wraparound",
        "fuzzy_axis",
        "linear_leaf",
        "sparse_set",
        "learning_rate_gradient_shrinkage",
    }

    for phase_metrics in metrics.values():
        assert phase_metrics["future_checks"]
        assert phase_metrics["models"]
        assert phase_metrics["inspection_metrics"]
        assert phase_metrics["acceptance_gates"]
        for model_metrics in phase_metrics["models"].values():
            for value in model_metrics.values():
                assert math.isfinite(value)
        for value in phase_metrics["inspection_metrics"].values():
            assert math.isfinite(value)
        for check in phase_metrics["acceptance_gates"]:
            assert check["passed"], check
            assert math.isfinite(check["actual"])
            assert math.isfinite(check["threshold"])

    assert metrics["axis_threshold"]["models"]["axis"]["train_rmse"] <= 1e-12
    assert (
        metrics["diagonal_2d"]["models"]["diagonal_2d"]["train_rmse"]
        < metrics["diagonal_2d"]["models"]["axis"]["train_rmse"]
    )
    assert (
        metrics["gaussian_2d"]["models"]["gaussian_2d"]["train_rmse"]
        < metrics["gaussian_2d"]["models"]["axis"]["train_rmse"]
    )
    assert (
        metrics["periodic_wraparound"]["models"]["periodic_time"]["train_rmse"]
        < metrics["periodic_wraparound"]["models"]["axis"]["train_rmse"]
    )
    assert (
        metrics["periodic_wraparound"]["inspection_metrics"]["periodic_wrap_gap"]
        < metrics["periodic_wraparound"]["inspection_metrics"]["axis_wrap_gap"]
    )
    assert (
        metrics["fuzzy_axis"]["inspection_metrics"]["fuzzy_boundary_jump"]
        < metrics["fuzzy_axis"]["inspection_metrics"]["hard_boundary_jump"]
    )
    assert metrics["fuzzy_axis"]["inspection_metrics"]["fuzzy_midpoint_prediction"] == 5.0
    assert (
        metrics["linear_leaf"]["models"]["linear_leaf"]["train_rmse"]
        < metrics["linear_leaf"]["models"]["constant_leaf"]["train_rmse"]
    )
    assert metrics["linear_leaf"]["models"]["linear_leaf"]["train_rmse"] <= 1e-10
    assert (
        metrics["sparse_set"]["models"]["sparse_set"]["train_rmse"]
        <= metrics["sparse_set"]["models"]["axis"]["train_rmse"]
    )
    assert (
        metrics["sparse_set"]["inspection_metrics"]["toll_cell_prediction"]
        > metrics["sparse_set"]["inspection_metrics"]["cold_cell_prediction"]
    )
    assert (
        metrics["learning_rate_gradient_shrinkage"]["models"]["axis_shrinkage"]["train_rmse"]
        <= 1e-12
    )
