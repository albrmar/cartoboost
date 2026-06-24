from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

EXPECTED_ARTIFACTS = {
    "README.md",
    "acceptance_metrics.json",
    "acceptance_metrics.md",
    "lane_heatmap.png",
    "hour_profile.png",
    "route_midpoint_cartometry.png",
}


def _assert_nonblank_image(path: Path) -> None:
    assert path.exists()
    image = mpimg.imread(path)
    assert image.ndim in {2, 3}
    assert image.shape[0] >= 400
    assert image.shape[1] >= 400

    rgb = image[..., :3] if image.ndim == 3 else image
    assert float(np.std(rgb)) > 0.01
    flattened = rgb.reshape(-1, rgb.shape[-1]) if rgb.ndim == 3 else rgb.reshape(-1, 1)
    sample = flattened[:: max(1, len(flattened) // 2048)]
    assert len(np.unique(np.round(sample, 3), axis=0)) >= 12


def test_lane_level_acceptance_metrics_are_generated(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "lane_level_tests"
    script = repo_root / "scripts" / "run_lane_level_acceptance_metrics.py"

    try:
        subprocess.run(
            [sys.executable, str(script), "--output-dir", str(output_dir)],
            check=True,
            cwd=repo_root,
        )
    except ImportError as exc:
        raise AssertionError("Rust extension must be available for lane-level metrics") from exc

    assert {path.name for path in output_dir.iterdir()} == EXPECTED_ARTIFACTS

    metrics = json.loads((output_dir / "acceptance_metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {
        "sparse_lane_membership",
        "route_midpoint_cartometry",
        "wraparound_lane_hour",
        "regional_lane_boosting",
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

    sparse = metrics["sparse_lane_membership"]
    assert sparse["models"]["sparse_lane_id"]["train_rmse"] <= 1e-12
    assert sparse["inspection_metrics"]["hot_lane_margin"] > 200.0

    route = metrics["route_midpoint_cartometry"]
    assert (
        route["models"]["gaussian_midpoint"]["train_rmse"]
        < route["models"]["axis_midpoint"]["train_rmse"]
    )
    assert route["inspection_metrics"]["center_outer_margin"] > 100.0

    hour = metrics["wraparound_lane_hour"]
    assert hour["models"]["periodic_hour"]["train_rmse"] <= 1e-12
    assert hour["inspection_metrics"]["periodic_23_vs_1_gap"] <= 1e-12

    combined = metrics["regional_lane_boosting"]
    assert (
        combined["models"]["lane_spatial_temporal"]["holdout_rmse"]
        < combined["models"]["axis_only"]["holdout_rmse"]
    )
    assert combined["inspection_metrics"]["uses_hidden_simulator_metadata_in_training"] == 0.0

    markdown = (output_dir / "acceptance_metrics.md").read_text(encoding="utf-8")
    assert "sparse_lane_membership" in markdown
    assert "regional_lane_boosting" in markdown

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "16 lanes" in readme
    assert "No hidden simulator metadata" in readme

    for image_name in ["lane_heatmap.png", "hour_profile.png", "route_midpoint_cartometry.png"]:
        _assert_nonblank_image(output_dir / image_name)
