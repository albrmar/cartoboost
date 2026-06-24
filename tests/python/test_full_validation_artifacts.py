from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_full_validation_artifacts_include_splitter_and_lane_metrics():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "run_full_validation.py"

    subprocess.run([sys.executable, str(script)], check=True, cwd=repo_root)

    validation_dir = repo_root / "target" / "validation"
    summary = json.loads((validation_dir / "metrics.json").read_text(encoding="utf-8"))

    assert summary["artifact_version"] == 1
    assert summary["splitter_phase_count"] == 9
    assert summary["lane_level_phase_count"] == 4
    assert "gaussian_2d" in summary["splitter_phases"]
    assert "regional_lane_boosting" in summary["lane_level_phases"]
    assert (validation_dir / "splitter_tests" / "acceptance_metrics.md").exists()
    assert (validation_dir / "lane_level_tests" / "acceptance_metrics.md").exists()

    report = (validation_dir / "report.md").read_text(encoding="utf-8")
    assert "splitter phases: 9" in report
    assert "lane-level phases: 4" in report
