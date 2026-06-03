#!/usr/bin/env python3
"""Generate GeoBoost validation artifacts used by local and CI smoke checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "target" / "validation"
SPLITTER_DIR = VALIDATION_DIR / "splitter_tests"
LANE_DIR = VALIDATION_DIR / "lane_level_tests"
PROOF_IMAGES = [
    "docs/assets/segmentation_diagonal_2d.png",
    "docs/assets/segmentation_gaussian_2d.png",
    "docs/assets/splitter_tests/phase_1_axis_threshold.png",
    "docs/assets/splitter_tests/phase_2_diagonal_2d.png",
    "docs/assets/splitter_tests/phase_3_gaussian_2d.png",
    "docs/assets/splitter_tests/phase_4_periodic_wraparound.png",
    "docs/assets/splitter_tests/phase_5_fuzzy_boundary.png",
    "docs/assets/splitter_tests/phase_6_linear_leaf.png",
    "docs/assets/splitter_tests/phase_7_sparse_set.png",
    "docs/assets/splitter_tests/phase_8_learning_rate_shrinkage.png",
    "docs/assets/splitter_tests/phase_9_fuzzy_periodic_wraparound.png",
]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "scripts/generate_segmentation_proofs.py"])
    run(
        [
            sys.executable,
            "scripts/run_splitter_acceptance_metrics.py",
            "--output-dir",
            str(SPLITTER_DIR),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/run_lane_level_acceptance_metrics.py",
            "--output-dir",
            str(LANE_DIR),
        ]
    )
    run([sys.executable, "scripts/run_v1_validation.py"])

    metrics = json.loads((SPLITTER_DIR / "acceptance_metrics.json").read_text(encoding="utf-8"))
    lane_metrics = json.loads((LANE_DIR / "acceptance_metrics.json").read_text(encoding="utf-8"))
    v1_metrics = json.loads(
        (VALIDATION_DIR / "v1" / "v1_validation.json").read_text(encoding="utf-8")
    )
    summary = {
        "artifact_version": 1,
        "splitter_phase_count": len(metrics),
        "splitter_phases": sorted(metrics),
        "lane_level_phase_count": len(lane_metrics),
        "lane_level_phases": sorted(lane_metrics),
        "v1_phase_count": len(v1_metrics["phases"]),
        "v1_phases": sorted(v1_metrics["phases"]),
        "proof_images": PROOF_IMAGES,
    }
    (VALIDATION_DIR / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# GeoBoost full validation",
        "",
        f"- splitter phases: {summary['splitter_phase_count']}",
        f"- lane-level phases: {summary['lane_level_phase_count']}",
        f"- v1 validation phases: {summary['v1_phase_count']}",
        "- proof images regenerated: yes",
        "",
        "See `target/validation/splitter_tests/acceptance_metrics.md` for detailed checks.",
        "See `target/validation/lane_level_tests/acceptance_metrics.md` for lane-level checks.",
        "See `target/validation/v1/v1_validation.md` for the v1 release-candidate report.",
        "",
    ]
    (VALIDATION_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
