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

    metrics = json.loads((SPLITTER_DIR / "acceptance_metrics.json").read_text(encoding="utf-8"))
    summary = {
        "artifact_version": 1,
        "splitter_phase_count": len(metrics),
        "splitter_phases": sorted(metrics),
        "proof_images": PROOF_IMAGES,
    }
    (VALIDATION_DIR / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# GeoBoost full validation",
        "",
        f"- splitter phases: {summary['splitter_phase_count']}",
        "- proof images regenerated: yes",
        "",
        "See `target/validation/splitter_tests/acceptance_metrics.md` for detailed checks.",
        "",
    ]
    (VALIDATION_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
