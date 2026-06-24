#!/usr/bin/env python3
"""Generate a deterministic v1 validation report for CartoBoost release checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "target" / "validation" / "v1"
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_lane_level_acceptance_metrics as lane_metrics  # noqa: E402
import run_splitter_acceptance_metrics as splitter_metrics  # noqa: E402

PHASE_METADATA: dict[str, dict[str, Any]] = {
    "axis_threshold": {
        "config": {"splitters": ["axis"], "n_estimators": 1, "max_depth": 1},
        "baseline": "constant initialization",
        "cartoboost": "axis threshold stump",
        "proves": "A clean dense threshold can be recovered by the axis splitter.",
        "does_not_prove": "General dense tabular accuracy or production calibration.",
    },
    "diagonal_2d": {
        "config": {"splitters": ["diagonal_2d"], "n_estimators": 1, "max_depth": 1},
        "baseline": "one axis stump on the same 2D boundary",
        "cartoboost": "diagonal 2D stump",
        "proves": "A declared oblique splitter can recover an x + y decision boundary.",
        "does_not_prove": "Arbitrary-angle spatial optimality or large-scale search speed.",
    },
    "gaussian_2d": {
        "config": {"splitters": ["gaussian_2d"], "n_estimators": 1, "max_depth": 1},
        "baseline": "one axis stump on the same radial boundary",
        "cartoboost": "Gaussian/radial 2D stump",
        "proves": "A radial splitter can recover a center-versus-outside fixture.",
        "does_not_prove": "Robust hotspot discovery on noisy production maps.",
    },
    "periodic_wraparound": {
        "config": {"splitters": ["periodic_time"], "n_estimators": 1, "max_depth": 1},
        "baseline": "axis hour split",
        "cartoboost": "periodic interval split",
        "proves": "Wraparound intervals such as late-night hours can route together.",
        "does_not_prove": "All calendar, timezone, or holiday effects.",
    },
    "fuzzy_axis": {
        "config": {
            "splitters": ["axis"],
            "fuzzy": True,
            "fuzzy_bandwidth": 1.0,
            "n_estimators": 1,
        },
        "baseline": "hard axis stump",
        "cartoboost": "fuzzy axis stump",
        "proves": "Boundary predictions can be smoothed by fractional fuzzy routing.",
        "does_not_prove": "Probabilistic uncertainty calibration.",
    },
    "linear_leaf": {
        "config": {
            "splitters": ["axis"],
            "leaf_predictor": "linear",
            "linear_leaf_features": ["0"],
        },
        "baseline": "constant leaves",
        "cartoboost": "linear leaf predictor",
        "proves": "A linear residual trend can be represented inside leaves.",
        "does_not_prove": "High-dimensional regularized linear modeling quality.",
    },
    "sparse_set": {
        "config": {"splitters": ["sparse_set"], "n_estimators": 1, "max_depth": 1},
        "baseline": "axis split over scalar IDs",
        "cartoboost": "sparse set membership split",
        "proves": "High-cardinality ID membership can be tested without dense one-hot columns.",
        "does_not_prove": "Memory profile on production-scale route-cell lists.",
    },
    "regional_lane_boosting": {
        "config": {
            "splitters": ["axis", "sparse_set", "gaussian_2d", "periodic_time"],
            "n_estimators": 4,
            "max_depth": 2,
        },
        "baseline": "axis-only lane model",
        "cartoboost": "combined sparse, spatial, temporal lane model",
        "proves": "Route, lane, and hour effects can combine on a deterministic fixture.",
        "does_not_prove": "Real marketplace lift or serving latency.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _gate_summary(phase_metrics: dict[str, Any]) -> dict[str, Any]:
    gates = phase_metrics.get("acceptance_gates", [])
    return {
        "passed": all(bool(gate["passed"]) for gate in gates),
        "passed_count": sum(1 for gate in gates if gate["passed"]),
        "total_count": len(gates),
        "gates": gates,
    }


def _assert_finite_report(report: dict[str, Any]) -> None:
    for phase_name, phase in report["phases"].items():
        for model_metrics in phase["models"].values():
            for value in model_metrics.values():
                if not math.isfinite(float(value)):
                    raise ValueError(f"non-finite model metric in {phase_name}: {value}")
        for value in phase["inspection_metrics"].values():
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite inspection metric in {phase_name}: {value}")


def _assert_gates(report: dict[str, Any]) -> None:
    failures = [
        f"{phase}.{gate['name']}"
        for phase, phase_report in report["phases"].items()
        for gate in phase_report["gate_summary"]["gates"]
        if not gate["passed"]
    ]
    if failures:
        raise AssertionError("v1 validation gates failed: " + ", ".join(failures))


def collect_report() -> dict[str, Any]:
    splitter = splitter_metrics.collect_metrics()
    lane = lane_metrics.collect_metrics()
    source_metrics = {
        **splitter,
        "regional_lane_boosting": lane["regional_lane_boosting"],
    }
    phases: dict[str, Any] = {}
    for phase_name, metadata in PHASE_METADATA.items():
        metrics = source_metrics[phase_name]
        phases[phase_name] = {
            **metadata,
            "models": metrics["models"],
            "inspection_metrics": metrics["inspection_metrics"],
            "gate_summary": _gate_summary(metrics),
        }
    report = {
        "artifact_version": 1,
        "scope": "deterministic v1 release-candidate validation fixtures",
        "claim_policy": (
            "These fixtures provide regression evidence for implemented behavior. "
            "They do not claim broad production superiority."
        ),
        "phases": phases,
    }
    _assert_finite_report(report)
    _assert_gates(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CartoBoost v1 Validation Report",
        "",
        report["claim_policy"],
        "",
        "| phase | baseline | cartoboost | gates |",
        "| --- | --- | --- | ---: |",
    ]
    for phase_name, phase in report["phases"].items():
        gate_summary = phase["gate_summary"]
        lines.append(
            f"| {phase_name} | {phase['baseline']} | {phase['cartoboost']} | "
            f"{gate_summary['passed_count']}/{gate_summary['total_count']} |"
        )
    lines.extend(["", "## Phase Details", ""])
    for phase_name, phase in report["phases"].items():
        lines.extend(
            [
                f"### {phase_name}",
                "",
                f"- config: `{json.dumps(phase['config'], sort_keys=True)}`",
                f"- proves: {phase['proves']}",
                f"- does not prove: {phase['does_not_prove']}",
                "",
                "| gate | result | actual | comparator | threshold |",
                "| --- | --- | ---: | --- | ---: |",
            ]
        )
        for gate in phase["gate_summary"]["gates"]:
            result = "PASS" if gate["passed"] else "FAIL"
            lines.append(
                f"| {gate['name']} | {result} | {gate['actual']:.12g} | "
                f"{gate['comparator']} | {gate['threshold']:.12g} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = collect_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v1_validation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "v1_validation.md").write_text(
        render_markdown(report) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
