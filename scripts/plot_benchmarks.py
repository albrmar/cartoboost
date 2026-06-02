#!/usr/bin/env python3
"""Plot Criterion benchmark estimates when benchmark artifacts are available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=Path("target/criterion"),
        help="Criterion output directory",
    )
    parser.add_argument("--output", type=Path, default=Path("benches/benchmark_summary.png"))
    parser.add_argument("--summary-json", type=Path, help="Write parsed estimates to JSON")
    return parser.parse_args()


def collect_estimates(root: Path) -> list[dict[str, float | str]]:
    estimates: list[dict[str, float | str]] = []
    for estimate_path in sorted(root.glob("**/new/estimates.json")):
        data = json.loads(estimate_path.read_text(encoding="utf-8"))
        mean = data.get("mean", {}).get("point_estimate")
        if mean is None:
            continue
        benchmark = str(estimate_path.relative_to(root).parent.parent)
        estimates.append({"benchmark": benchmark, "mean_ns": float(mean)})
    return estimates


def write_plot(estimates: list[dict[str, float | str]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; use --summary-json for a dependency-free summary"
        ) from exc

    labels = [str(item["benchmark"]) for item in estimates]
    values = [float(item["mean_ns"]) / 1_000_000.0 for item in estimates]

    output.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.5))
    axis.bar(labels, values, color="#2f6f73")
    axis.set_ylabel("mean time (ms)")
    axis.set_title("GeoBoost benchmark summary")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)


def main() -> None:
    args = parse_args()
    estimates = collect_estimates(args.criterion_dir)
    if not estimates:
        raise SystemExit(f"no Criterion estimates found under {args.criterion_dir}")

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(estimates, indent=2) + "\n", encoding="utf-8")

    write_plot(estimates, args.output)


if __name__ == "__main__":
    main()
