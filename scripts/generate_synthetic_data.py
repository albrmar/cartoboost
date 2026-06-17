#!/usr/bin/env python3
"""Generate small synthetic CSV datasets for CartoBoost examples.

The output format is intentionally simple:

* feature columns are named f0, f1, ...
* target is named target
* ranking data includes a query_id column

No CartoBoost package import is required, which keeps this usable while the public
API is still being built.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON config file")
    parser.add_argument("--output", type=Path, help="CSV output path")
    parser.add_argument(
        "--kind",
        choices=("regression", "binary", "ranking"),
        default="regression",
        help="Dataset shape to generate",
    )
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))

    cli_values = {
        "kind": args.kind,
        "rows": args.rows,
        "features": args.features,
        "groups": args.groups,
        "seed": args.seed,
        "noise": args.noise,
        "missing_rate": args.missing_rate,
    }
    if args.output:
        cli_values["output"] = str(args.output)

    # Config files provide defaults; explicit CLI output always wins.
    merged = {**cli_values, **config}
    if args.output:
        merged["output"] = str(args.output)
    return merged


def make_features(rng: random.Random, feature_count: int) -> list[float]:
    base = rng.gauss(0.0, 1.0)
    return [
        base * (0.25 if index % 3 == 0 else 0.05) + rng.gauss(0.0, 1.0)
        for index in range(feature_count)
    ]


def score_row(features: list[float], rng: random.Random, noise: float) -> float:
    linear = sum((index + 1) * value for index, value in enumerate(features[:5]))
    interaction = 0.8 * features[0] * features[1] if len(features) > 1 else 0.0
    periodic = math.sin(features[2]) if len(features) > 2 else 0.0
    return linear / 8.0 + interaction + periodic + rng.gauss(0.0, noise)


def mask_missing(features: list[float], rng: random.Random, missing_rate: float) -> list[str]:
    values: list[str] = []
    for value in features:
        if missing_rate > 0.0 and rng.random() < missing_rate:
            values.append("")
        else:
            values.append(f"{value:.8f}")
    return values


def generate(config: dict[str, Any]) -> None:
    kind = str(config["kind"])
    rows = int(config["rows"])
    features = int(config["features"])
    groups = int(config.get("groups", 100))
    seed = int(config["seed"])
    noise = float(config["noise"])
    missing_rate = float(config.get("missing_rate", 0.0))
    output = Path(str(config.get("output", f"examples/data/{kind}.csv")))

    if rows <= 0:
        raise ValueError("rows must be positive")
    if features <= 0:
        raise ValueError("features must be positive")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0.0, 1.0)")

    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [f"f{index}" for index in range(features)]
    if kind == "ranking":
        fieldnames = ["query_id", *fieldnames]
    fieldnames.append("target")

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(rows):
            xs = make_features(rng, features)
            score = score_row(xs, rng, noise)
            row = dict(
                zip(
                    [f"f{index}" for index in range(features)],
                    mask_missing(xs, rng, missing_rate),
                    strict=True,
                )
            )

            if kind == "binary":
                probability = 1.0 / (1.0 + math.exp(-score))
                row["target"] = str(int(rng.random() < probability))
            elif kind == "ranking":
                row["query_id"] = f"q{row_index % groups:04d}"
                row["target"] = str(max(0, min(4, int(round(score + 2.0)))))
            else:
                row["target"] = f"{score:.8f}"

            writer.writerow(row)


def main() -> None:
    generate(load_config(parse_args()))


if __name__ == "__main__":
    main()
