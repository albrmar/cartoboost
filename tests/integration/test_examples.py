from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_example(*args: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def test_taxi_od_graph_example_runs() -> None:
    payload = run_example(
        "examples/03_taxi_od_graph_regression.py",
        "--sample-size",
        "600",
        "--graph-epochs",
        "1",
        "--graph-dim",
        "3",
        "--n-estimators",
        "8",
    )

    assert payload["rows"] == 600
    assert payload["graph_augmented"]["feature_count"] > 4
    assert np.isfinite(payload["dense"]["rmse"])
    assert np.isfinite(payload["graph_augmented"]["rmse"])


def test_taxi_pickup_zone_graph_example_runs_and_improves_spatial_holdout() -> None:
    payload = run_example("examples/04_taxi_pickup_zone_graph.py")

    assert payload["task"] == "pickup_zone_demand_spatial_holdout"
    assert payload["graph_augmented"]["graph_edges"] > 0
    assert payload["graph_augmented"]["feature_count"] > 3
    assert payload["graph_augmented"]["rmse"] < payload["dense"]["rmse"]
