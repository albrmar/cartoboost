"""Load and validate public benchmark manifest files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = ROOT / "tracks"
CONFIGS_DIR = ROOT / "configs"


class ManifestError(ValueError):
    """Raised when a benchmark manifest is incomplete or inconsistent."""


@dataclass(frozen=True)
class TrackSpec:
    """Resolved manifest files for a benchmark track."""

    name: str
    datasets: dict[str, Any]
    tasks: dict[str, Any]
    metrics: dict[str, Any]
    splits: dict[str, Any]
    search_spaces: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value


def load_track(track: str, root: Path = TRACKS_DIR) -> TrackSpec:
    track_dir = root / track
    if not track_dir.exists():
        raise ManifestError(f"unknown benchmark track: {track}")

    loaded = {}
    for key, filename in {
        "datasets": "datasets.json",
        "tasks": "tasks.json",
        "metrics": "metrics.json",
        "splits": "splits.json",
        "search_spaces": "search_spaces.json",
    }.items():
        path = track_dir / filename
        if not path.exists():
            raise ManifestError(f"{track} is missing {filename}")
        loaded[key] = load_json(path)

    spec = TrackSpec(name=track, **loaded)
    validate_track(spec)
    return spec


def validate_track(spec: TrackSpec) -> None:
    dataset_ids = {item["id"] for item in _items(spec.datasets, "datasets", spec.name)}
    split_ids = {item["id"] for item in _items(spec.splits, "splits", spec.name)}
    metric_ids = {item["id"] for item in _items(spec.metrics, "metrics", spec.name)}
    search_ids = {
        item["model_family"] for item in _items(spec.search_spaces, "search_spaces", spec.name)
    }

    if "cartoboost" not in search_ids:
        raise ManifestError(f"{spec.name} must define a cartoboost search space")

    for task in _items(spec.tasks, "tasks", spec.name):
        _require(task, ["id", "dataset_id", "split_id", "primary_metric", "required_model_families"])
        if task["dataset_id"] not in dataset_ids:
            raise ManifestError(f"{spec.name}/{task['id']} references unknown dataset_id")
        if task["split_id"] not in split_ids:
            raise ManifestError(f"{spec.name}/{task['id']} references unknown split_id")
        if task["primary_metric"] not in metric_ids:
            raise ManifestError(f"{spec.name}/{task['id']} references unknown primary_metric")
        missing_models = set(task["required_model_families"]) - search_ids
        if missing_models:
            missing = ", ".join(sorted(missing_models))
            raise ManifestError(f"{spec.name}/{task['id']} missing search spaces: {missing}")


def load_config(name: str, root: Path = CONFIGS_DIR) -> dict[str, Any]:
    return load_json(root / f"{name}.json")


def validate_configs(root: Path = CONFIGS_DIR) -> None:
    seeds = load_config("seeds", root)
    _require(seeds, ["benchmark_seeds", "hpo_seeds"])
    if not seeds["benchmark_seeds"] or not seeds["hpo_seeds"]:
        raise ManifestError("seed lists must not be empty")

    budgets = load_config("budgets", root)
    for budget in _items(budgets, "budgets", "configs"):
        _require(budget, ["id", "max_trials", "max_wallclock_minutes", "early_stop_rounds"])
        if budget["max_trials"] <= 0 or budget["max_wallclock_minutes"] <= 0:
            raise ManifestError(f"invalid budget: {budget['id']}")

    baselines = load_config("required_baselines", root)
    for track in ["tabular", "spatial", "graph", "forecasting"]:
        if track not in baselines:
            raise ManifestError(f"required_baselines missing {track}")


def load_all_tracks(root: Path = TRACKS_DIR) -> list[TrackSpec]:
    return [load_track(path.name, root) for path in sorted(root.iterdir()) if path.is_dir()]


def _items(mapping: dict[str, Any], key: str, owner: str) -> list[dict[str, Any]]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{owner} must define a non-empty {key} list")
    for item in value:
        if not isinstance(item, dict):
            raise ManifestError(f"{owner}.{key} entries must be objects")
    return value


def _require(mapping: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ManifestError(f"missing required keys: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", action="append", help="Track to validate. Defaults to all.")
    args = parser.parse_args(argv)

    validate_configs()
    tracks = args.track or [path.name for path in sorted(TRACKS_DIR.iterdir()) if path.is_dir()]
    for track in tracks:
        spec = load_track(track)
        print(f"validated {spec.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

