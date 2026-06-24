from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generate_synthetic_data",
    ROOT / "scripts" / "generate_synthetic_data.py",
)
assert SPEC is not None
generate_synthetic_data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generate_synthetic_data)
generate = generate_synthetic_data.generate


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_synthetic_regression_generation_is_deterministic(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    config = {
        "kind": "regression",
        "rows": 12,
        "features": 4,
        "seed": 42,
        "noise": 0.0,
        "missing_rate": 0.0,
    }

    generate({**config, "output": first})
    generate({**config, "output": second})

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    rows = _read_rows(first)
    assert len(rows) == 12
    assert set(rows[0]) == {"f0", "f1", "f2", "f3", "target"}
    assert all(float(row["target"]) == pytest.approx(float(row["target"])) for row in rows)


def test_synthetic_binary_and_ranking_shapes(tmp_path: Path):
    binary = tmp_path / "binary.csv"
    ranking = tmp_path / "ranking.csv"

    generate(
        {
            "kind": "binary",
            "rows": 20,
            "features": 3,
            "seed": 7,
            "noise": 0.1,
            "missing_rate": 0.0,
            "output": binary,
        }
    )
    generate(
        {
            "kind": "ranking",
            "rows": 20,
            "features": 3,
            "groups": 4,
            "seed": 7,
            "noise": 0.1,
            "missing_rate": 0.0,
            "output": ranking,
        }
    )

    binary_rows = _read_rows(binary)
    ranking_rows = _read_rows(ranking)

    assert {row["target"] for row in binary_rows}.issubset({"0", "1"})
    assert set(ranking_rows[0]) == {"query_id", "f0", "f1", "f2", "target"}
    assert {row["query_id"] for row in ranking_rows} == {"q0000", "q0001", "q0002", "q0003"}
    assert all(0 <= int(row["target"]) <= 4 for row in ranking_rows)


def test_synthetic_generation_validates_config(tmp_path: Path):
    base = {
        "kind": "regression",
        "rows": 4,
        "features": 2,
        "seed": 1,
        "noise": 0.0,
        "missing_rate": 0.0,
        "output": tmp_path / "out.csv",
    }

    with pytest.raises(ValueError, match="rows must be positive"):
        generate({**base, "rows": 0})
    with pytest.raises(ValueError, match="features must be positive"):
        generate({**base, "features": 0})
    with pytest.raises(ValueError, match="missing_rate"):
        generate({**base, "missing_rate": 1.0})


def test_synthetic_missing_rate_masks_feature_values_only(tmp_path: Path):
    output = tmp_path / "missing.csv"

    generate(
        {
            "kind": "regression",
            "rows": 30,
            "features": 5,
            "seed": 11,
            "noise": 0.0,
            "missing_rate": 0.5,
            "output": output,
        }
    )

    rows = _read_rows(output)
    feature_values = [row[f"f{index}"] for row in rows for index in range(5)]
    assert any(value == "" for value in feature_values)
    assert all(row["target"] != "" for row in rows)
