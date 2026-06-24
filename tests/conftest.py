"""Shared fixture helpers for CartoBoost contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
GOLDENS_DIR = TESTS_DIR / "goldens"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def goldens_dir() -> Path:
    return GOLDENS_DIR


@pytest.fixture
def neighborhood_points() -> dict[str, Any]:
    return load_json(FIXTURES_DIR / "neighborhood_points.geojson")


@pytest.fixture
def delivery_zones() -> dict[str, Any]:
    return load_json(FIXTURES_DIR / "delivery_zones.geojson")


@pytest.fixture
def expected_neighborhood_boosts() -> dict[str, Any]:
    return load_json(GOLDENS_DIR / "neighborhood_boosts.json")
