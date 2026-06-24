from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from cartoboost import CartoBoostRegressor


def test_committed_parity_fixture_matches_saved_rust_artifact():
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures" / "parity"
    fixture = json.loads((fixture_dir / "parity_fixture.json").read_text(encoding="utf-8"))

    model = CartoBoostRegressor.load(fixture_dir / fixture["model_path"])
    pred = model.predict(fixture["rows"])

    np.testing.assert_allclose(pred, fixture["expected_predictions"], rtol=0.0, atol=1e-12)


def test_parity_fixture_generation_is_deterministic(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    generator = repo_root / "tests" / "generate_parity_fixture.py"
    committed_dir = repo_root / "tests" / "fixtures" / "parity"

    subprocess.run(
        [sys.executable, str(generator), "--output-dir", str(tmp_path)],
        check=True,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "python")},
    )

    generated = json.loads((tmp_path / "parity_fixture.json").read_text(encoding="utf-8"))
    committed = json.loads((committed_dir / "parity_fixture.json").read_text(encoding="utf-8"))
    assert generated == committed

    generated_model = CartoBoostRegressor.load(tmp_path / generated["model_path"])
    committed_model = CartoBoostRegressor.load(committed_dir / committed["model_path"])
    np.testing.assert_allclose(
        generated_model.predict(generated["rows"]),
        committed_model.predict(committed["rows"]),
        rtol=0.0,
        atol=1e-12,
    )
