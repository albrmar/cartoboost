from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_forecast(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root() / "python")
    return subprocess.run(
        [sys.executable, "-m", "cartoboost.forecasting.cli", *args],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
    )


def _write_panel_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "timestamp,PULocationID,pickup_demand",
                "2026-01-01,142,10",
                "2026-01-02,142,12",
                "2026-01-03,142,14",
                "2026-01-01,236,20",
                "2026-01-02,236,21",
                "2026-01-03,236,23",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_fit_writes_artifact_with_rust_theta_binding(tmp_path: Path) -> None:
    data = tmp_path / "pickup.csv"
    _write_panel_csv(data)

    result = _run_forecast(
        "fit",
        "--input",
        str(data),
        "--timestamp-col",
        "timestamp",
        "--target-col",
        "pickup_demand",
        "--series-id-col",
        "PULocationID",
        "--model",
        "theta",
        "--horizon",
        "2",
        "--artifact-dir",
        str(tmp_path / "artifact"),
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "artifact" / "resolved_config.json").exists()
    assert (tmp_path / "artifact" / "model.json").exists()


def test_predict_exits_nonzero_until_rust_artifact_binding_exists(tmp_path: Path) -> None:
    result = _run_forecast(
        "predict",
        "--artifact-dir",
        str(tmp_path / "artifact"),
        "--horizon",
        "3",
        "--output",
        str(tmp_path / "predict.csv"),
    )

    assert result.returncode != 0
    assert "Rust binding for forecasting artifact loading/prediction" in result.stderr


def test_compare_exposes_all_models_but_does_not_fake_metrics(tmp_path: Path) -> None:
    data = tmp_path / "pickup.csv"
    _write_panel_csv(data)

    result = _run_forecast(
        "compare",
        "--input",
        str(data),
        "--timestamp-col",
        "timestamp",
        "--target-col",
        "pickup_demand",
        "--series-id-col",
        "PULocationID",
        "--model",
        "all",
        "--horizon",
        "2",
        "--output",
        str(tmp_path / "compare.json"),
    )

    assert result.returncode != 0
    assert "Rust binding for forecasting model comparison is not available" in result.stderr


def test_config_file_and_cli_overrides_use_rust_seasonal_naive_binding(tmp_path: Path) -> None:
    data = tmp_path / "pickup.csv"
    _write_panel_csv(data)
    config = tmp_path / "forecast.toml"
    config.write_text(
        f"""
input = "{data}"
timestamp_col = "timestamp"
target_col = "pickup_demand"
series_id_col = "PULocationID"
model = "naive"
horizon = 1
""".strip(),
        encoding="utf-8",
    )

    result = _run_forecast(
        "fit",
        "--config",
        str(config),
        "--model",
        "seasonal_naive",
        "--season-length",
        "2",
        "--artifact-dir",
        str(tmp_path / "artifact"),
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "artifact" / "model.json").exists()


def test_invalid_config_exits_nonzero_with_helpful_error(tmp_path: Path) -> None:
    config = tmp_path / "forecast.toml"
    config.write_text('model = "not_a_model"\n', encoding="utf-8")

    result = _run_forecast("fit", "--config", str(config), "--artifact-dir", str(tmp_path))

    assert result.returncode != 0
    assert "unknown model 'not_a_model'" in result.stderr
