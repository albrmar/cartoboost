from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_forecast(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/forecast.py", *args],
        cwd=_repo_root(),
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
                "2026-01-04,142,16",
                "2026-01-05,142,18",
                "2026-01-01,236,20",
                "2026-01-02,236,21",
                "2026-01-03,236,23",
                "2026-01-04,236,24",
                "2026-01-05,236,26",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_fit_writes_artifact_resolved_config_and_forecast_csv(tmp_path: Path) -> None:
    data = tmp_path / "pickup.csv"
    _write_panel_csv(data)
    artifact_dir = tmp_path / "artifact"
    output = tmp_path / "forecast.csv"

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
        str(artifact_dir),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert (artifact_dir / "model.json").exists()
    resolved = json.loads((artifact_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["target_col"] == "pickup_demand"
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert set(rows[0]) == {
        "series_id",
        "timestamp",
        "model",
        "horizon",
        "forecast",
        "lower_80",
        "upper_80",
    }


def test_predict_uses_saved_artifact(tmp_path: Path) -> None:
    data = tmp_path / "pickup.csv"
    _write_panel_csv(data)
    artifact_dir = tmp_path / "artifact"
    train = _run_forecast(
        "fit",
        "--input",
        str(data),
        "--timestamp-col",
        "timestamp",
        "--target-col",
        "pickup_demand",
        "--series-id-col",
        "PULocationID",
        "--artifact-dir",
        str(artifact_dir),
    )
    assert train.returncode == 0, train.stderr
    output = tmp_path / "predict.csv"

    result = _run_forecast(
        "predict",
        "--artifact-dir",
        str(artifact_dir),
        "--horizon",
        "3",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert rows[0]["timestamp"] == "2026-01-06T00:00:00"


def test_backtest_writes_json_metrics_and_artifacts(tmp_path: Path) -> None:
    data = tmp_path / "pickup.csv"
    _write_panel_csv(data)
    output = tmp_path / "metrics.json"
    artifact_dir = tmp_path / "backtest"

    result = _run_forecast(
        "backtest",
        "--input",
        str(data),
        "--timestamp-col",
        "timestamp",
        "--target-col",
        "pickup_demand",
        "--series-id-col",
        "PULocationID",
        "--model",
        "naive",
        "--horizon",
        "2",
        "--output",
        str(output),
        "--artifact-dir",
        str(artifact_dir),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["command"] == "backtest"
    assert payload["metrics"]["model"] == "naive"
    assert payload["metrics"]["n"] == 4
    assert (artifact_dir / "backtest_forecasts.csv").exists()


def test_compare_exposes_all_models(tmp_path: Path) -> None:
    data = tmp_path / "pickup.csv"
    _write_panel_csv(data)
    output = tmp_path / "compare.json"

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
        str(output),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert {row["model"] for row in payload["metrics"]} == {
        "naive",
        "seasonal_naive",
        "theta",
        "optimized_theta",
        "ets",
        "auto_arima",
        "cartoboost_lag",
        "weighted_ensemble",
    }


def test_config_file_and_cli_overrides(tmp_path: Path) -> None:
    data = tmp_path / "pickup.csv"
    _write_panel_csv(data)
    config = tmp_path / "forecast.toml"
    config.write_text(
        f"""
input = "{data}"
timestamp_col = "timestamp"
target_col = "pickup_demand"
series_id_col = "PULocationID"
model = "mean"
horizon = 1
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "forecast.csv"

    result = _run_forecast(
        "fit",
        "--config",
        str(config),
        "--model",
        "drift",
        "--artifact-dir",
        str(tmp_path / "artifact"),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["model"] for row in rows} == {"drift"}


def test_invalid_config_exits_nonzero_with_helpful_error(tmp_path: Path) -> None:
    config = tmp_path / "forecast.toml"
    config.write_text('model = "not_a_model"\n', encoding="utf-8")

    result = _run_forecast("fit", "--config", str(config), "--artifact-dir", str(tmp_path))

    assert result.returncode != 0
    assert "unknown model 'not_a_model'" in result.stderr
