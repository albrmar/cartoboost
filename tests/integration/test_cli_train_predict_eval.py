from __future__ import annotations

import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cargo", "run", "--quiet", "-p", "geoboost-cli", "--", *args],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
    )


def _write_training_csv(tmp_path: Path) -> Path:
    path = tmp_path / "train.csv"
    path.write_text(
        "x1,x2,target\n0,0,1\n1,0,2\n0,1,3\n1,1,4\n",
        encoding="utf-8",
    )
    return path


def test_cli_train_accepts_max_depth_zero(tmp_path: Path) -> None:
    data = _write_training_csv(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text('target = "target"\nmax_depth = 0\n', encoding="utf-8")
    model = tmp_path / "model.geoboost"

    result = _run_cli(
        "train",
        "--data",
        str(data),
        "--config",
        str(config),
        "--model-out",
        str(model),
    )

    assert result.returncode == 0, result.stderr
    assert model.exists()


def test_cli_predict_rejects_wrong_feature_count(tmp_path: Path) -> None:
    data = _write_training_csv(tmp_path)
    model = tmp_path / "model.geoboost"
    train = _run_cli("train", "--data", str(data), "--model-out", str(model))
    assert train.returncode == 0, train.stderr
    input_csv = tmp_path / "predict.csv"
    input_csv.write_text("x1,x2,extra\n0,0,99\n1,1,99\n", encoding="utf-8")

    result = _run_cli("predict", "--model", str(model), "--input", str(input_csv))

    assert result.returncode != 0
    assert "model expects 2" in result.stderr
