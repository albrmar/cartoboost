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


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_rejects_unknown_splitter(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'splitter = "axis,spline"\n')

    result = _run_cli("inspect", "--config", str(config))

    assert result.returncode != 0
    assert "unknown splitter 'spline'" in result.stderr


def test_cli_rejects_unknown_leaf_predictor(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'leaf_predictor = "spline"\n')

    result = _run_cli("inspect", "--config", str(config))

    assert result.returncode != 0
    assert "unknown leaf_predictor 'spline'" in result.stderr


def test_cli_rejects_invalid_config_value(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'learning_rate = "fast"\n')

    result = _run_cli("inspect", "--config", str(config))

    assert result.returncode != 0
    assert "invalid config value for 'learning_rate'" in result.stderr


def test_cli_rejects_malformed_config_line(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "learning_rate 0.1\n")

    result = _run_cli("inspect", "--config", str(config))

    assert result.returncode != 0
    assert "invalid config line 1" in result.stderr
