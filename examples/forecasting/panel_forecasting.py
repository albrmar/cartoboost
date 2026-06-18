from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "examples" / "forecasting" / "forecast_cli_input.csv"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / "panel_artifact"
        forecast_csv = Path(tmp) / "panel_forecast.csv"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "forecast.py"),
                "fit",
                "--input",
                str(INPUT),
                "--timestamp-col",
                "timestamp",
                "--target-col",
                "pickup_demand",
                "--series-id-col",
                "PULocationID",
                "--freq",
                "D",
                "--model",
                "cartoboost",
                "--horizon",
                "2",
                "--artifact-dir",
                str(artifact_dir),
                "--output",
                str(forecast_csv),
            ],
            cwd=ROOT,
            check=True,
        )
        print((artifact_dir / "resolved_config.json").read_text(encoding="utf-8"))
        print(forecast_csv.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
