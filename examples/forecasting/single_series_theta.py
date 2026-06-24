from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "examples" / "forecasting" / "forecast_cli_input.csv"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp) / "artifact"
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
                "--model",
                "theta",
                "--horizon",
                "3",
                "--season-length",
                "7",
                "--artifact-dir",
                str(artifact_dir),
            ],
            cwd=ROOT,
            check=True,
        )
        print((artifact_dir / "model.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
