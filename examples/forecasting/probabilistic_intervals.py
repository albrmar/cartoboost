from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "examples" / "forecasting" / "forecast_cli_input.csv"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        forecast_csv = Path(tmp) / "interval_forecast.csv"
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
                "seasonal_naive",
                "--horizon",
                "4",
                "--artifact-dir",
                str(Path(tmp) / "artifact"),
                "--output",
                str(forecast_csv),
            ],
            cwd=ROOT,
            check=True,
        )
        with forecast_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                print(
                    row["PULocationID"] if "PULocationID" in row else row["series_id"],
                    row["timestamp"],
                    row["forecast"],
                    row["lower_80"],
                    row["upper_80"],
                )


if __name__ == "__main__":
    main()
