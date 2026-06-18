from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "examples" / "forecasting" / "forecast_cli_input.csv"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = Path(tmp) / "backtest_metrics.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "forecast.py"),
                "backtest",
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
                "--output",
                str(metrics),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            print(metrics.read_text(encoding="utf-8"))
        else:
            print(result.stderr.strip())


if __name__ == "__main__":
    main()
