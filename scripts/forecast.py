from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python"

if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from cartoboost.forecasting.cli import main  # noqa: E402,I001


if __name__ == "__main__":
    raise SystemExit(main())
