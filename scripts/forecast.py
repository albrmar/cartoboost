from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "python" / "cartoboost" / "forecasting" / "cli.py"


spec = spec_from_file_location("cartoboost_forecasting_cli", CLI_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load forecasting CLI from {CLI_PATH}")
module = module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
main = module.main


if __name__ == "__main__":
    raise SystemExit(main())
