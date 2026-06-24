from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_geojson(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
