from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class RecordingForecastFrame:
    def __init__(self, rows: list[tuple[str, str, float]], freq: str) -> None:
        self.rows = rows
        self.freq = freq


@pytest.fixture
def install_fake_native(monkeypatch: pytest.MonkeyPatch):
    def install(class_name: str):
        import cartoboost

        calls: list[tuple[Any, ...]] = []

        class RecordingModel:
            def __init__(self, **params: Any) -> None:
                calls.append(("init", params))

            def fit(self, frame: RecordingForecastFrame):
                calls.append(("fit", frame))
                return self

            def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                calls.append(("predict", args, kwargs))
                return {"args": args, "kwargs": kwargs}

        native = SimpleNamespace(ForecastFrame=RecordingForecastFrame)
        setattr(native, class_name, RecordingModel)
        monkeypatch.setattr(cartoboost, "_native", native, raising=False)
        return SimpleNamespace(calls=calls, frame_class=RecordingForecastFrame)

    return install
