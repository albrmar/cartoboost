"""Serializable forecasting artifacts with JSON manifests and tabular forecasts."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ForecastArtifactManifest:
    """Portable metadata for a saved forecast artifact."""

    model_name: str
    horizon: int
    columns: tuple[str, ...]
    forecast_path: str
    forecast_format: str
    freq: str | None = None
    target_column: str | None = None
    time_column: str | None = None
    panel_columns: tuple[str, ...] = ()
    lower_bound: float | None = None
    upper_bound: float | None = None
    feature_config: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    backtest_metrics: Mapping[str, Any] = field(default_factory=dict)
    interval_metadata: Mapping[str, Any] = field(default_factory=dict)
    ensemble_metadata: Mapping[str, Any] = field(default_factory=dict)
    reconciliation_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if not self.columns:
            raise ValueError("columns must contain at least one forecast column")
        if not self.forecast_path:
            raise ValueError("forecast_path must be non-empty")
        if self.forecast_format not in {"csv", "parquet"}:
            raise ValueError("forecast_format must be 'csv' or 'parquet'")
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "panel_columns", tuple(self.panel_columns))
        object.__setattr__(self, "feature_config", dict(self.feature_config))
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "backtest_metrics", dict(self.backtest_metrics))
        object.__setattr__(self, "interval_metadata", dict(self.interval_metadata))
        object.__setattr__(self, "ensemble_metadata", dict(self.ensemble_metadata))
        object.__setattr__(self, "reconciliation_metadata", dict(self.reconciliation_metadata))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ForecastArtifactManifest:
        return cls(**dict(values))


@dataclass
class ForecastArtifact:
    """Forecast rows plus their portable manifest."""

    forecast: Sequence[Mapping[str, Any]]
    manifest: ForecastArtifactManifest

    def __post_init__(self) -> None:
        self.forecast = [dict(row) for row in self.forecast]
        _validate_columns(self.forecast, self.manifest.columns)

    def save(self, directory: str | Path, *, forecast_format: str | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        selected_format = forecast_format or self.manifest.forecast_format
        if selected_format == "auto":
            selected_format = "parquet" if _has_pyarrow() else "csv"
        if selected_format not in {"csv", "parquet"}:
            raise ValueError("forecast_format must be 'csv', 'parquet', or 'auto'")
        forecast_path = directory / f"forecast.{selected_format}"
        if selected_format == "csv":
            _write_csv(forecast_path, self.forecast, self.manifest.columns)
        else:
            _write_parquet(forecast_path, self.forecast, self.manifest.columns)
        manifest = ForecastArtifactManifest(
            **{
                **self.manifest.to_dict(),
                "forecast_path": forecast_path.name,
                "forecast_format": selected_format,
            }
        )
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
        self.manifest = manifest
        return manifest_path

    @classmethod
    def load(cls, directory: str | Path) -> ForecastArtifact:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest = ForecastArtifactManifest.from_dict(json.loads(manifest_path.read_text()))
        forecast_path = directory / manifest.forecast_path
        if manifest.forecast_format == "csv":
            rows = _read_csv(forecast_path)
        elif manifest.forecast_format == "parquet":
            rows = _read_parquet(forecast_path)
        else:
            raise ValueError(f"unsupported forecast format: {manifest.forecast_format}")
        return cls(rows, manifest)


def build_manifest(
    *,
    model_name: str,
    horizon: int,
    columns: Sequence[str],
    forecast_format: str = "csv",
    forecast_path: str | None = None,
    **metadata: Any,
) -> ForecastArtifactManifest:
    """Convenience helper for constructing a manifest with a matching file name."""

    if forecast_format == "auto":
        forecast_format = "parquet" if _has_pyarrow() else "csv"
    return ForecastArtifactManifest(
        model_name=model_name,
        horizon=horizon,
        columns=tuple(columns),
        forecast_path=forecast_path or f"forecast.{forecast_format}",
        forecast_format=forecast_format,
        **metadata,
    )


def _validate_columns(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    required = set(columns)
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"forecast row {index} is missing required column(s): {names}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [_coerce_csv_row(row) for row in csv.DictReader(handle)]


def _coerce_csv_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {key: _coerce_scalar(value) for key, value in row.items()}


def _coerce_scalar(value: str) -> Any:
    if value == "":
        return None
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Parquet forecast artifacts require the optional 'pyarrow' package. "
            "Install pyarrow or save with forecast_format='csv'."
        ) from exc
    table = pa.Table.from_pylist([{column: row.get(column) for column in columns} for row in rows])
    pq.write_table(table, path)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Reading Parquet forecast artifacts requires the optional 'pyarrow' package."
        ) from exc
    return pq.read_table(path).to_pylist()


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True
