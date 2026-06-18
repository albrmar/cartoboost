from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .registry import ForecastRegistry

MODEL_NAMES = (
    "naive",
    "seasonal_naive",
    "theta",
    "optimized_theta",
    "ets",
    "auto_arima",
    "cartoboost_lag",
    "weighted_ensemble",
)


class ForecastCliError(ValueError):
    """User-facing CLI error."""


@dataclass(frozen=True)
class ForecastConfig:
    input: Path | None
    timestamp_col: str
    target_col: str
    series_id_col: str | None
    freq: str
    model: str
    horizon: int
    season_length: int
    output: Path | None
    artifact_dir: Path | None

    def to_json(self) -> dict[str, Any]:
        return {
            "input": None if self.input is None else str(self.input),
            "timestamp_col": self.timestamp_col,
            "target_col": self.target_col,
            "series_id_col": self.series_id_col,
            "freq": self.freq,
            "model": self.model,
            "horizon": self.horizon,
            "season_length": self.season_length,
            "output": None if self.output is None else str(self.output),
            "artifact_dir": None if self.artifact_dir is None else str(self.artifact_dir),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _resolve_config(args)
        if args.command == "fit":
            _fit(config)
        elif args.command == "predict":
            _predict(config)
        elif args.command == "backtest":
            _backtest(config)
        elif args.command == "compare":
            _compare(config)
        else:
            raise ForecastCliError(f"unknown command {args.command!r}")
    except ForecastCliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cartoboost forecast",
        description=(
            f"Forecast taxi pickup demand or trip metrics. Models: {', '.join(MODEL_NAMES)}."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "predict", "backtest", "compare"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--input")
        sub.add_argument("--timestamp-col", default=None)
        sub.add_argument("--target-col", default=None)
        sub.add_argument("--series-id-col", default=None)
        sub.add_argument("--freq", default=None)
        sub.add_argument("--model", default=None, help=f"one of: {', '.join(MODEL_NAMES)}")
        sub.add_argument("--horizon", type=int, default=None)
        sub.add_argument("--season-length", type=int, default=None)
        sub.add_argument("--output")
        sub.add_argument("--artifact-dir")
        sub.add_argument("--config")
    return parser


def _resolve_config(args: argparse.Namespace) -> ForecastConfig:
    file_config = _read_config(Path(args.config)) if args.config else {}

    def option(name: str, default: Any = None) -> Any:
        value = getattr(args, name, None)
        if value is not None:
            return value
        return file_config.get(name.replace("_", "-"), file_config.get(name, default))

    model = str(option("model", "theta")).strip().lower()
    if model not in MODEL_NAMES and model != "all":
        raise ForecastCliError(f"unknown model {model!r}; expected one of {', '.join(MODEL_NAMES)}")
    if model == "all" and args.command != "compare":
        raise ForecastCliError("--model all is only valid for compare")
    horizon = _positive_int(option("horizon", 7), "horizon")
    season_length = _positive_int(option("season_length", 7), "season_length")
    freq = str(option("freq", "D")).strip()
    if freq not in {"D", "H", "W", "M"}:
        raise ForecastCliError("freq must be one of D, H, W, or M")
    timestamp_col = str(option("timestamp_col", option("time_column", "timestamp"))).strip()
    target_col = str(option("target_col", option("target_column", "target"))).strip()
    if not timestamp_col:
        raise ForecastCliError("timestamp-col must not be empty")
    if not target_col:
        raise ForecastCliError("target-col must not be empty")
    input_path = option("input")
    output_path = option("output")
    artifact_dir = option("artifact_dir")
    return ForecastConfig(
        input=None if input_path is None else Path(input_path),
        timestamp_col=timestamp_col,
        target_col=target_col,
        series_id_col=option("series_id_col", _first_panel_column(option("panel_columns"))),
        freq=freq,
        model=model,
        horizon=horizon,
        season_length=season_length,
        output=None if output_path is None else Path(output_path),
        artifact_dir=None if artifact_dir is None else Path(artifact_dir),
    )


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ForecastCliError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = _read_simple_toml(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ForecastCliError(f"invalid config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ForecastCliError("config file must contain an object/table")
    valid = {
        "input",
        "timestamp_col",
        "timestamp-col",
        "target_col",
        "target-col",
        "target_column",
        "target-column",
        "series_id_col",
        "series-id-col",
        "time_column",
        "time-column",
        "panel_columns",
        "panel-columns",
        "freq",
        "model",
        "horizon",
        "season_length",
        "season-length",
        "output",
        "artifact_dir",
        "artifact-dir",
    }
    for key in data:
        if key not in valid:
            raise ForecastCliError(f"unknown config key {key!r}")
    return data


def _read_simple_toml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"invalid config line {line_number}")
        key, value = [part.strip() for part in line.split("=", 1)]
        if not key:
            raise ValueError(f"invalid config line {line_number}")
        data[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"invalid unquoted config value {value!r}") from exc


def _first_panel_column(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        values = list(value)
    except TypeError as exc:
        raise ForecastCliError("panel_columns must be a string or list of strings") from exc
    if len(values) > 1:
        raise ForecastCliError("CLI forecasting supports one panel column; use --series-id-col")
    return None if not values else str(values[0])


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ForecastCliError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ForecastCliError(f"{name} must be a positive integer")
    return parsed


def _fit(config: ForecastConfig) -> None:
    rows = _read_series(config)
    model = _new_model(config)
    model.fit(_target_payload(rows))
    artifact_dir = _require_artifact_dir(config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_dir / "resolved_config.json", config.to_json())
    _write_json(
        artifact_dir / "model.json",
        {
            "model": config.model,
            "native_class": getattr(model, "native_class_name", None),
            "note": "Native forecasting artifact serialization is delegated to Rust bindings.",
        },
    )


def _predict(config: ForecastConfig) -> None:
    _require_artifact_dir(config)
    _require_output(config)
    raise NotImplementedError(
        "Rust binding for forecasting artifact loading/prediction is not available."
    )


def _backtest(config: ForecastConfig) -> None:
    _read_series(config)
    _require_output(config)
    raise NotImplementedError("Rust binding for forecasting backtests is not available.")


def _compare(config: ForecastConfig) -> None:
    _read_series(config)
    _require_output(config)
    _split_models(config.model if config.model != "all" else ",".join(MODEL_NAMES))
    raise NotImplementedError("Rust binding for forecasting model comparison is not available.")


def _split_models(value: str) -> list[str]:
    models = [part.strip().lower() for part in value.split(",") if part.strip()]
    for model in models:
        if model not in MODEL_NAMES:
            raise ForecastCliError(
                f"unknown model {model!r}; expected one of {', '.join(MODEL_NAMES)}"
            )
    return models


def _read_series(config: ForecastConfig) -> dict[str, list[dict[str, Any]]]:
    if config.input is None:
        raise ForecastCliError("--input is required")
    if not config.input.exists():
        raise ForecastCliError(f"input file not found: {config.input}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with config.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ForecastCliError("input CSV is missing a header")
        for column in (config.timestamp_col, config.target_col):
            if column not in reader.fieldnames:
                raise ForecastCliError(f"required column {column!r} not found in input CSV")
        if config.series_id_col is not None and config.series_id_col not in reader.fieldnames:
            raise ForecastCliError(
                f"series id column {config.series_id_col!r} not found in input CSV"
            )
        for row_number, row in enumerate(reader, start=2):
            timestamp = _parse_datetime(row.get(config.timestamp_col, ""), row_number)
            target = _parse_float(row.get(config.target_col, ""), row_number, config.target_col)
            series_id = row[config.series_id_col] if config.series_id_col else "series"
            grouped[series_id].append(
                {"timestamp": timestamp, "target": target, "series_id": series_id}
            )
    for series_id, values in grouped.items():
        if len(values) < 2:
            raise ForecastCliError(f"series {series_id!r} must contain at least 2 rows")
        values.sort(key=lambda item: item["timestamp"])
    if not grouped:
        raise ForecastCliError("input CSV contains no data rows")
    return dict(grouped)


def _parse_datetime(value: str | None, row_number: int) -> datetime:
    if value is None or not value.strip():
        raise ForecastCliError(f"row {row_number}: timestamp is empty")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ForecastCliError(f"row {row_number}: invalid timestamp {value!r}") from exc


def _parse_float(value: str | None, row_number: int, column: str) -> float:
    try:
        parsed = float("" if value is None else value)
    except ValueError as exc:
        raise ForecastCliError(f"row {row_number}: invalid numeric value for {column!r}") from exc
    if not math.isfinite(parsed):
        raise ForecastCliError(f"row {row_number}: {column!r} must be finite")
    return parsed


def _new_model(config: ForecastConfig) -> Any:
    if config.model == "seasonal_naive":
        return ForecastRegistry.defaults().create(config.model, season_length=config.season_length)
    if config.model == "weighted_ensemble":
        raise NotImplementedError(
            "Rust binding for WeightedEnsembleForecaster is not available through the CLI."
        )
    return ForecastRegistry.defaults().create(config.model)


def _target_payload(rows: dict[str, list[dict[str, Any]]]) -> list[float] | dict[str, list[float]]:
    payload = {
        series_id: [float(item["target"]) for item in values] for series_id, values in rows.items()
    }
    if set(payload) == {"series"}:
        return payload["series"]
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_artifact_dir(config: ForecastConfig) -> Path:
    if config.artifact_dir is None:
        raise ForecastCliError("--artifact-dir is required")
    return config.artifact_dir


def _require_output(config: ForecastConfig) -> Path:
    if config.output is None:
        raise ForecastCliError("--output is required")
    return config.output


if __name__ == "__main__":
    raise SystemExit(main())
