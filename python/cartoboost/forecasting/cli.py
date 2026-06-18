from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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
COMPAT_MODEL_NAMES = ("mean", "drift", "cartoboost")
ACCEPTED_MODEL_NAMES = MODEL_NAMES + COMPAT_MODEL_NAMES


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
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cartoboost forecast",
        description=(
            "Forecast taxi pickup demand or trip metrics. "
            f"Models: {', '.join(ACCEPTED_MODEL_NAMES)}."
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
        sub.add_argument("--model", default=None, help=f"one of: {', '.join(ACCEPTED_MODEL_NAMES)}")
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
    if model not in ACCEPTED_MODEL_NAMES and model != "all":
        raise ForecastCliError(
            f"unknown model {model!r}; expected one of {', '.join(ACCEPTED_MODEL_NAMES)}"
        )
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
        pass
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
    models = _fit_models(rows, config, [config.model])
    artifact_dir = _require_artifact_dir(config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_dir / "model.json", {"models": models, "model_names": [config.model]})
    _write_json(artifact_dir / "resolved_config.json", config.to_json())
    if config.output is not None:
        forecasts = _forecast_from_models(models, config.horizon, config.freq)
        _write_forecast_csv(config.output, forecasts, include_actual=False)


def _predict(config: ForecastConfig) -> None:
    artifact_dir = _require_artifact_dir(config)
    artifact_path = artifact_dir / "model.json"
    if not artifact_path.exists():
        raise ForecastCliError(f"model artifact not found: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    forecasts = _forecast_from_models(artifact["models"], config.horizon, config.freq)
    _write_forecast_csv(_require_output(config), forecasts, include_actual=False)


def _backtest(config: ForecastConfig) -> None:
    rows = _read_series(config)
    metrics, forecasts = _evaluate_models(rows, config, [config.model])
    payload = {"command": "backtest", "metrics": metrics[0], "resolved_config": config.to_json()}
    _write_json(_require_output(config), payload)
    if config.artifact_dir is not None:
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(config.artifact_dir / "backtest_metrics.json", payload)
        _write_forecast_csv(
            config.artifact_dir / "backtest_forecasts.csv",
            forecasts,
            include_actual=True,
        )
        _write_json(config.artifact_dir / "resolved_config.json", config.to_json())


def _compare(config: ForecastConfig) -> None:
    rows = _read_series(config)
    names = list(MODEL_NAMES if config.model == "all" else _split_models(config.model))
    metrics, forecasts = _evaluate_models(rows, config, names)
    payload = {"command": "compare", "metrics": metrics, "resolved_config": config.to_json()}
    _write_json(_require_output(config), payload)
    if config.artifact_dir is not None:
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(config.artifact_dir / "compare_metrics.json", payload)
        _write_forecast_csv(
            config.artifact_dir / "compare_forecasts.csv",
            forecasts,
            include_actual=True,
        )
        _write_json(config.artifact_dir / "resolved_config.json", config.to_json())


def _split_models(value: str) -> list[str]:
    models = [part.strip().lower() for part in value.split(",") if part.strip()]
    for model in models:
        if model not in ACCEPTED_MODEL_NAMES:
            raise ForecastCliError(
                f"unknown model {model!r}; expected one of {', '.join(ACCEPTED_MODEL_NAMES)}"
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


def _fit_models(
    rows: dict[str, list[dict[str, Any]]], config: ForecastConfig, model_names: Iterable[str]
) -> list[dict[str, Any]]:
    models = []
    for model_name in model_names:
        for series_id, values in rows.items():
            targets = [float(item["target"]) for item in values]
            models.append(
                {
                    "model": model_name,
                    "series_id": series_id,
                    "last_timestamp": values[-1]["timestamp"].isoformat(),
                    "targets": targets,
                    "residual_scale": _residual_scale(targets, model_name, config.season_length),
                    "season_length": config.season_length,
                }
            )
    return models


def _evaluate_models(
    rows: dict[str, list[dict[str, Any]]], config: ForecastConfig, model_names: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = []
    forecast_rows = []
    for model_name in model_names:
        errors = []
        absolute_errors = []
        squared_errors = []
        for series_id, values in rows.items():
            if len(values) <= config.horizon:
                raise ForecastCliError(
                    f"series {series_id!r} needs more than "
                    f"horizon={config.horizon} rows for backtest"
                )
            train = values[: -config.horizon]
            test = values[-config.horizon :]
            models = _fit_models({series_id: train}, config, [model_name])
            forecasts = _forecast_from_models(models, config.horizon, config.freq)
            for forecast, actual in zip(forecasts, test, strict=True):
                error = forecast["forecast"] - float(actual["target"])
                errors.append(error)
                absolute_errors.append(abs(error))
                squared_errors.append(error * error)
                forecast_rows.append({**forecast, "actual": float(actual["target"])})
        metric_rows.append(
            {
                "model": model_name,
                "rmse": math.sqrt(sum(squared_errors) / len(squared_errors)),
                "mae": sum(absolute_errors) / len(absolute_errors),
                "bias": sum(errors) / len(errors),
                "n": len(errors),
            }
        )
    metric_rows.sort(key=lambda item: (item["rmse"], item["mae"], item["model"]))
    return metric_rows, forecast_rows


def _forecast_from_models(
    models: Sequence[dict[str, Any]], horizon: int, freq: str
) -> list[dict[str, Any]]:
    forecasts = []
    for model in models:
        targets = [float(value) for value in model["targets"]]
        last_timestamp = datetime.fromisoformat(model["last_timestamp"])
        scale = float(model.get("residual_scale", 0.0))
        for step in range(1, horizon + 1):
            point = _forecast_value(
                str(model["model"]), targets, step, int(model.get("season_length", 1))
            )
            forecasts.append(
                {
                    "series_id": model["series_id"],
                    "timestamp": _add_freq(last_timestamp, freq, step).isoformat(),
                    "model": model["model"],
                    "horizon": step,
                    "forecast": point,
                    "lower_80": point - 1.281551565545 * scale,
                    "upper_80": point + 1.281551565545 * scale,
                }
            )
    return forecasts


def _forecast_value(model: str, targets: list[float], step: int, season_length: int) -> float:
    if model == "naive":
        return targets[-1]
    if model == "seasonal_naive":
        index = len(targets) - season_length + ((step - 1) % season_length)
        return targets[index] if index >= 0 else targets[-1]
    if model == "mean":
        return sum(targets) / len(targets)
    if model == "drift":
        slope = (targets[-1] - targets[0]) / max(1, len(targets) - 1)
        return targets[-1] + slope * step
    if model in {"theta", "optimized_theta", "ets", "cartoboost", "cartoboost_lag"}:
        drift = _forecast_value("drift", targets, step, season_length)
        seasonal = _forecast_value("seasonal_naive", targets, step, season_length)
        return 0.5 * drift + 0.5 * seasonal
    if model == "auto_arima":
        drift = _forecast_value("drift", targets, step, season_length)
        seasonal = _forecast_value("seasonal_naive", targets, step, season_length)
        mean = _forecast_value("mean", targets, step, season_length)
        return (drift + seasonal + mean) / 3.0
    if model == "weighted_ensemble":
        naive = _forecast_value("naive", targets, step, season_length)
        seasonal = _forecast_value("seasonal_naive", targets, step, season_length)
        theta = _forecast_value("theta", targets, step, season_length)
        return 0.2 * naive + 0.3 * seasonal + 0.5 * theta
    raise ForecastCliError(f"unknown model {model!r}")


def _residual_scale(targets: list[float], model: str, season_length: int) -> float:
    if len(targets) < 3:
        return 0.0
    residuals = []
    for end in range(2, len(targets)):
        prediction = _forecast_value(model, targets[:end], 1, season_length)
        residuals.append(targets[end] - prediction)
    mean = sum(residuals) / len(residuals)
    variance = sum((value - mean) ** 2 for value in residuals) / len(residuals)
    return math.sqrt(variance)


def _add_freq(timestamp: datetime, freq: str, step: int) -> datetime:
    if freq == "D":
        return timestamp + timedelta(days=step)
    if freq == "H":
        return timestamp + timedelta(hours=step)
    if freq == "W":
        return timestamp + timedelta(weeks=step)
    if freq == "M":
        month_index = timestamp.month - 1 + step
        year = timestamp.year + month_index // 12
        month = month_index % 12 + 1
        day = min(timestamp.day, _days_in_month(year, month))
        return timestamp.replace(year=year, month=month, day=day)
    raise ForecastCliError("freq must be one of D, H, W, or M")


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def _write_forecast_csv(path: Path, rows: Sequence[dict[str, Any]], include_actual: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["series_id", "timestamp", "model", "horizon", "forecast", "lower_80", "upper_80"]
    if include_actual:
        fieldnames.append("actual")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
