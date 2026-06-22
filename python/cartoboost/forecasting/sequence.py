from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SequenceRow:
    row_id: str
    position: float
    target: float | None = None
    reference_axis: float | None = None
    reference_signal: float | None = None
    auxiliary_rate: float | None = None


@dataclass(frozen=True)
class SequenceSeries:
    series_id: str
    rows: list[SequenceRow]


@dataclass(frozen=True)
class ReferenceSignal:
    axis: list[float]
    signal: list[float]


@dataclass(frozen=True)
class SequenceStateSpaceConfig:
    initial_axis_variance: float = 1.0
    initial_rate_variance: float = 1.0
    axis_process_variance: float = 1e-3
    rate_process_variance: float = 1e-3
    signal_observation_variance: float = 1e-2
    rate_observation_variance: float | None = None
    sigma_point_alpha: float = 1e-3
    sigma_point_beta: float = 2.0
    sigma_point_kappa: float = 0.0


@dataclass(frozen=True)
class ReferencePathConfig:
    emission_scale: float = 1.0
    student_t_df: float = 4.0
    transition_penalty: float = 0.01
    smoothness_penalty: float = 0.0
    start_axis: float | None = None
    start_penalty: float = 0.0


def validate_sequence_frame(series: list[SequenceSeries]) -> dict[str, Any]:
    native = _native()
    return json.loads(native.sequence_validate_value(json.dumps({"series": _convert(series)})))


def forward_ekf(
    series: SequenceSeries,
    reference: ReferenceSignal,
    config: SequenceStateSpaceConfig | None = None,
) -> dict[str, Any]:
    return _state_space(series, reference, config, "ekf")


def ukf_reference(
    series: SequenceSeries,
    reference: ReferenceSignal,
    config: SequenceStateSpaceConfig | None = None,
) -> dict[str, Any]:
    return _state_space(series, reference, config, "ukf")


def rts_smoother(
    series: SequenceSeries,
    reference: ReferenceSignal,
    config: SequenceStateSpaceConfig | None = None,
) -> dict[str, Any]:
    return _state_space(series, reference, config, "rts")


def missing_target_continuation(
    series: SequenceSeries,
    reference: ReferenceSignal,
    config: SequenceStateSpaceConfig | None = None,
) -> dict[str, Any]:
    return _state_space(series, reference, config, "continuation")


def reference_path_viterbi(
    series: SequenceSeries,
    reference: ReferenceSignal,
    config: ReferencePathConfig | None = None,
) -> dict[str, Any]:
    native = _native()
    return json.loads(
        native.sequence_reference_path_viterbi_value(
            json.dumps(_convert(series)),
            json.dumps(_convert(reference)),
            None if config is None else json.dumps(_convert(config)),
        )
    )


def reference_path_posterior_mean(
    series: SequenceSeries,
    reference: ReferenceSignal,
    config: ReferencePathConfig | None = None,
) -> dict[str, Any]:
    native = _native()
    return json.loads(
        native.sequence_reference_path_posterior_mean_value(
            json.dumps(_convert(series)),
            json.dumps(_convert(reference)),
            None if config is None else json.dumps(_convert(config)),
        )
    )


def sequence_blend(
    candidates: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
    actuals: list[dict[str, Any]] | None = None,
    mode: str = "fixed",
) -> dict[str, Any]:
    native = _native()
    return json.loads(
        native.sequence_blend_value(
            json.dumps(candidates),
            None if weights is None else json.dumps(weights),
            None if actuals is None else json.dumps(actuals),
            mode,
        )
    )


def validate_oof_meta_training(rows: list[dict[str, Any]]) -> None:
    _native().sequence_validate_oof_meta_training_value(json.dumps(rows))


def generate_group_oof_candidate_rows(
    *,
    validation_group_id: str,
    train_group_ids: list[str],
    actuals: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return json.loads(
        _native().sequence_generate_group_oof_candidate_rows_value(
            json.dumps(
                {
                    "validation_group_id": validation_group_id,
                    "train_group_ids": train_group_ids,
                    "actuals": actuals,
                    "candidates": candidates,
                }
            )
        )
    )


def per_group_error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return json.loads(_native().sequence_group_error_summary_value(json.dumps(rows)))


def _state_space(
    series: SequenceSeries,
    reference: ReferenceSignal,
    config: SequenceStateSpaceConfig | None,
    method: str,
) -> dict[str, Any]:
    native = _native()
    return json.loads(
        native.sequence_state_space_value(
            json.dumps(_convert(series)),
            json.dumps(_convert(reference)),
            None if config is None else json.dumps(_convert(config)),
            method,
        )
    )


def _convert(value: Any) -> Any:
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _convert(asdict(value))
    if isinstance(value, dict):
        return {key: _convert(item) for key, item in value.items()}
    return value


def _native() -> Any:
    try:
        from cartoboost import _native as native
    except ImportError as exc:
        raise NotImplementedError("cartoboost._native is required for sequence utilities") from exc
    return native
