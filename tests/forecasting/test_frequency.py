import pytest

pd = pytest.importorskip("pandas")

from cartoboost.forecasting.frequency import (  # noqa: E402
    infer_frequency,
    next_timestamps,
    normalize_frequency,
    validate_horizon,
    validate_regular_frequency,
)


def test_normalize_frequency_uses_pandas_canonical_form():
    assert normalize_frequency("1D") == "D"
    assert normalize_frequency("2h") == "2h"


def test_infer_frequency_accepts_unsorted_regular_timestamps():
    timestamps = pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02"])

    assert infer_frequency(timestamps) == "D"


def test_validate_regular_frequency_rejects_irregular_timestamps():
    timestamps = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-04"])

    with pytest.raises(ValueError, match="irregular"):
        validate_regular_frequency(timestamps, "D", label="taxi zone 12")


@pytest.mark.parametrize("horizon", [0, -1, True, 2.5, "3"])
def test_validate_horizon_rejects_invalid_values(horizon):
    with pytest.raises(ValueError, match="positive integer"):
        validate_horizon(horizon)


def test_validate_horizon_accepts_positive_integer():
    assert validate_horizon(3) == 3


def test_next_timestamps_starts_after_last_observation():
    result = next_timestamps(pd.Timestamp("2025-01-31"), 2, "MS")

    assert result == [pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-01")]
