from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from cartoboost.forecasting.lag_features import (  # noqa: E402
    CalendarFeatureConfig,
    LagFeatureBuilder,
    LagFeatureConfig,
    RollingFeatureConfig,
)


def test_lag_features_are_panel_isolated_and_leakage_safe() -> None:
    frame = pd.DataFrame(
        {
            "pickup_zone": ["A", "A", "A", "B", "B", "B"],
            "pickup_hour": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                ]
            ),
            "trips": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "trip_distance": [1.0, 1.5, 1.2, 2.0, 2.2, 2.4],
            "zone_capacity": [5.0, 5.0, 5.0, 9.0, 9.0, 9.0],
        }
    )
    builder = LagFeatureBuilder(
        time_col="pickup_hour",
        target_col="trips",
        panel_cols=["pickup_zone"],
        lag_config=LagFeatureConfig(lags=[1, 2]),
        rolling_config=RollingFeatureConfig(windows=[2], aggregations=["mean", "max"]),
        calendar_config=CalendarFeatureConfig(features=["hour", "dayofweek"]),
        static_cols=["zone_capacity"],
        known_future_cols=["trip_distance"],
    )

    features = builder.fit_transform(frame, drop_missing=False)
    row = features[(features["pickup_zone"] == "A") & (features["pickup_hour"].dt.hour == 2)].iloc[
        0
    ]
    first_b = features[
        (features["pickup_zone"] == "B") & (features["pickup_hour"].dt.hour == 0)
    ].iloc[0]

    assert row["trips_lag_1"] == pytest.approx(20.0)
    assert row["trips_lag_2"] == pytest.approx(10.0)
    assert row["trips_roll_2_mean"] == pytest.approx(15.0)
    assert row["trips_roll_2_max"] == pytest.approx(20.0)
    assert pd.isna(first_b["trips_lag_1"])
    assert builder.feature_names == [
        "trips_lag_1",
        "trips_lag_2",
        "trips_roll_2_mean",
        "trips_roll_2_max",
        "pickup_hour_hour",
        "pickup_hour_dayofweek",
        "zone_capacity",
        "trip_distance",
    ]


def test_future_row_uses_only_history_before_timestamp_and_holiday_hook() -> None:
    history = pd.DataFrame(
        {
            "pickup_zone": ["A", "A"],
            "pickup_hour": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
            "trips": [10.0, 20.0],
            "zone_capacity": [5.0, 5.0],
        }
    )
    future = pd.Series(
        {
            "pickup_zone": "A",
            "pickup_hour": pd.Timestamp("2026-01-01 02:00"),
            "zone_capacity": 5.0,
        }
    )
    builder = LagFeatureBuilder(
        time_col="pickup_hour",
        target_col="trips",
        panel_cols=["pickup_zone"],
        lag_config=LagFeatureConfig(lags=[1]),
        rolling_config=RollingFeatureConfig(
            windows=[2],
            aggregations=["mean"],
            include_expanding=True,
            expanding_aggregations=["mean"],
        ),
        calendar_config=CalendarFeatureConfig(features=["hour", "is_holiday"]),
        static_cols=["zone_capacity"],
        holiday_fn=lambda ts: ts.hour == 2,
    ).fit(history)

    row = builder.transform_future_row(history, future)

    assert row["trips_lag_1"] == pytest.approx(20.0)
    assert row["trips_roll_2_mean"] == pytest.approx(15.0)
    assert row["trips_expand_mean"] == pytest.approx(15.0)
    assert row["pickup_hour_hour"] == pytest.approx(2.0)
    assert row["pickup_hour_is_holiday"] == pytest.approx(1.0)


def test_delta_and_trend_features_are_shared_and_leakage_safe() -> None:
    frame = pd.DataFrame(
        {
            "pickup_hour": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "trips": [10.0, 12.0, 15.0, 19.0],
        }
    )
    builder = LagFeatureBuilder(
        time_col="pickup_hour",
        target_col="trips",
        lag_config=LagFeatureConfig(lags=[1], difference_lags=[2], rolling_trend_windows=[3]),
    )

    features = builder.fit_transform(frame, drop_missing=False)
    row = features[features["pickup_hour"] == pd.Timestamp("2026-01-04")].iloc[0]
    future = builder.transform_future_row(
        frame,
        {"pickup_hour": pd.Timestamp("2026-01-05")},
    )

    assert builder.feature_names == [
        "trips_lag_1",
        "trips_delta_lag_2",
        "trips_roll_trend_3",
    ]
    assert row["trips_lag_1"] == pytest.approx(15.0)
    assert row["trips_delta_lag_2"] == pytest.approx(5.0)
    assert row["trips_roll_trend_3"] == pytest.approx(2.5)
    assert future["trips_delta_lag_2"] == pytest.approx(7.0)
    assert future["trips_roll_trend_3"] == pytest.approx(3.5)


def test_history_features_exclude_same_timestamp_targets() -> None:
    frame = pd.DataFrame(
        {
            "pickup_hour": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                ]
            ),
            "trips": [10.0, 20.0, 999.0, 30.0],
        }
    )
    builder = LagFeatureBuilder(
        time_col="pickup_hour",
        target_col="trips",
        lag_config=LagFeatureConfig(lags=[1]),
        rolling_config=RollingFeatureConfig(windows=[2], aggregations=["mean"], min_periods=1),
    )

    features = builder.fit_transform(frame, drop_missing=False)
    duplicate_time = features[features["pickup_hour"] == pd.Timestamp("2026-01-01 01:00")]

    assert duplicate_time["trips_lag_1"].tolist() == pytest.approx([10.0, 10.0])
    assert duplicate_time["trips_roll_2_mean"].tolist() == pytest.approx([10.0, 10.0])
    assert "__cartoboost_single_panel__" not in features.columns
