use cartoboost_core::forecasting::{
    DirectFeatureMatrix, ForecastDiagnostics, ForecastFeatureFactory, ForecastFrame,
    ForecastFrequency, ForecastRow, LagFeatureConfig,
};
use chrono::{NaiveDate, NaiveDateTime};

fn ts(day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2026, 1, day)
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .expect("valid timestamp")
}

fn taxi_panel_frame() -> ForecastFrame {
    ForecastFrame::new(
        vec![
            ForecastRow::new("PU1_DO2", ts(1), 0.0),
            ForecastRow::new("PU1_DO2", ts(2), 10.0),
            ForecastRow::new("PU1_DO2", ts(3), 0.0),
            ForecastRow::new("PU1_DO2", ts(4), 20.0),
            ForecastRow::new("PU1_DO2", ts(5), 0.0),
            ForecastRow::new("PU9_DO3", ts(1), 100.0),
            ForecastRow::new("PU9_DO3", ts(2), 110.0),
            ForecastRow::new("PU9_DO3", ts(3), 120.0),
            ForecastRow::new("PU9_DO3", ts(4), 130.0),
            ForecastRow::new("PU9_DO3", ts(5), 140.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid panel")
}

fn direct_factory(max_horizon: usize) -> ForecastFeatureFactory {
    ForecastFeatureFactory::new(
        LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        },
        max_horizon,
    )
    .expect("factory")
}

#[test]
fn diagnostics_report_zero_fraction_and_intermittency_by_series() {
    let frame = taxi_panel_frame();

    let diagnostics = ForecastDiagnostics::from_frame(&frame);
    let sparse = diagnostics.series("PU1_DO2").expect("sparse series");
    let dense = diagnostics.series("PU9_DO3").expect("dense series");

    assert_eq!(diagnostics.n_rows, 10);
    assert_eq!(diagnostics.n_series, 2);
    assert_eq!(diagnostics.zero_count, 3);
    assert_eq!(diagnostics.zero_fraction, 0.3);
    assert_eq!(diagnostics.intermittent_series_count, 1);
    assert_eq!(sparse.zero_count, 3);
    assert_eq!(sparse.nonzero_count, 2);
    assert_eq!(sparse.zero_fraction, 0.6);
    assert_eq!(sparse.intermittency_ratio, Some(1.5));
    assert_eq!(sparse.mean_nonzero_interval, Some(2.0));
    assert_eq!(sparse.max_zero_run, 1);
    assert!(sparse.is_intermittent);
    assert_eq!(dense.zero_fraction, 0.0);
    assert!(!dense.is_intermittent);
}

#[test]
fn direct_feature_matrix_is_panel_isolated() {
    let frame = taxi_panel_frame();
    let matrix = direct_factory(1)
        .build_direct_matrix(&frame)
        .expect("matrix");
    let pu1_day3 = find_row(&matrix, "PU1_DO2", ts(3), 1);
    let pu9_day3 = find_row(&matrix, "PU9_DO3", ts(3), 1);

    assert_eq!(
        matrix.feature_names,
        vec![
            "target_lag_1",
            "target_lag_2",
            "target_roll_mean_2",
            "horizon"
        ]
    );
    assert_eq!(matrix.features[pu1_day3], vec![10.0, 0.0, 5.0, 1.0]);
    assert_eq!(matrix.targets[pu1_day3], 0.0);
    assert_eq!(matrix.features[pu9_day3], vec![110.0, 100.0, 105.0, 1.0]);
    assert_eq!(matrix.targets[pu9_day3], 120.0);
}

#[test]
fn direct_feature_matrix_output_is_deterministic() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU9_DO3", ts(3), 120.0),
            ForecastRow::new("PU1_DO2", ts(2), 10.0),
            ForecastRow::new("PU9_DO3", ts(1), 100.0),
            ForecastRow::new("PU1_DO2", ts(1), 0.0),
            ForecastRow::new("PU9_DO3", ts(2), 110.0),
            ForecastRow::new("PU1_DO2", ts(3), 0.0),
            ForecastRow::new("PU1_DO2", ts(4), 20.0),
            ForecastRow::new("PU9_DO3", ts(4), 130.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid panel");
    let factory = direct_factory(2);

    let first = factory.build_direct_matrix(&frame).expect("first matrix");
    let second = factory.build_direct_matrix(&frame).expect("second matrix");

    assert_eq!(first, second);
    assert_eq!(
        first
            .series_ids
            .iter()
            .zip(first.target_timestamps.iter())
            .map(|(series_id, timestamp)| (series_id.as_str(), *timestamp))
            .collect::<Vec<_>>(),
        vec![
            ("PU1_DO2", ts(3)),
            ("PU1_DO2", ts(4)),
            ("PU1_DO2", ts(4)),
            ("PU9_DO3", ts(3)),
            ("PU9_DO3", ts(4)),
            ("PU9_DO3", ts(4)),
        ]
    );
}

#[test]
fn direct_feature_matrix_does_not_use_future_targets_for_longer_horizons() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1_DO2", ts(1), 10.0),
            ForecastRow::new("PU1_DO2", ts(2), 20.0),
            ForecastRow::new("PU1_DO2", ts(3), 999.0),
            ForecastRow::new("PU1_DO2", ts(4), 40.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let matrix = direct_factory(2)
        .build_direct_matrix(&frame)
        .expect("matrix");
    let day2_to_day4 = find_row(&matrix, "PU1_DO2", ts(4), 2);

    assert_eq!(matrix.origin_timestamps[day2_to_day4], ts(2));
    assert_eq!(matrix.targets[day2_to_day4], 40.0);
    assert_eq!(matrix.features[day2_to_day4], vec![20.0, 10.0, 15.0, 2.0]);
    assert!(!matrix.features[day2_to_day4].contains(&999.0));
}

fn find_row(
    matrix: &DirectFeatureMatrix,
    series_id: &str,
    target_timestamp: NaiveDateTime,
    horizon: usize,
) -> usize {
    matrix
        .series_ids
        .iter()
        .zip(matrix.target_timestamps.iter())
        .zip(matrix.horizons.iter())
        .position(|((row_series_id, row_timestamp), row_horizon)| {
            row_series_id == series_id
                && *row_timestamp == target_timestamp
                && *row_horizon == horizon
        })
        .expect("matching matrix row")
}
