use cartoboost_core::booster::BoosterConfig;
use cartoboost_core::forecasting::{
    CalendarFeature, CartoBoostLagForecaster, ForecastFrame, ForecastFrequency, ForecastRow,
    Forecaster, LagFeatureBuilder, LagFeatureConfig,
};
use chrono::{NaiveDate, NaiveDateTime};

fn ts(day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2026, 1, day)
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .expect("valid fixture timestamp")
}

fn small_booster_config() -> BoosterConfig {
    BoosterConfig {
        n_estimators: 8,
        learning_rate: 0.3,
        max_depth: 2,
        min_samples_leaf: 1,
        min_gain: 0.0,
        ..BoosterConfig::default()
    }
}

#[test]
fn lag_builder_is_leakage_safe_and_panel_isolated() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1", ts(1), 10.0),
            ForecastRow::new("PU1", ts(2), 20.0),
            ForecastRow::new("PU1", ts(3), 30.0),
            ForecastRow::new("PU9", ts(1), 100.0),
            ForecastRow::new("PU9", ts(2), 200.0),
            ForecastRow::new("PU9", ts(3), 300.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let builder = LagFeatureBuilder::new(LagFeatureConfig {
        lags: vec![1, 2],
        rolling_mean_windows: vec![2],
        calendar_features: vec![CalendarFeature::DayOfWeek, CalendarFeature::Month],
    })
    .expect("valid config");

    let rows = builder.transform_frame(&frame).expect("features");
    let pu1_day3 = rows
        .iter()
        .find(|row| row.series_id == "PU1" && row.timestamp == ts(3))
        .expect("PU1 day 3 row");
    let pu9_day3 = rows
        .iter()
        .find(|row| row.series_id == "PU9" && row.timestamp == ts(3))
        .expect("PU9 day 3 row");

    assert_eq!(
        builder.feature_names(),
        &[
            "target_lag_1".to_string(),
            "target_lag_2".to_string(),
            "target_roll_mean_2".to_string(),
            "calendar_day_of_week".to_string(),
            "calendar_month".to_string(),
        ]
    );
    assert_eq!(pu1_day3.features[0], 20.0);
    assert_eq!(pu1_day3.features[1], 10.0);
    assert_eq!(pu1_day3.features[2], 15.0);
    assert_eq!(pu9_day3.features[0], 200.0);
    assert_eq!(pu9_day3.features[1], 100.0);
    assert_eq!(pu9_day3.features[2], 150.0);
    assert_eq!(rows.len(), 2);
}

#[test]
fn lag_builder_drops_rows_without_complete_lag_history() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1", ts(1), 10.0),
            ForecastRow::new("PU1", ts(2), 20.0),
            ForecastRow::new("PU1", ts(3), 30.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let builder = LagFeatureBuilder::new(LagFeatureConfig {
        lags: vec![2],
        rolling_mean_windows: Vec::new(),
        calendar_features: Vec::new(),
    })
    .expect("valid config");

    let rows = builder.transform_frame(&frame).expect("features");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].timestamp, ts(3));
    assert_eq!(rows[0].features, vec![10.0]);
}

#[test]
fn cartoboost_lag_forecaster_predicts_recursively_per_panel() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1", ts(1), 10.0),
            ForecastRow::new("PU1", ts(2), 12.0),
            ForecastRow::new("PU1", ts(3), 14.0),
            ForecastRow::new("PU1", ts(4), 16.0),
            ForecastRow::new("PU9", ts(1), 40.0),
            ForecastRow::new("PU9", ts(2), 42.0),
            ForecastRow::new("PU9", ts(3), 44.0),
            ForecastRow::new("PU9", ts(4), 46.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut forecaster = CartoBoostLagForecaster::new(
        LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: vec![2],
            calendar_features: vec![CalendarFeature::Day],
        },
        small_booster_config(),
    )
    .expect("forecaster");

    forecaster.fit(&frame).expect("fit");
    let forecast = forecaster.predict(2).expect("predict");
    let predictions = forecast.predictions();

    assert_eq!(forecaster.training_rows(), Some(4));
    assert_eq!(predictions.len(), 4);
    assert_eq!(predictions[0].series_id, "PU1");
    assert_eq!(predictions[0].timestamp, ts(5));
    assert_eq!(predictions[0].horizon, 1);
    assert_eq!(predictions[1].series_id, "PU1");
    assert_eq!(predictions[1].timestamp, ts(6));
    assert_eq!(predictions[1].horizon, 2);
    assert_eq!(predictions[2].series_id, "PU9");
    assert_eq!(predictions[2].timestamp, ts(5));
    assert!(predictions
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
    assert!(predictions[2].mean > predictions[0].mean);
}
