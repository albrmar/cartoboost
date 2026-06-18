use cartoboost_core::forecasting::{
    ForecastActual, ForecastFrame, ForecastFrequency, ForecastRow, Forecaster, NaiveForecaster,
    SeasonalNaiveForecaster,
};
use chrono::{NaiveDate, NaiveDateTime};

fn ts(day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2026, 1, day)
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .expect("valid fixture timestamp")
}

#[test]
fn validates_and_sorts_panel_frame() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("B", ts(2), 20.0),
            ForecastRow::new("A", ts(1), 1.0),
            ForecastRow::new("B", ts(1), 10.0),
            ForecastRow::new("A", ts(2), 2.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid panel frame");

    let keys = frame
        .rows()
        .iter()
        .map(|row| (row.series_id.as_str(), row.timestamp))
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        vec![("A", ts(1)), ("A", ts(2)), ("B", ts(1)), ("B", ts(2))]
    );
}

#[test]
fn rejects_duplicate_panel_timestamp() {
    let err = ForecastFrame::new(
        vec![
            ForecastRow::new("A", ts(1), 1.0),
            ForecastRow::new("A", ts(1), 2.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect_err("duplicate should be rejected");
    assert!(err.to_string().contains("duplicate forecast timestamp"));
}

#[test]
fn rejects_irregular_frequency() {
    let err = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 1.0),
            ForecastRow::single(ts(3), 3.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect_err("gap should be rejected");
    assert!(err.to_string().contains("irregular forecast frequency"));
}

#[test]
fn rejects_non_finite_targets() {
    let err = ForecastFrame::new(
        vec![ForecastRow::single(ts(1), f64::NAN)],
        ForecastFrequency::Daily,
    )
    .expect_err("nan should be rejected");
    assert!(err.to_string().contains("forecast targets must be finite"));
}

#[test]
fn naive_forecasts_each_panel_series_without_bleeding() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PU1->DO2", ts(1), 11.0),
            ForecastRow::new("PU1->DO2", ts(2), 12.0),
            ForecastRow::new("PU9->DO8", ts(1), 31.0),
            ForecastRow::new("PU9->DO8", ts(2), 32.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid panel frame");
    let mut model = NaiveForecaster::new();
    model.fit(&frame).expect("fit");
    let forecast = model.predict(2).expect("predict");

    let means = forecast
        .predictions()
        .iter()
        .map(|row| (row.series_id.as_str(), row.horizon, row.mean))
        .collect::<Vec<_>>();
    assert_eq!(
        means,
        vec![
            ("PU1->DO2", 1, 12.0),
            ("PU1->DO2", 2, 12.0),
            ("PU9->DO8", 1, 32.0),
            ("PU9->DO8", 2, 32.0)
        ]
    );
}

#[test]
fn seasonal_naive_repeats_last_season() {
    let frame = ForecastFrame::new(
        (1..=14)
            .map(|day| ForecastRow::single(ts(day), f64::from(day % 7)))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut model = SeasonalNaiveForecaster::new(7).expect("valid season");
    model.fit(&frame).expect("fit");
    let forecast = model.predict(3).expect("predict");

    let means = forecast
        .predictions()
        .iter()
        .map(|row| row.mean)
        .collect::<Vec<_>>();
    assert_eq!(means, vec![1.0, 2.0, 3.0]);
}

#[test]
fn seasonal_naive_rejects_insufficient_history() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 1.0),
            ForecastRow::single(ts(2), 2.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut model = SeasonalNaiveForecaster::new(7).expect("valid season");
    let err = model.fit(&frame).expect_err("insufficient history");
    assert!(err.to_string().contains("requires at least 7"));
}

#[test]
fn forecast_result_json_round_trips() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 1.0),
            ForecastRow::single(ts(2), 2.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut model = NaiveForecaster::new();
    model.fit(&frame).expect("fit");
    let forecast = model.predict(1).expect("predict");
    let json = forecast.to_json_string().expect("json");
    let restored = cartoboost_core::forecasting::ForecastResult::from_json_string(&json)
        .expect("json round trip");
    assert_eq!(forecast, restored);
}

#[test]
fn metrics_align_by_series_timestamp_and_horizon() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("A", ts(1), 10.0),
            ForecastRow::new("A", ts(2), 12.0),
            ForecastRow::new("B", ts(1), 20.0),
            ForecastRow::new("B", ts(2), 22.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid frame");
    let mut model = NaiveForecaster::new();
    model.fit(&frame).expect("fit");
    let forecast = model.predict(1).expect("predict");

    let metrics = cartoboost_core::forecasting::evaluate_forecast(
        &forecast,
        &[
            ForecastActual {
                series_id: "B".to_string(),
                timestamp: ts(3),
                horizon: 1,
                actual: 23.0,
            },
            ForecastActual {
                series_id: "A".to_string(),
                timestamp: ts(3),
                horizon: 1,
                actual: 13.0,
            },
        ],
    )
    .expect("metrics");
    assert_eq!(metrics.mae, 1.0);
    assert_eq!(metrics.bias, -1.0);
}
