use cartoboost_core::forecasting::{
    ExpertScore, ForecastEnsemble, ForecastFrame, ForecastFrequency, ForecastRow, Forecaster,
    NaiveForecaster, RuleBasedGating, SeasonalNaiveForecaster, ValidationScoreTable,
};
use chrono::NaiveDate;

#[test]
fn rule_based_gating_converts_validation_errors_to_weights() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("naive", "rmse", 4.0),
        ExpertScore::global("seasonal", "rmse", 2.0),
    ])
    .expect("score table");
    let gating = RuleBasedGating::new("rmse", table).expect("gating");

    let weights = gating.weights_for(None, None).expect("weights");

    assert_eq!(weights.len(), 2);
    assert!((weights["naive"] - (1.0 / 3.0)).abs() < 1e-12);
    assert!((weights["seasonal"] - (2.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn rule_based_gating_uses_series_scores_for_panel_frames() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::for_series("naive", "PU1->DO2", "mae", 1.0),
        ExpertScore::for_series("seasonal", "PU1->DO2", "mae", 3.0),
        ExpertScore::for_series("naive", "PU3->DO4", "mae", 3.0),
        ExpertScore::for_series("seasonal", "PU3->DO4", "mae", 1.0),
    ])
    .expect("score table");
    let gating = RuleBasedGating::new("mae", table).expect("gating");
    let frame = taxi_panel_frame();

    let weights = gating.weights_for_frame(&frame).expect("weights");

    assert!((weights["naive"] - 0.5).abs() < 1e-12);
    assert!((weights["seasonal"] - 0.5).abs() < 1e-12);
}

#[test]
fn forecast_ensemble_alias_keeps_weighted_deterministic_predictions() {
    let mut ensemble = ForecastEnsemble::new(vec![
        ("naive".to_string(), Box::new(NaiveForecaster::new()), 1.0),
        (
            "seasonal".to_string(),
            Box::new(SeasonalNaiveForecaster::new(2).expect("seasonal")),
            3.0,
        ),
    ])
    .expect("ensemble");
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::single(ts(1), 10.0),
            ForecastRow::single(ts(2), 20.0),
            ForecastRow::single(ts(3), 30.0),
            ForecastRow::single(ts(4), 40.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("frame");

    ensemble.fit(&frame).expect("fit");
    let predictions = ensemble.predict(2).expect("predict");
    let means = predictions
        .predictions()
        .iter()
        .map(|prediction| prediction.mean)
        .collect::<Vec<_>>();

    assert_eq!(means, vec![32.5, 40.0]);
}

fn taxi_panel_frame() -> ForecastFrame {
    ForecastFrame::new(
        vec![
            ForecastRow::new("PU1->DO2", ts(1), 10.0),
            ForecastRow::new("PU1->DO2", ts(2), 12.0),
            ForecastRow::new("PU1->DO2", ts(3), 14.0),
            ForecastRow::new("PU3->DO4", ts(1), 30.0),
            ForecastRow::new("PU3->DO4", ts(2), 28.0),
            ForecastRow::new("PU3->DO4", ts(3), 26.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("frame")
}

fn ts(day: u32) -> chrono::NaiveDateTime {
    NaiveDate::from_ymd_opt(2024, 1, day)
        .expect("date")
        .and_hms_opt(0, 0, 0)
        .expect("time")
}
