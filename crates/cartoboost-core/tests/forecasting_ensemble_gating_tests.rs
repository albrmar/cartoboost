use cartoboost_core::forecasting::{
    AutoForecastConfig, AutoForecastModel, ExpertScore, ForecastEnsemble, ForecastFrame,
    ForecastFrequency, ForecastObjective, ForecastRow, Forecaster, LagFeatureConfig,
    NaiveForecaster, RuleBasedGating, RuleBasedGatingGuardrails, SeasonalNaiveForecaster,
    ValidationScoreTable,
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
fn rule_based_gating_can_use_hard_winner_when_validation_gap_is_decisive() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("cartoboost_lag", "wrmsse", 0.50),
        ExpertScore::global("classical_bank", "wrmsse", 0.80),
        ExpertScore::global("direct_tree", "wrmsse", 0.90),
    ])
    .expect("score table");
    let gating = RuleBasedGating::with_guardrails(
        "wrmsse",
        table,
        RuleBasedGatingGuardrails {
            top_k: Some(3),
            hard_winner_relative_gain: Some(0.05),
            weight_bounds: Some((0.15, 0.85)),
            ..RuleBasedGatingGuardrails::default()
        },
    )
    .expect("gating");

    let weights = gating.weights_for(None, None).expect("weights");

    assert_eq!(
        weights,
        std::collections::BTreeMap::from([("cartoboost_lag".to_string(), 1.0)])
    );
}

#[test]
fn rule_based_gating_protects_baseline_until_displacement_gain_is_met() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("cartoboost_lag", "rps", 0.257),
        ExpertScore::global("point_auto", "rps", 0.252),
    ])
    .expect("score table");
    let gating = RuleBasedGating::with_guardrails(
        "rps",
        table,
        RuleBasedGatingGuardrails {
            top_k: Some(2),
            weight_bounds: Some((0.15, 0.85)),
            baseline: Some("cartoboost_lag".to_string()),
            baseline_displacement_gain: Some(0.03),
            ..RuleBasedGatingGuardrails::default()
        },
    )
    .expect("gating");

    let weights = gating.weights_for(None, None).expect("weights");

    assert_eq!(
        weights,
        std::collections::BTreeMap::from([("cartoboost_lag".to_string(), 1.0)])
    );
}

#[test]
fn rule_based_gating_bounds_close_race_blends() {
    let table = ValidationScoreTable::new(vec![
        ExpertScore::global("lag_plus", "rmse", 1.00),
        ExpertScore::global("direct_tree", "rmse", 1.01),
    ])
    .expect("score table");
    let gating = RuleBasedGating::with_guardrails(
        "rmse",
        table,
        RuleBasedGatingGuardrails {
            top_k: Some(2),
            hard_winner_relative_gain: Some(0.05),
            weight_bounds: Some((0.15, 0.85)),
            ..RuleBasedGatingGuardrails::default()
        },
    )
    .expect("gating");

    let weights = gating.weights_for(None, None).expect("weights");

    assert_eq!(weights.len(), 2);
    assert!(weights
        .values()
        .all(|weight| (0.15..=0.85).contains(weight)));
    assert!((weights.values().sum::<f64>() - 1.0).abs() < 1e-12);
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

#[test]
fn auto_forecast_model_fits_guarded_hybrid_and_emits_weights() {
    let frame = ForecastFrame::new(
        (1..=18)
            .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day)))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![2],
        },
        validation_window: Some(2),
        objective: ForecastObjective::Rmse,
        season_length: 7,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let forecast = model.predict(2).expect("forecast");
    let metadata = model.metadata();

    assert_eq!(forecast.predictions().len(), 2);
    assert_eq!(metadata["model"], "cartoboost_auto_forecast");
    assert_eq!(metadata["objective"], "rmse");
    assert_eq!(metadata["nonnegative_output"], true);
    assert!(!metadata["weights"].as_object().expect("weights").is_empty());
    assert!(forecast
        .predictions()
        .iter()
        .all(|prediction| prediction.mean >= 0.0));
    assert_eq!(
        metadata["validation_scores"]
            .as_array()
            .expect("scores")
            .iter()
            .filter(|score| score["metric"] == "rmse"
                && score["series_id"].is_null()
                && score["horizon"].is_null())
            .count(),
        9
    );
    assert!(metadata["horizon_weights"]
        .as_object()
        .expect("horizon weights")
        .contains_key("1"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["metric"] == "rmse" && score["horizon"] == 1));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "lag_plus"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "scaled_lag"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "delta_lag"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "scaled_delta_lag"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "cartoboost_direct"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "cartoboost_rectified_recursive"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "log1p_scaled_lag"));
}

#[test]
fn auto_forecast_model_skips_log1p_candidate_for_negative_targets() {
    let frame = ForecastFrame::new(
        (1..=18)
            .map(|day| ForecastRow::single(ts(day), f64::from(day) - 10.0))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![2],
        },
        validation_window: Some(2),
        objective: ForecastObjective::Rmse,
        season_length: 7,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();

    assert_eq!(metadata["nonnegative_output"], false);
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .all(|score| score["expert"] != "log1p_scaled_lag"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .all(|score| score["expert"] != "intermittent_demand"));
}

#[test]
fn auto_forecast_model_includes_intermittent_candidate_for_sparse_demand() {
    let values = [
        0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0, 11.0, 0.0, 0.0, 12.0, 0.0, 0.0,
        13.0,
    ];
    let frame = ForecastFrame::new(
        values
            .iter()
            .enumerate()
            .map(|(index, value)| ForecastRow::single(ts(index as u32 + 1), *value))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![2],
        },
        validation_window: Some(3),
        objective: ForecastObjective::Wape,
        season_length: 7,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();

    assert_eq!(metadata["nonnegative_output"], true);
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["expert"] == "intermittent_demand"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .all(|score| score["expert"] != "cartoboost_direct"
            && score["expert"] != "cartoboost_rectified_recursive"));
}

#[test]
fn auto_forecast_model_uses_series_weights_for_panels() {
    let mut rows = Vec::new();
    for day in 1..=18 {
        rows.push(ForecastRow::new(
            "pickup_zone_1",
            ts(day),
            10.0 + f64::from(day),
        ));
        rows.push(ForecastRow::new(
            "pickup_zone_2",
            ts(day),
            60.0 - f64::from(day * 2),
        ));
    }
    let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![1, 2],
            rolling_mean_windows: vec![2],
            calendar_features: Vec::new(),
            difference_lags: vec![2],
            rolling_trend_windows: vec![2],
        },
        validation_window: Some(2),
        objective: ForecastObjective::Rmse,
        season_length: 7,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let forecast = model.predict(2).expect("forecast");
    let metadata = model.metadata();
    let series_weights = metadata["series_weights"]
        .as_object()
        .expect("series weights");

    assert_eq!(forecast.predictions().len(), 4);
    assert!(series_weights.contains_key("pickup_zone_1"));
    assert!(series_weights.contains_key("pickup_zone_2"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["series_id"] == "pickup_zone_1"));
    assert!(metadata["validation_scores"]
        .as_array()
        .expect("scores")
        .iter()
        .any(|score| score["series_id"] == "pickup_zone_2"));
}

#[test]
fn auto_forecast_model_falls_back_when_lag_cannot_validate() {
    let frame = ForecastFrame::new(
        (1..=8)
            .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day)))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("frame");
    let mut model = AutoForecastModel::new(AutoForecastConfig {
        lag_config: LagFeatureConfig {
            lags: vec![28],
            rolling_mean_windows: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
        },
        validation_window: Some(2),
        season_length: 2,
        ..AutoForecastConfig::default()
    })
    .expect("model");

    model.fit(&frame).expect("fit");
    let metadata = model.metadata();

    let weights = metadata["weights"].as_object().expect("weights");
    assert_eq!(weights.len(), 1);
    assert!(weights.contains_key("classical_expert_bank"));
    assert_eq!(model.predict(1).expect("forecast").predictions().len(), 1);
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
