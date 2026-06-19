use cartoboost_core::booster::BoosterConfig;
use cartoboost_core::forecasting::{
    ArimaForecaster, AutoARIMAForecaster, CalendarFeature, CartoBoostLagForecaster, ETSForecaster,
    ForecastFrame, ForecastFrequency, ForecastRow, Forecaster, GlobalForecastTargetMode,
    KalmanForecaster, KrigingForecaster, LagFeatureConfig, NaiveForecaster,
    OptimizedThetaForecaster, SeasonalNaiveForecaster, ThetaForecaster, WeightedEnsembleForecaster,
};
use chrono::{NaiveDate, NaiveDateTime};
use std::collections::BTreeMap;

fn ts(day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(2026, 1, day)
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .expect("valid fixture timestamp")
}

fn single_frame(values: &[f64]) -> ForecastFrame {
    ForecastFrame::new(
        values
            .iter()
            .enumerate()
            .map(|(idx, value)| ForecastRow::single(ts(idx as u32 + 1), *value))
            .collect(),
        ForecastFrequency::Daily,
    )
    .expect("valid single-series frame")
}

fn means<M: Forecaster>(model: &M, horizon: usize) -> Vec<f64> {
    model
        .predict(horizon)
        .expect("predict")
        .predictions()
        .iter()
        .map(|prediction| prediction.mean)
        .collect()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-6,
        "expected {actual} to be within 1e-6 of {expected}"
    );
}

#[test]
fn naive_and_seasonal_naive_have_exact_known_answer_forecasts() {
    let frame = single_frame(&[10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

    let mut naive = NaiveForecaster::new();
    naive.fit(&frame).expect("fit naive");
    assert_eq!(means(&naive, 3), vec![20.0, 20.0, 20.0]);
    assert_eq!(naive.metadata()["model"], "naive");

    let mut seasonal = SeasonalNaiveForecaster::new(3).expect("valid seasonal naive");
    seasonal.fit(&frame).expect("fit seasonal naive");
    assert_eq!(means(&seasonal, 5), vec![16.0, 18.0, 20.0, 16.0, 18.0]);
    assert_eq!(seasonal.metadata()["season_length"], 3);
}

#[test]
fn theta_models_preserve_constant_series_and_select_best_grid_member() {
    let frame = single_frame(&[42.0, 42.0, 42.0, 42.0, 42.0]);

    let mut theta = ThetaForecaster::new(2.0, 0.4).expect("valid theta");
    theta.fit(&frame).expect("fit theta");
    for mean in means(&theta, 3) {
        assert_close(mean, 42.0);
    }
    assert_eq!(theta.fitted_values("__single__").expect("fitted").len(), 5);
    assert!(theta
        .residuals("__single__")
        .expect("residuals")
        .iter()
        .all(|residual| residual.abs() < 1.0e-6));

    let mut optimized =
        OptimizedThetaForecaster::new(vec![2.0, 1.0], vec![0.8, 0.2]).expect("valid grid");
    optimized.fit(&frame).expect("fit optimized theta");
    assert_eq!(optimized.selected_theta(), Some(1.0));
    assert_eq!(optimized.selected_alpha(), Some(0.2));
    assert_eq!(optimized.validation_scores().len(), 4);
    for mean in means(&optimized, 2) {
        assert_close(mean, 42.0);
    }
    assert_eq!(optimized.metadata()["model"], "optimized_theta");
}

#[test]
fn ets_and_arima_reproduce_linear_series_known_answers() {
    let frame = single_frame(&[10.0, 12.0, 14.0, 16.0]);

    let mut ets = ETSForecaster::new(1.0, 1.0).expect("valid ets");
    ets.fit(&frame).expect("fit ets");
    assert_eq!(means(&ets, 2), vec![18.0, 20.0]);
    assert_eq!(
        ets.fitted_values("__single__").expect("fitted"),
        &[10.0, 12.0, 14.0, 16.0]
    );
    assert!(ets
        .residuals("__single__")
        .expect("residuals")
        .iter()
        .all(|residual| residual.abs() < 1.0e-6));

    let mut arima = ArimaForecaster::new(0, 1, 0).expect("valid arima");
    arima.fit(&frame).expect("fit arima");
    assert_eq!(arima.order(), (0, 1, 0));
    assert_eq!(means(&arima, 2), vec![18.0, 20.0]);
    assert_eq!(
        arima.fitted_values("__single__").expect("fitted"),
        &[10.0, 12.0, 14.0, 16.0]
    );

    let mut auto_arima = AutoARIMAForecaster::with_max_order(0, 1, 0).expect("valid auto arima");
    auto_arima.fit(&frame).expect("fit auto arima");
    assert_eq!(auto_arima.selected_order(), Some((0, 1, 0)));
    assert_eq!(means(&auto_arima, 2), vec![18.0, 20.0]);
    assert_eq!(auto_arima.metadata()["selected_order"]["d"], 1);
}

#[test]
fn kalman_tracks_linear_signal_and_preserves_panel_indexing() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PULocationID=132", ts(1), 10.0),
            ForecastRow::new("PULocationID=132", ts(2), 12.0),
            ForecastRow::new("PULocationID=132", ts(3), 14.0),
            ForecastRow::new("PULocationID=132", ts(4), 16.0),
            ForecastRow::new("PULocationID=236", ts(1), 40.0),
            ForecastRow::new("PULocationID=236", ts(2), 38.0),
            ForecastRow::new("PULocationID=236", ts(3), 36.0),
            ForecastRow::new("PULocationID=236", ts(4), 34.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid panel frame");
    let mut model = KalmanForecaster::new(10.0, 10.0, 1.0e-6).expect("valid kalman");

    model.fit(&frame).expect("fit kalman");
    let predictions = model.predict(2).expect("predict").predictions().to_vec();

    assert_eq!(predictions.len(), 4);
    assert_eq!(predictions[0].series_id, "PULocationID=132");
    assert_eq!(predictions[0].timestamp, ts(5));
    assert_eq!(predictions[1].horizon, 2);
    assert_eq!(predictions[2].series_id, "PULocationID=236");
    assert!(predictions[1].mean > predictions[0].mean);
    assert!(predictions[3].mean < predictions[2].mean);
    assert_eq!(model.metadata()["model"], "kalman");
}

#[test]
fn kriging_interpolates_observed_coordinates_and_reports_configuration() {
    let frame = ForecastFrame::new(
        vec![
            ForecastRow::new("PULocationID=142", ts(1), 10.0),
            ForecastRow::new("PULocationID=142", ts(2), 12.0),
            ForecastRow::new("PULocationID=236", ts(1), 40.0),
            ForecastRow::new("PULocationID=236", ts(2), 42.0),
        ],
        ForecastFrequency::Daily,
    )
    .expect("valid panel frame");
    let coordinates = BTreeMap::from([
        ("PULocationID=142".to_string(), (0.0, 0.0)),
        ("PULocationID=236".to_string(), (10.0, 0.0)),
    ]);
    let mut model = KrigingForecaster::new(coordinates, 1.0, 1.0e-9).expect("valid kriging");

    model.fit(&frame).expect("fit kriging");
    let forecast = model.predict(1).expect("predict");
    let predictions = forecast.predictions();

    assert_close(predictions[0].mean, 12.0);
    assert_close(predictions[1].mean, 42.0);
    assert_eq!(model.metadata()["series_count"], 2);
    assert_eq!(model.metadata()["model"], "kriging");
}

#[test]
fn weighted_ensemble_uses_normalized_weights_for_exact_member_average() {
    let frame = single_frame(&[10.0, 12.0, 14.0]);
    let mut model = WeightedEnsembleForecaster::new(vec![
        ("last".to_string(), Box::new(NaiveForecaster::new()), 1.0),
        (
            "seasonal".to_string(),
            Box::new(SeasonalNaiveForecaster::new(2).expect("valid seasonal")),
            3.0,
        ),
    ])
    .expect("valid ensemble");

    model.fit(&frame).expect("fit ensemble");

    assert_eq!(means(&model, 2), vec![12.5, 14.0]);
    assert_close(model.weights()["last"], 0.25);
    assert_close(model.weights()["seasonal"], 0.75);
    assert_eq!(model.metadata()["model"], "weighted_ensemble");
}

#[test]
fn cartoboost_lag_delta_mode_learns_linear_increment_exactly() {
    let frame = single_frame(&[10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    let booster_config = BoosterConfig {
        n_estimators: 1,
        learning_rate: 1.0,
        max_depth: 0,
        min_samples_leaf: 1,
        min_gain: 0.0,
        ..Default::default()
    };
    let mut model = CartoBoostLagForecaster::new_with_target_mode(
        LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            difference_lags: vec![1],
            rolling_trend_windows: Vec::new(),
            calendar_features: vec![CalendarFeature::DayOfWeek],
        },
        booster_config,
        GlobalForecastTargetMode::DeltaFromLast,
    )
    .expect("valid cartoboost lag");

    model.fit(&frame).expect("fit cartoboost lag");

    assert_eq!(model.training_rows(), Some(4));
    assert_eq!(means(&model, 2), vec![22.0, 24.0]);
    assert_eq!(model.metadata()["target_mode"], "delta_from_last");
}
