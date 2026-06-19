use cartoboost_core::forecasting::{
    pinball_loss, rank_probability_score, repair_non_crossing_quantiles, ConformalCalibrator,
    ProbabilisticDirectForecaster, RankProbabilityForecast,
};

#[test]
fn quantile_repair_and_pinball_are_deterministic() {
    let repaired = repair_non_crossing_quantiles(&[12.0, 10.0, 13.0]).expect("repair");
    assert_eq!(repaired, vec![12.0, 12.0, 13.0]);

    let loss = pinball_loss(&[10.0, 20.0], &[11.0, 18.0], 0.5).expect("pinball");
    assert!((loss - 0.75).abs() < 1e-12);
}

#[test]
fn probabilistic_direct_repairs_each_horizon() {
    let forecaster = ProbabilisticDirectForecaster::new(vec![0.1, 0.5, 0.9]).expect("forecaster");
    let repaired = forecaster
        .repair_matrix(&[vec![3.0, 2.0, 5.0], vec![1.0, 4.0, 3.0]])
        .expect("repair matrix");

    assert_eq!(repaired[0].values, vec![3.0, 3.0, 5.0]);
    assert_eq!(repaired[1].values, vec![1.0, 4.0, 4.0]);
}

#[test]
fn conformal_calibrator_enforces_split_ordering() {
    let mut calibrator = ConformalCalibrator::new(0.5).expect("calibrator");
    calibrator
        .fit_with_strict_ordering(&[10.0, 20.0], &[11.0, 18.0], 4, 4, 6, 6)
        .expect("fit");
    let interval = calibrator
        .predict_interval(&[30.0, 40.0], 6)
        .expect("interval");

    assert_eq!(interval.residual_quantile, 2.0);
    assert_eq!(interval.lower, vec![28.0, 38.0]);
    assert_eq!(interval.upper, vec![32.0, 42.0]);

    let err = ConformalCalibrator::new(0.1)
        .expect("calibrator")
        .fit_with_strict_ordering(&[1.0], &[1.0], 5, 4, 6, 6)
        .expect_err("overlapping train/calibration split should fail");
    assert!(err.to_string().contains("training rows must end"));
}

#[test]
fn rank_probability_score_uses_ordered_cdf_distance() {
    let score = rank_probability_score(&[0.2, 0.3, 0.5], 2).expect("rps");
    assert!((score - 0.145).abs() < 1e-12);

    let forecast = RankProbabilityForecast::new(vec![0.2, 0.3, 0.5]).expect("forecast");
    assert!((forecast.score(2).expect("score") - score).abs() < 1e-12);
}
