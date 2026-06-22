use cartoboost_core::forecasting::{
    pinball_loss, rank_probability_score, repair_non_crossing_quantiles,
    weighted_blend_candidate_forecast, ConformalCalibrator, ProbabilisticDirectForecaster,
    RankProbabilityForecast,
};
use cartoboost_core::metrics::{
    calibrated_rank_bucket_probabilities, extreme_portfolio_decisions, portfolio_summary,
    rank_buckets, rank_hit_rates, rank_portfolio_decision_loss, rank_portfolio_summary,
    rank_probability_calibration, rank_scored_assets, PortfolioAsset, PortfolioDecision,
    PortfolioSide, RankBucketPrediction,
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

#[test]
fn portfolio_summary_scores_long_short_decisions() {
    let decisions = [
        PortfolioDecision {
            side: PortfolioSide::Short,
            weight: -0.25,
            actual_return: -0.04,
            predicted_return: -0.02,
        },
        PortfolioDecision {
            side: PortfolioSide::Short,
            weight: -0.25,
            actual_return: 0.02,
            predicted_return: -0.01,
        },
        PortfolioDecision {
            side: PortfolioSide::Long,
            weight: 0.5,
            actual_return: 0.06,
            predicted_return: 0.03,
        },
    ];

    let summary = portfolio_summary(&decisions).expect("summary");

    assert_eq!(summary.long_count, 1);
    assert_eq!(summary.short_count, 2);
    assert!((summary.gross_exposure - 1.0).abs() < 1e-12);
    assert!(summary.net_exposure.abs() < 1e-12);
    assert!((summary.long_return - 0.03).abs() < 1e-12);
    assert!((summary.short_return - 0.005).abs() < 1e-12);
    assert!((summary.net_return - 0.035).abs() < 1e-12);
}

#[test]
fn extreme_portfolio_decisions_select_top_and_bottom_fifths() {
    let assets = vec![
        PortfolioAsset {
            series_id: "a".to_string(),
            actual_return: -0.03,
            predicted_return: -0.02,
        },
        PortfolioAsset {
            series_id: "b".to_string(),
            actual_return: 0.01,
            predicted_return: -0.01,
        },
        PortfolioAsset {
            series_id: "c".to_string(),
            actual_return: 0.02,
            predicted_return: 0.00,
        },
        PortfolioAsset {
            series_id: "d".to_string(),
            actual_return: 0.03,
            predicted_return: 0.01,
        },
        PortfolioAsset {
            series_id: "e".to_string(),
            actual_return: 0.04,
            predicted_return: 0.02,
        },
        PortfolioAsset {
            series_id: "f".to_string(),
            actual_return: 0.05,
            predicted_return: 0.03,
        },
        PortfolioAsset {
            series_id: "g".to_string(),
            actual_return: -0.01,
            predicted_return: -0.03,
        },
        PortfolioAsset {
            series_id: "h".to_string(),
            actual_return: 0.06,
            predicted_return: 0.04,
        },
        PortfolioAsset {
            series_id: "i".to_string(),
            actual_return: 0.07,
            predicted_return: 0.05,
        },
        PortfolioAsset {
            series_id: "j".to_string(),
            actual_return: 0.08,
            predicted_return: 0.06,
        },
    ];

    let decisions = extreme_portfolio_decisions(&assets).expect("decisions");

    assert_eq!(decisions.len(), 4);
    assert_eq!(decisions[0].series_id, "i");
    assert_eq!(decisions[0].side, PortfolioSide::Long);
    assert!((decisions[0].weight - 0.25).abs() < 1e-12);
    assert_eq!(decisions[1].series_id, "j");
    assert_eq!(decisions[1].side, PortfolioSide::Long);
    assert_eq!(decisions[2].series_id, "a");
    assert_eq!(decisions[2].side, PortfolioSide::Short);
    assert!((decisions[2].weight + 0.25).abs() < 1e-12);
    assert_eq!(decisions[3].series_id, "g");
    assert_eq!(decisions[3].side, PortfolioSide::Short);
}

#[test]
fn rank_hit_rates_score_exact_near_and_extreme_predictions() {
    let rows = [
        RankBucketPrediction {
            observed_bucket: 0,
            predicted_bucket: 0,
        },
        RankBucketPrediction {
            observed_bucket: 2,
            predicted_bucket: 1,
        },
        RankBucketPrediction {
            observed_bucket: 4,
            predicted_bucket: 0,
        },
    ];

    let summary = rank_hit_rates(&rows, 5).expect("hit rates");

    assert_eq!(summary.asset_count, 3);
    assert!((summary.exact_bucket_rate - (1.0 / 3.0)).abs() < 1e-12);
    assert!((summary.within_one_bucket_rate - (2.0 / 3.0)).abs() < 1e-12);
    assert_eq!(summary.directional_extreme_count, 2);
    assert!((summary.directional_extreme_rate - 0.5).abs() < 1e-12);
}

#[test]
fn rank_buckets_are_deterministic_with_index_tie_breaks() {
    let buckets = rank_buckets(&[2.0, 1.0, 1.0, 5.0, 4.0], 5).expect("buckets");
    assert_eq!(buckets, vec![2, 0, 1, 4, 3]);

    let coarse = rank_buckets(&[10.0, 20.0, 30.0, 40.0], 2).expect("coarse");
    assert_eq!(coarse, vec![0, 0, 1, 1]);
}

#[test]
fn rank_scored_assets_compute_buckets_probabilities_and_rps() {
    let assets = vec![
        PortfolioAsset {
            series_id: "a".to_string(),
            actual_return: -0.03,
            predicted_return: -0.02,
        },
        PortfolioAsset {
            series_id: "b".to_string(),
            actual_return: 0.01,
            predicted_return: -0.01,
        },
        PortfolioAsset {
            series_id: "c".to_string(),
            actual_return: 0.04,
            predicted_return: 0.03,
        },
    ];
    let calibration = vec![vec![1.0 / 3.0; 3]; 3];

    let scored = rank_scored_assets(&assets, 3, &calibration, 0.0).expect("scored");

    assert_eq!(scored.len(), 3);
    assert_eq!(scored[0].series_id, "a");
    assert_eq!(scored[0].observed_rank_bucket, 0);
    assert_eq!(scored[0].predicted_rank_bucket, 0);
    assert_eq!(scored[0].rank_probabilities, vec![1.0 / 3.0; 3]);
    assert!((scored[0].rps - (5.0 / 18.0)).abs() < 1e-12);
    assert_eq!(scored[1].observed_rank_bucket, 1);
    assert!((scored[1].rps - (1.0 / 9.0)).abs() < 1e-12);
    assert_eq!(scored[2].observed_rank_bucket, 2);
    assert!((scored[2].rps - (5.0 / 18.0)).abs() < 1e-12);
}

#[test]
fn rank_portfolio_summary_combines_rps_portfolio_and_hit_rates() {
    let assets = vec![
        PortfolioAsset {
            series_id: "a".to_string(),
            actual_return: -0.03,
            predicted_return: -0.02,
        },
        PortfolioAsset {
            series_id: "b".to_string(),
            actual_return: 0.01,
            predicted_return: -0.01,
        },
        PortfolioAsset {
            series_id: "c".to_string(),
            actual_return: 0.04,
            predicted_return: 0.03,
        },
    ];
    let calibration = vec![vec![1.0 / 3.0; 3]; 3];

    let summary = rank_portfolio_summary(&assets, 3, &calibration, 0.0).expect("summary");

    assert_eq!(summary.asset_count, 3);
    assert!((summary.mean_rps - (2.0 / 9.0)).abs() < 1e-12);
    assert_eq!(summary.decisions.len(), 2);
    assert_eq!(summary.portfolio.long_count, 1);
    assert_eq!(summary.portfolio.short_count, 1);
    assert_eq!(summary.hit_rates.asset_count, 3);
    assert!((summary.hit_rates.exact_bucket_rate - 1.0).abs() < 1e-12);
}

#[test]
fn rank_portfolio_decision_loss_prioritizes_portfolio_return() {
    let calibration = vec![vec![1.0 / 3.0; 3]; 3];
    let profitable = vec![
        PortfolioAsset {
            series_id: "a".to_string(),
            actual_return: -0.05,
            predicted_return: -0.04,
        },
        PortfolioAsset {
            series_id: "b".to_string(),
            actual_return: 0.0,
            predicted_return: 0.0,
        },
        PortfolioAsset {
            series_id: "c".to_string(),
            actual_return: 0.05,
            predicted_return: 0.04,
        },
    ];
    let inverted = vec![
        PortfolioAsset {
            series_id: "a".to_string(),
            actual_return: -0.05,
            predicted_return: 0.04,
        },
        PortfolioAsset {
            series_id: "b".to_string(),
            actual_return: 0.0,
            predicted_return: 0.0,
        },
        PortfolioAsset {
            series_id: "c".to_string(),
            actual_return: 0.05,
            predicted_return: -0.04,
        },
    ];

    let profitable_loss =
        rank_portfolio_decision_loss(&profitable, 3, &calibration, 0.0, 1.0e-4).expect("loss");
    let inverted_loss =
        rank_portfolio_decision_loss(&inverted, 3, &calibration, 0.0, 1.0e-4).expect("loss");

    assert!(profitable_loss < 0.0);
    assert!(profitable_loss < inverted_loss);
    assert!(rank_portfolio_decision_loss(&profitable, 3, &calibration, 0.0, -1.0).is_err());
}

#[test]
fn rank_probability_calibration_smooths_validation_confusion() {
    let calibration =
        rank_probability_calibration(&[0, 1, 2, 2], &[0, 1, 1, 2], 3, 4).expect("calibration");

    assert_eq!(calibration.metadata.bucket_count, 3);
    assert_eq!(calibration.metadata.validation_support, 4);
    assert_eq!(calibration.metadata.fallback, "none");
    assert!((calibration.shrinkage - (4.0 / 64.0)).abs() < 1e-12);
    assert_eq!(calibration.probabilities.len(), 3);
    assert!((calibration.probabilities[1][2] - 0.4).abs() < 1e-12);

    let probabilities = calibrated_rank_bucket_probabilities(
        1,
        3,
        &calibration.probabilities,
        calibration.shrinkage,
    )
    .expect("probabilities");

    assert_eq!(probabilities.len(), 3);
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(probabilities[2] > probabilities[0]);
}

#[test]
fn rank_probability_calibration_falls_back_to_uniform_without_support() {
    let calibration = rank_probability_calibration(&[], &[], 5, 0).expect("calibration");
    let probabilities = calibrated_rank_bucket_probabilities(
        3,
        5,
        &calibration.probabilities,
        calibration.shrinkage,
    )
    .expect("probabilities");

    assert_eq!(
        calibration.metadata.fallback,
        "uniform_when_no_validation_support"
    );
    assert_eq!(probabilities, vec![0.2; 5]);
}

#[test]
fn weighted_blend_candidate_is_native_validated() {
    let forecast =
        weighted_blend_candidate_forecast(&[1.0, -2.0], &[3.0, 4.0], 0.85).expect("blend");
    assert!((forecast[0] - 1.3).abs() < 1e-12);
    assert!((forecast[1] + 1.1).abs() < 1e-12);

    let err = weighted_blend_candidate_forecast(&[1.0], &[2.0, 3.0], 0.85)
        .expect_err("length mismatch should fail");
    assert!(err.to_string().contains("equal length"));

    let err = weighted_blend_candidate_forecast(&[1.0], &[2.0], 1.2)
        .expect_err("invalid weight should fail");
    assert!(err.to_string().contains("between 0 and 1"));
}
