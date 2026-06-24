use crate::Result;

use super::pinball::pinball_loss;
use super::rps::rank_probability_score;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankPortfolioMetricSummary {
    pub pinball_loss: f64,
    pub rank_probability_score: f64,
    pub combined_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortfolioSide {
    Long,
    Short,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortfolioDecision {
    pub side: PortfolioSide,
    pub weight: f64,
    pub actual_return: f64,
    pub predicted_return: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioAsset {
    pub series_id: String,
    pub actual_return: f64,
    pub predicted_return: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioDecisionRow {
    pub series_id: String,
    pub side: PortfolioSide,
    pub weight: f64,
    pub actual_return: f64,
    pub predicted_return: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortfolioSummary {
    pub long_count: usize,
    pub short_count: usize,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub long_return: f64,
    pub short_return: f64,
    pub net_return: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankBucketPrediction {
    pub observed_bucket: usize,
    pub predicted_bucket: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankScoredAsset {
    pub series_id: String,
    pub actual_return: f64,
    pub predicted_return: f64,
    pub observed_rank_bucket: usize,
    pub predicted_rank_bucket: usize,
    pub rank_probabilities: Vec<f64>,
    pub rps: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankPortfolioSummary {
    pub mean_rps: f64,
    pub asset_count: usize,
    pub assets: Vec<RankScoredAsset>,
    pub decisions: Vec<PortfolioDecisionRow>,
    pub portfolio: PortfolioSummary,
    pub hit_rates: RankHitRateSummary,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankHitRateSummary {
    pub asset_count: usize,
    pub exact_bucket_rate: f64,
    pub within_one_bucket_rate: f64,
    pub directional_extreme_count: usize,
    pub directional_extreme_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankProbabilityCalibration {
    pub probabilities: Vec<Vec<f64>>,
    pub shrinkage: f64,
    pub metadata: RankProbabilityCalibrationMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankProbabilityCalibrationMetadata {
    pub method: String,
    pub bucket_count: usize,
    pub validation_support: usize,
    pub dirichlet_prior: f64,
    pub shrinkage_to_confusion: f64,
    pub fallback: String,
}

pub fn rank_portfolio_combined_score(
    pinball_loss: f64,
    rank_probability_score: f64,
) -> Result<f64> {
    if !pinball_loss.is_finite() || pinball_loss < 0.0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "pinball_loss must be finite and non-negative".to_string(),
        ));
    }
    if !rank_probability_score.is_finite() || rank_probability_score < 0.0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "rank_probability_score must be finite and non-negative".to_string(),
        ));
    }
    Ok(0.5 * pinball_loss + 0.5 * rank_probability_score)
}

pub fn evaluate_rank_portfolio_metrics(
    actual_returns: &[f64],
    quantile_predictions: &[f64],
    quantile: f64,
    rank_probabilities: &[f64],
    observed_rank: usize,
) -> Result<RankPortfolioMetricSummary> {
    let pinball = pinball_loss(actual_returns, quantile_predictions, quantile)?;
    let rps = rank_probability_score(rank_probabilities, observed_rank)?;
    Ok(RankPortfolioMetricSummary {
        pinball_loss: pinball,
        rank_probability_score: rps,
        combined_score: rank_portfolio_combined_score(pinball, rps)?,
    })
}

pub fn rank_probability_calibration(
    actual_buckets: &[usize],
    predicted_buckets: &[usize],
    bucket_count: usize,
    validation_support: usize,
) -> Result<RankProbabilityCalibration> {
    if bucket_count == 0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "bucket_count must be positive".to_string(),
        ));
    }
    if actual_buckets.len() != predicted_buckets.len() {
        return Err(crate::CartoBoostError::InvalidInput(
            "actual and predicted bucket lists must have the same length".to_string(),
        ));
    }
    let prior = 1.0;
    let mut rows = vec![vec![prior; bucket_count]; bucket_count];
    for (&actual_bucket, &predicted_bucket) in actual_buckets.iter().zip(predicted_buckets) {
        if actual_bucket >= bucket_count || predicted_bucket >= bucket_count {
            return Err(crate::CartoBoostError::InvalidInput(
                "actual and predicted buckets must be inside bucket_count".to_string(),
            ));
        }
        rows[predicted_bucket][actual_bucket] += 1.0;
    }
    let probabilities = rows
        .into_iter()
        .map(|row| {
            let total = row.iter().sum::<f64>();
            row.into_iter()
                .map(|value| value / total)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let shrinkage = if validation_support == 0 {
        0.0
    } else {
        validation_support as f64 / (validation_support as f64 + bucket_count as f64 * 20.0)
    };
    Ok(RankProbabilityCalibration {
        probabilities,
        shrinkage,
        metadata: RankProbabilityCalibrationMetadata {
            method: "dirichlet_confusion_with_uniform_shrinkage".to_string(),
            bucket_count,
            validation_support,
            dirichlet_prior: prior,
            shrinkage_to_confusion: shrinkage,
            fallback: if validation_support == 0 {
                "uniform_when_no_validation_support".to_string()
            } else {
                "none".to_string()
            },
        },
    })
}

pub fn calibrated_rank_bucket_probabilities(
    predicted_bucket: usize,
    bucket_count: usize,
    calibration_probabilities: &[Vec<f64>],
    shrinkage: f64,
) -> Result<Vec<f64>> {
    if bucket_count == 0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "bucket_count must be positive".to_string(),
        ));
    }
    if predicted_bucket >= bucket_count {
        return Err(crate::CartoBoostError::InvalidInput(
            "predicted_bucket must be inside bucket_count".to_string(),
        ));
    }
    if !shrinkage.is_finite() || !(0.0..=1.0).contains(&shrinkage) {
        return Err(crate::CartoBoostError::InvalidInput(
            "shrinkage must be finite and between 0 and 1".to_string(),
        ));
    }
    if calibration_probabilities.len() != bucket_count {
        return Err(crate::CartoBoostError::InvalidInput(
            "calibration probability rows must match bucket_count".to_string(),
        ));
    }
    let row = calibration_probabilities
        .get(predicted_bucket)
        .expect("predicted bucket was validated");
    if row.len() != bucket_count {
        return Err(crate::CartoBoostError::InvalidInput(
            "calibration probability columns must match bucket_count".to_string(),
        ));
    }
    if row.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(crate::CartoBoostError::InvalidInput(
            "calibration probabilities must be finite and non-negative".to_string(),
        ));
    }
    let row_sum = row.iter().sum::<f64>();
    if (row_sum - 1.0).abs() > 1.0e-9 {
        return Err(crate::CartoBoostError::InvalidInput(
            "calibration probability rows must sum to 1".to_string(),
        ));
    }
    let uniform = 1.0 / bucket_count as f64;
    Ok(row
        .iter()
        .map(|row_value| shrinkage * row_value + (1.0 - shrinkage) * uniform)
        .collect())
}

pub fn portfolio_summary(decisions: &[PortfolioDecision]) -> Result<PortfolioSummary> {
    let mut long_count = 0usize;
    let mut short_count = 0usize;
    let mut gross_exposure = 0.0;
    let mut net_exposure = 0.0;
    let mut long_return = 0.0;
    let mut short_return = 0.0;

    for decision in decisions {
        if !decision.weight.is_finite()
            || !decision.actual_return.is_finite()
            || !decision.predicted_return.is_finite()
        {
            return Err(crate::CartoBoostError::InvalidInput(
                "rank portfolio decision weights and returns must be finite".to_string(),
            ));
        }
        gross_exposure += decision.weight.abs();
        net_exposure += decision.weight;
        let contribution = decision.actual_return * decision.weight;
        match decision.side {
            PortfolioSide::Long => {
                long_count += 1;
                long_return += contribution;
            }
            PortfolioSide::Short => {
                short_count += 1;
                short_return += contribution;
            }
        }
    }

    Ok(PortfolioSummary {
        long_count,
        short_count,
        gross_exposure,
        net_exposure,
        long_return,
        short_return,
        net_return: long_return + short_return,
    })
}

pub fn extreme_portfolio_decisions(
    asset_rows: &[PortfolioAsset],
) -> Result<Vec<PortfolioDecisionRow>> {
    if asset_rows.is_empty() {
        return Ok(Vec::new());
    }
    for row in asset_rows {
        if row.series_id.trim().is_empty() {
            return Err(crate::CartoBoostError::InvalidInput(
                "rank portfolio portfolio asset ids must be non-empty".to_string(),
            ));
        }
        if !row.actual_return.is_finite() || !row.predicted_return.is_finite() {
            return Err(crate::CartoBoostError::InvalidInput(
                "rank portfolio portfolio asset returns must be finite".to_string(),
            ));
        }
    }

    let mut ordered = asset_rows.to_vec();
    ordered.sort_by(|left, right| {
        left.predicted_return
            .total_cmp(&right.predicted_return)
            .then_with(|| left.series_id.cmp(&right.series_id))
    });
    let side_count = (ordered.len() / 5).max(1);
    let short_weight = -0.5 / side_count as f64;
    let long_weight = 0.5 / side_count as f64;

    let mut decisions = Vec::with_capacity(side_count * 2);
    decisions.extend(
        ordered
            .iter()
            .take(side_count)
            .map(|row| PortfolioDecisionRow {
                series_id: row.series_id.clone(),
                side: PortfolioSide::Short,
                weight: short_weight,
                actual_return: row.actual_return,
                predicted_return: row.predicted_return,
            }),
    );
    decisions.extend(
        ordered
            .iter()
            .rev()
            .take(side_count)
            .map(|row| PortfolioDecisionRow {
                series_id: row.series_id.clone(),
                side: PortfolioSide::Long,
                weight: long_weight,
                actual_return: row.actual_return,
                predicted_return: row.predicted_return,
            }),
    );
    decisions.sort_by(|left, right| {
        portfolio_side_sort_key(left.side)
            .cmp(&portfolio_side_sort_key(right.side))
            .then_with(|| left.series_id.cmp(&right.series_id))
    });
    Ok(decisions)
}

fn portfolio_side_sort_key(side: PortfolioSide) -> u8 {
    match side {
        PortfolioSide::Long => 0,
        PortfolioSide::Short => 1,
    }
}

pub fn rank_hit_rates(
    asset_rows: &[RankBucketPrediction],
    bucket_count: usize,
) -> Result<RankHitRateSummary> {
    if bucket_count == 0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "bucket_count must be positive".to_string(),
        ));
    }
    if asset_rows.is_empty() {
        return Ok(RankHitRateSummary {
            asset_count: 0,
            exact_bucket_rate: 0.0,
            within_one_bucket_rate: 0.0,
            directional_extreme_count: 0,
            directional_extreme_rate: 0.0,
        });
    }

    let mut exact = 0usize;
    let mut within_one = 0usize;
    let mut directional_extreme_hits = 0usize;
    let mut directional_extreme_count = 0usize;
    let last_bucket = bucket_count - 1;

    for row in asset_rows {
        if row.observed_bucket >= bucket_count || row.predicted_bucket >= bucket_count {
            return Err(crate::CartoBoostError::InvalidInput(
                "observed and predicted buckets must be inside bucket_count".to_string(),
            ));
        }
        let distance = row.observed_bucket.abs_diff(row.predicted_bucket);
        exact += usize::from(distance == 0);
        within_one += usize::from(distance <= 1);
        if row.predicted_bucket == 0 || row.predicted_bucket == last_bucket {
            directional_extreme_count += 1;
            directional_extreme_hits += usize::from(row.observed_bucket == row.predicted_bucket);
        }
    }

    Ok(RankHitRateSummary {
        asset_count: asset_rows.len(),
        exact_bucket_rate: exact as f64 / asset_rows.len() as f64,
        within_one_bucket_rate: within_one as f64 / asset_rows.len() as f64,
        directional_extreme_count,
        directional_extreme_rate: if directional_extreme_count == 0 {
            0.0
        } else {
            directional_extreme_hits as f64 / directional_extreme_count as f64
        },
    })
}

pub fn rank_buckets(values: &[f64], bucket_count: usize) -> Result<Vec<usize>> {
    if bucket_count == 0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "bucket_count must be positive".to_string(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(crate::CartoBoostError::InvalidInput(
            "rank bucket values must be finite".to_string(),
        ));
    }
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then(left.cmp(&right))
    });

    let mut buckets = vec![0usize; values.len()];
    for (rank, idx) in order.into_iter().enumerate() {
        buckets[idx] = ((rank * bucket_count) / values.len()).min(bucket_count - 1);
    }
    Ok(buckets)
}

pub fn rank_scored_assets(
    asset_rows: &[PortfolioAsset],
    bucket_count: usize,
    calibration_probabilities: &[Vec<f64>],
    shrinkage: f64,
) -> Result<Vec<RankScoredAsset>> {
    if asset_rows.is_empty() {
        return Ok(Vec::new());
    }
    for row in asset_rows {
        if row.series_id.trim().is_empty() {
            return Err(crate::CartoBoostError::InvalidInput(
                "rank portfolio rank-scored asset ids must be non-empty".to_string(),
            ));
        }
        if !row.actual_return.is_finite() || !row.predicted_return.is_finite() {
            return Err(crate::CartoBoostError::InvalidInput(
                "rank portfolio rank-scored asset returns must be finite".to_string(),
            ));
        }
    }

    let actual_values = asset_rows
        .iter()
        .map(|row| row.actual_return)
        .collect::<Vec<_>>();
    let predicted_values = asset_rows
        .iter()
        .map(|row| row.predicted_return)
        .collect::<Vec<_>>();
    let actual_buckets = rank_buckets(&actual_values, bucket_count)?;
    let predicted_buckets = rank_buckets(&predicted_values, bucket_count)?;

    asset_rows
        .iter()
        .zip(actual_buckets)
        .zip(predicted_buckets)
        .map(|((row, observed_rank_bucket), predicted_rank_bucket)| {
            let rank_probabilities = calibrated_rank_bucket_probabilities(
                predicted_rank_bucket,
                bucket_count,
                calibration_probabilities,
                shrinkage,
            )?;
            let rps = rank_probability_score(&rank_probabilities, observed_rank_bucket)?;
            Ok(RankScoredAsset {
                series_id: row.series_id.clone(),
                actual_return: row.actual_return,
                predicted_return: row.predicted_return,
                observed_rank_bucket,
                predicted_rank_bucket,
                rank_probabilities,
                rps,
            })
        })
        .collect()
}

pub fn rank_portfolio_summary(
    asset_rows: &[PortfolioAsset],
    bucket_count: usize,
    calibration_probabilities: &[Vec<f64>],
    shrinkage: f64,
) -> Result<RankPortfolioSummary> {
    let assets = rank_scored_assets(
        asset_rows,
        bucket_count,
        calibration_probabilities,
        shrinkage,
    )?;
    let mean_rps = if assets.is_empty() {
        f64::NAN
    } else {
        assets.iter().map(|row| row.rps).sum::<f64>() / assets.len() as f64
    };
    let decisions = extreme_portfolio_decisions(asset_rows)?;
    let portfolio_decisions = decisions
        .iter()
        .map(|row| PortfolioDecision {
            side: row.side,
            weight: row.weight,
            actual_return: row.actual_return,
            predicted_return: row.predicted_return,
        })
        .collect::<Vec<_>>();
    let portfolio = portfolio_summary(&portfolio_decisions)?;
    let hit_rows = assets
        .iter()
        .map(|row| RankBucketPrediction {
            observed_bucket: row.observed_rank_bucket,
            predicted_bucket: row.predicted_rank_bucket,
        })
        .collect::<Vec<_>>();
    let hit_rates = rank_hit_rates(&hit_rows, bucket_count)?;

    Ok(RankPortfolioSummary {
        mean_rps,
        asset_count: assets.len(),
        assets,
        decisions,
        portfolio,
        hit_rates,
    })
}

pub fn rank_portfolio_decision_loss(
    asset_rows: &[PortfolioAsset],
    bucket_count: usize,
    calibration_probabilities: &[Vec<f64>],
    shrinkage: f64,
    rps_tiebreak_weight: f64,
) -> Result<f64> {
    if !rps_tiebreak_weight.is_finite() || rps_tiebreak_weight < 0.0 {
        return Err(crate::CartoBoostError::InvalidInput(
            "rank portfolio RPS tie-break weight must be finite and non-negative".to_string(),
        ));
    }
    let summary = rank_portfolio_summary(
        asset_rows,
        bucket_count,
        calibration_probabilities,
        shrinkage,
    )?;
    if !summary.portfolio.net_return.is_finite() {
        return Err(crate::CartoBoostError::InvalidInput(
            "rank portfolio investment decision return must be finite".to_string(),
        ));
    }
    if !summary.mean_rps.is_finite() {
        return Err(crate::CartoBoostError::InvalidInput(
            "rank portfolio rank probability score must be finite".to_string(),
        ));
    }
    Ok(-summary.portfolio.net_return + rps_tiebreak_weight * summary.mean_rps)
}
