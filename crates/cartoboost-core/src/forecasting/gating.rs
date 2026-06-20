use crate::forecasting::ForecastFrame;
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertScore {
    pub expert: String,
    pub series_id: Option<String>,
    pub horizon: Option<usize>,
    pub metric: String,
    pub value: f64,
}

impl ExpertScore {
    pub fn global(expert: impl Into<String>, metric: impl Into<String>, value: f64) -> Self {
        Self {
            expert: expert.into(),
            series_id: None,
            horizon: None,
            metric: metric.into(),
            value,
        }
    }

    pub fn for_series(
        expert: impl Into<String>,
        series_id: impl Into<String>,
        metric: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            expert: expert.into(),
            series_id: Some(series_id.into()),
            horizon: None,
            metric: metric.into(),
            value,
        }
    }

    pub fn for_horizon(
        expert: impl Into<String>,
        horizon: usize,
        metric: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            expert: expert.into(),
            series_id: None,
            horizon: Some(horizon),
            metric: metric.into(),
            value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationScoreTable {
    scores: Vec<ExpertScore>,
}

impl ValidationScoreTable {
    pub fn new(scores: Vec<ExpertScore>) -> Result<Self> {
        if scores.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "validation score table requires at least one score".to_string(),
            ));
        }
        let mut seen = BTreeMap::new();
        for score in &scores {
            if score.expert.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "validation score experts must be non-empty".to_string(),
                ));
            }
            if score.metric.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "validation score metrics must be non-empty".to_string(),
                ));
            }
            if !score.value.is_finite() || score.value < 0.0 {
                return Err(CartoBoostError::InvalidInput(
                    "validation scores must be finite and non-negative".to_string(),
                ));
            }
            if let Some(horizon) = score.horizon {
                if horizon == 0 {
                    return Err(CartoBoostError::InvalidInput(
                        "validation score horizons must be positive".to_string(),
                    ));
                }
            }
            let key = (
                score.expert.clone(),
                score.series_id.clone(),
                score.horizon,
                score.metric.clone(),
            );
            if seen.insert(key, ()).is_some() {
                return Err(CartoBoostError::InvalidInput(
                    "duplicate validation score entry".to_string(),
                ));
            }
        }
        Ok(Self { scores })
    }

    pub fn scores(&self) -> &[ExpertScore] {
        &self.scores
    }

    pub fn experts(&self) -> Vec<String> {
        let mut experts = Vec::new();
        for score in &self.scores {
            if !experts.contains(&score.expert) {
                experts.push(score.expert.clone());
            }
        }
        experts
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleBasedGating {
    metric: String,
    score_table: ValidationScoreTable,
    error_floor: f64,
    top_k: Option<usize>,
    hard_winner_relative_gain: Option<f64>,
    min_weight: Option<f64>,
    max_weight: Option<f64>,
    baseline: Option<String>,
    baseline_displacement_gain: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleBasedGatingGuardrails {
    pub error_floor: f64,
    pub top_k: Option<usize>,
    pub hard_winner_relative_gain: Option<f64>,
    pub weight_bounds: Option<(f64, f64)>,
    pub baseline: Option<String>,
    pub baseline_displacement_gain: Option<f64>,
}

impl Default for RuleBasedGatingGuardrails {
    fn default() -> Self {
        Self {
            error_floor: 1e-9,
            top_k: None,
            hard_winner_relative_gain: None,
            weight_bounds: None,
            baseline: None,
            baseline_displacement_gain: None,
        }
    }
}

impl RuleBasedGating {
    pub fn new(metric: impl Into<String>, score_table: ValidationScoreTable) -> Result<Self> {
        Self::with_options(metric, score_table, 1e-9, None)
    }

    pub fn with_options(
        metric: impl Into<String>,
        score_table: ValidationScoreTable,
        error_floor: f64,
        top_k: Option<usize>,
    ) -> Result<Self> {
        let metric = metric.into();
        if metric.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "gating metric must be non-empty".to_string(),
            ));
        }
        if !error_floor.is_finite() || error_floor <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "gating error_floor must be finite and positive".to_string(),
            ));
        }
        if top_k == Some(0) {
            return Err(CartoBoostError::InvalidInput(
                "gating top_k must be positive when provided".to_string(),
            ));
        }
        Ok(Self {
            metric,
            score_table,
            error_floor,
            top_k,
            hard_winner_relative_gain: None,
            min_weight: None,
            max_weight: None,
            baseline: None,
            baseline_displacement_gain: None,
        })
    }

    pub fn with_guardrails(
        metric: impl Into<String>,
        score_table: ValidationScoreTable,
        guardrails: RuleBasedGatingGuardrails,
    ) -> Result<Self> {
        let mut gating = Self::with_options(
            metric,
            score_table,
            guardrails.error_floor,
            guardrails.top_k,
        )?;
        if let Some(gain) = guardrails.hard_winner_relative_gain {
            validate_relative_gain(gain, "hard_winner_relative_gain")?;
        }
        let (min_weight, max_weight) = match guardrails.weight_bounds {
            Some((min_weight, max_weight)) => {
                validate_weight_bounds(min_weight, max_weight)?;
                (Some(min_weight), Some(max_weight))
            }
            None => (None, None),
        };
        if let Some(baseline) = &guardrails.baseline {
            if baseline.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "gating baseline must be non-empty".to_string(),
                ));
            }
        }
        if let Some(gain) = guardrails.baseline_displacement_gain {
            validate_relative_gain(gain, "baseline_displacement_gain")?;
        }
        if guardrails.baseline.is_none() && guardrails.baseline_displacement_gain.is_some() {
            return Err(CartoBoostError::InvalidInput(
                "baseline_displacement_gain requires a baseline".to_string(),
            ));
        }
        gating.hard_winner_relative_gain = guardrails.hard_winner_relative_gain;
        gating.min_weight = min_weight;
        gating.max_weight = max_weight;
        gating.baseline = guardrails.baseline;
        gating.baseline_displacement_gain = guardrails.baseline_displacement_gain;
        Ok(gating)
    }

    pub fn weights_for(
        &self,
        series_id: Option<&str>,
        horizon: Option<usize>,
    ) -> Result<BTreeMap<String, f64>> {
        let mut candidates = self.matching_scores(series_id, horizon);
        if candidates.is_empty() && series_id.is_some() {
            candidates = self.matching_scores(None, horizon);
        }
        if candidates.is_empty() && horizon.is_some() {
            candidates = self.matching_scores(series_id, None);
        }
        if candidates.is_empty() {
            candidates = self.matching_scores(None, None);
        }
        if candidates.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "no validation scores found for gating metric '{}'",
                self.metric
            )));
        }
        candidates.sort_by(|left, right| {
            left.value
                .total_cmp(&right.value)
                .then_with(|| left.expert.cmp(&right.expert))
        });
        if let Some(weights) = self.baseline_guardrail(&candidates)? {
            return Ok(weights);
        }
        if let Some(weights) = self.hard_winner_guardrail(&candidates) {
            return Ok(weights);
        }
        if let Some(top_k) = self.top_k {
            candidates.truncate(top_k.min(candidates.len()));
        }
        let mut raw = BTreeMap::new();
        for score in candidates {
            let weight = 1.0 / score.value.max(self.error_floor);
            raw.insert(score.expert.clone(), weight);
        }
        normalize_with_bounds(raw, self.min_weight, self.max_weight)
    }

    pub fn weights_for_frame(&self, frame: &ForecastFrame) -> Result<BTreeMap<String, f64>> {
        if frame.is_panel() {
            let mut totals: BTreeMap<String, f64> = BTreeMap::new();
            for series_id in frame.series_ids() {
                for (expert, weight) in self.weights_for(Some(&series_id), None)? {
                    *totals.entry(expert).or_insert(0.0) += weight;
                }
            }
            normalize(totals)
        } else {
            self.weights_for(None, None)
        }
    }

    pub fn metadata(&self) -> Value {
        json!({
            "metric": self.metric,
            "error_floor": self.error_floor,
            "top_k": self.top_k,
            "hard_winner_relative_gain": self.hard_winner_relative_gain,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "baseline": self.baseline,
            "baseline_displacement_gain": self.baseline_displacement_gain,
            "scores": self.score_table.scores(),
        })
    }

    fn matching_scores(
        &self,
        series_id: Option<&str>,
        horizon: Option<usize>,
    ) -> Vec<&ExpertScore> {
        self.score_table
            .scores()
            .iter()
            .filter(|score| score.metric == self.metric)
            .filter(|score| score.series_id.as_deref() == series_id)
            .filter(|score| score.horizon == horizon)
            .collect()
    }

    fn baseline_guardrail(
        &self,
        candidates: &[&ExpertScore],
    ) -> Result<Option<BTreeMap<String, f64>>> {
        let Some(baseline) = &self.baseline else {
            return Ok(None);
        };
        let Some(required_gain) = self.baseline_displacement_gain else {
            return Ok(None);
        };
        let baseline_score = candidates
            .iter()
            .find(|score| &score.expert == baseline)
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "gating baseline '{baseline}' has no validation score for metric '{}'",
                    self.metric
                ))
            })?;
        let best = candidates.first().ok_or_else(|| {
            CartoBoostError::InvalidInput("gating requires at least one candidate".to_string())
        })?;
        if best.expert == *baseline {
            return Ok(Some(single_weight(baseline)));
        }
        let displacement_gain = relative_gain(best.value, baseline_score.value);
        if displacement_gain < required_gain {
            return Ok(Some(single_weight(baseline)));
        }
        Ok(None)
    }

    fn hard_winner_guardrail(&self, candidates: &[&ExpertScore]) -> Option<BTreeMap<String, f64>> {
        let required_gain = self.hard_winner_relative_gain?;
        if candidates.len() < 2 {
            return Some(single_weight(&candidates[0].expert));
        }
        let best = candidates[0];
        let second = candidates[1];
        if relative_gain(best.value, second.value) >= required_gain {
            return Some(single_weight(&best.expert));
        }
        None
    }
}

fn normalize(raw: BTreeMap<String, f64>) -> Result<BTreeMap<String, f64>> {
    let total: f64 = raw.values().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "gating produced no positive expert weights".to_string(),
        ));
    }
    Ok(raw
        .into_iter()
        .map(|(expert, weight)| (expert, weight / total))
        .collect())
}

fn normalize_with_bounds(
    raw: BTreeMap<String, f64>,
    min_weight: Option<f64>,
    max_weight: Option<f64>,
) -> Result<BTreeMap<String, f64>> {
    let mut weights = normalize(raw)?;
    let Some(min_weight) = min_weight else {
        return Ok(weights);
    };
    let max_weight = max_weight.expect("weight bounds are set together");
    let n = weights.len();
    let selected_count = n as f64;
    if min_weight * selected_count > 1.0 + 1e-12 || max_weight * selected_count < 1.0 - 1e-12 {
        return Err(CartoBoostError::InvalidInput(
            "gating weight bounds cannot satisfy the number of selected experts".to_string(),
        ));
    }
    for weight in weights.values_mut() {
        *weight = weight.clamp(min_weight, max_weight);
    }
    normalize(weights)
}

fn single_weight(expert: &str) -> BTreeMap<String, f64> {
    BTreeMap::from([(expert.to_string(), 1.0)])
}

fn relative_gain(candidate_loss: f64, baseline_loss: f64) -> f64 {
    if baseline_loss <= 0.0 {
        if candidate_loss <= 0.0 {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    } else {
        ((baseline_loss - candidate_loss) / baseline_loss).max(0.0)
    }
}

fn validate_relative_gain(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(CartoBoostError::InvalidInput(format!(
            "gating {name} must be finite and between 0 and 1"
        )));
    }
    Ok(())
}

fn validate_weight_bounds(min_weight: f64, max_weight: f64) -> Result<()> {
    if !min_weight.is_finite()
        || !max_weight.is_finite()
        || min_weight < 0.0
        || max_weight <= 0.0
        || min_weight > max_weight
        || max_weight > 1.0
    {
        return Err(CartoBoostError::InvalidInput(
            "gating weight bounds must satisfy 0 <= min <= max <= 1".to_string(),
        ));
    }
    Ok(())
}
