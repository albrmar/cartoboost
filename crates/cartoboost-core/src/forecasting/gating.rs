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
        })
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
        if let Some(top_k) = self.top_k {
            candidates.truncate(top_k.min(candidates.len()));
        }
        let mut raw = BTreeMap::new();
        for score in candidates {
            let weight = 1.0 / score.value.max(self.error_floor);
            raw.insert(score.expert.clone(), weight);
        }
        normalize(raw)
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
