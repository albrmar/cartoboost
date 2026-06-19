use crate::forecasting::{
    ForecastFrame, ForecastPrediction, ForecastResult, Forecaster, RuleBasedGating,
};
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub struct WeightedEnsembleForecaster {
    members: Vec<WeightedMember>,
}

pub type ForecastEnsemble = WeightedEnsembleForecaster;

struct WeightedMember {
    name: String,
    weight: f64,
    forecaster: Box<dyn Forecaster>,
}

pub struct GatedEnsembleForecaster {
    members: Vec<NamedMember>,
    gating: RuleBasedGating,
    weights: Option<BTreeMap<String, f64>>,
}

struct NamedMember {
    name: String,
    forecaster: Box<dyn Forecaster>,
}

#[derive(Debug, Clone, PartialEq)]
struct ForecastKey {
    series_id: String,
    timestamp: chrono::NaiveDateTime,
    horizon: usize,
}

impl Ord for ForecastKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.series_id
            .cmp(&other.series_id)
            .then_with(|| self.timestamp.cmp(&other.timestamp))
            .then_with(|| self.horizon.cmp(&other.horizon))
    }
}

impl PartialOrd for ForecastKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ForecastKey {}

impl WeightedEnsembleForecaster {
    pub fn new(members: Vec<(String, Box<dyn Forecaster>, f64)>) -> Result<Self> {
        if members.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "weighted ensemble requires at least one member".to_string(),
            ));
        }
        let mut total = 0.0;
        let mut cleaned = Vec::with_capacity(members.len());
        for (name, forecaster, weight) in members {
            if name.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "weighted ensemble member names must be non-empty".to_string(),
                ));
            }
            if cleaned
                .iter()
                .any(|member: &WeightedMember| member.name == name)
            {
                return Err(CartoBoostError::InvalidInput(format!(
                    "duplicate weighted ensemble member name '{name}'"
                )));
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(CartoBoostError::InvalidInput(
                    "weighted ensemble weights must be finite and non-negative".to_string(),
                ));
            }
            total += weight;
            cleaned.push(WeightedMember {
                name,
                weight,
                forecaster,
            });
        }
        if total <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "weighted ensemble requires at least one positive weight".to_string(),
            ));
        }
        for member in &mut cleaned {
            member.weight /= total;
        }
        Ok(Self { members: cleaned })
    }

    pub fn weights(&self) -> BTreeMap<String, f64> {
        self.members
            .iter()
            .map(|member| (member.name.clone(), member.weight))
            .collect()
    }
}

impl GatedEnsembleForecaster {
    pub fn new(
        members: Vec<(String, Box<dyn Forecaster>)>,
        gating: RuleBasedGating,
    ) -> Result<Self> {
        if members.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "gated ensemble requires at least one member".to_string(),
            ));
        }
        let mut cleaned = Vec::with_capacity(members.len());
        for (name, forecaster) in members {
            if name.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "gated ensemble member names must be non-empty".to_string(),
                ));
            }
            if cleaned
                .iter()
                .any(|member: &NamedMember| member.name == name)
            {
                return Err(CartoBoostError::InvalidInput(format!(
                    "duplicate gated ensemble member name '{name}'"
                )));
            }
            cleaned.push(NamedMember { name, forecaster });
        }
        Ok(Self {
            members: cleaned,
            gating,
            weights: None,
        })
    }

    pub fn weights(&self) -> Option<&BTreeMap<String, f64>> {
        self.weights.as_ref()
    }

    fn weighted_result(
        &self,
        weights: &BTreeMap<String, f64>,
        horizon: usize,
    ) -> Result<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        for member in &self.members {
            if !weights.contains_key(&member.name) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "gating weights missing ensemble member '{}'",
                    member.name
                )));
            }
        }
        for expert in weights.keys() {
            if !self.members.iter().any(|member| &member.name == expert) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "gating weights reference unknown ensemble member '{expert}'"
                )));
            }
        }

        let member_results = self
            .members
            .par_iter()
            .map(|member| member.forecaster.predict(horizon))
            .collect::<Result<Vec<_>>>()?;
        let mut weighted: BTreeMap<ForecastKey, f64> = BTreeMap::new();
        let mut expected_keys: Option<Vec<ForecastKey>> = None;
        for (member, result) in self.members.iter().zip(member_results) {
            let weight = weights[&member.name];
            let mut current_keys = Vec::with_capacity(result.predictions().len());
            for prediction in result.predictions() {
                let key = ForecastKey {
                    series_id: prediction.series_id.clone(),
                    timestamp: prediction.timestamp,
                    horizon: prediction.horizon,
                };
                current_keys.push(key.clone());
                *weighted.entry(key).or_insert(0.0) += weight * prediction.mean;
            }
            if let Some(expected) = &expected_keys {
                if expected != &current_keys {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "ensemble member '{}' produced a mismatched forecast index",
                        member.name
                    )));
                }
            } else {
                expected_keys = Some(current_keys);
            }
        }
        let predictions = weighted
            .into_iter()
            .map(|(key, mean)| ForecastPrediction {
                series_id: key.series_id,
                timestamp: key.timestamp,
                horizon: key.horizon,
                model: self.model_name().to_string(),
                mean,
            })
            .collect();
        ForecastResult::new(predictions)
    }
}

impl Forecaster for WeightedEnsembleForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.members
            .par_iter_mut()
            .map(|member| member.forecaster.fit(frame))
            .collect()
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let mut weighted: BTreeMap<ForecastKey, f64> = BTreeMap::new();
        let mut expected_keys: Option<Vec<ForecastKey>> = None;
        let member_results = self
            .members
            .par_iter()
            .map(|member| member.forecaster.predict(horizon))
            .collect::<Result<Vec<_>>>()?;
        for (member, result) in self.members.iter().zip(member_results) {
            let mut current_keys = Vec::with_capacity(result.predictions().len());
            for prediction in result.predictions() {
                let key = ForecastKey {
                    series_id: prediction.series_id.clone(),
                    timestamp: prediction.timestamp,
                    horizon: prediction.horizon,
                };
                current_keys.push(key.clone());
                *weighted.entry(key).or_insert(0.0) += member.weight * prediction.mean;
            }
            if let Some(expected) = &expected_keys {
                if expected != &current_keys {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "ensemble member '{}' produced a mismatched forecast index",
                        member.name
                    )));
                }
            } else {
                expected_keys = Some(current_keys);
            }
        }
        let predictions = weighted
            .into_iter()
            .map(|(key, mean)| ForecastPrediction {
                series_id: key.series_id,
                timestamp: key.timestamp,
                horizon: key.horizon,
                model: self.model_name().to_string(),
                mean,
            })
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "weighted_ensemble"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "weights": self.weights(),
        })
    }
}

impl Forecaster for GatedEnsembleForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.members
            .par_iter_mut()
            .map(|member| member.forecaster.fit(frame))
            .collect::<Result<Vec<_>>>()?;
        self.weights = Some(self.gating.weights_for_frame(frame)?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            CartoBoostError::InvalidInput("gated ensemble must be fit before predict".to_string())
        })?;
        self.weighted_result(weights, horizon)
    }

    fn model_name(&self) -> &'static str {
        "gated_ensemble"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "weights": self.weights,
            "gating": self.gating.metadata(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::{
        ForecastFrequency, ForecastRow, NaiveForecaster, SeasonalNaiveForecaster,
    };
    use chrono::NaiveDate;
    use serde_json::Value;

    struct FixedForecaster {
        predictions: Vec<ForecastPrediction>,
        name: &'static str,
    }

    impl Forecaster for FixedForecaster {
        fn fit(&mut self, _frame: &ForecastFrame) -> Result<()> {
            Ok(())
        }

        fn predict(&self, _horizon: usize) -> Result<ForecastResult> {
            ForecastResult::new(self.predictions.clone())
        }

        fn model_name(&self) -> &'static str {
            self.name
        }

        fn metadata(&self) -> Value {
            json!({"model": self.name})
        }
    }

    #[test]
    fn weighted_ensemble_averages_forecast_means() {
        let rows = vec![
            ForecastRow::single(ts(1), 10.0),
            ForecastRow::single(ts(2), 12.0),
            ForecastRow::single(ts(3), 14.0),
        ];
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("frame");
        let mut ensemble = WeightedEnsembleForecaster::new(vec![
            ("last".to_string(), Box::new(NaiveForecaster::new()), 1.0),
            (
                "seasonal".to_string(),
                Box::new(SeasonalNaiveForecaster::new(2).expect("seasonal")),
                3.0,
            ),
        ])
        .expect("ensemble");

        ensemble.fit(&frame).expect("fit");
        let result = ensemble.predict(2).expect("predict");
        let means: Vec<f64> = result
            .predictions()
            .iter()
            .map(|prediction| prediction.mean)
            .collect();

        assert_eq!(means, vec![12.5, 14.0]);
    }

    #[test]
    fn weighted_ensemble_rejects_invalid_weights() {
        let err = WeightedEnsembleForecaster::new(vec![(
            "last".to_string(),
            Box::new(NaiveForecaster::new()),
            0.0,
        )])
        .err()
        .expect("invalid weights");

        assert!(err.to_string().contains("at least one positive weight"));
    }

    #[test]
    fn weighted_ensemble_rejects_duplicate_member_names() {
        let err = WeightedEnsembleForecaster::new(vec![
            ("last".to_string(), Box::new(NaiveForecaster::new()), 1.0),
            ("last".to_string(), Box::new(NaiveForecaster::new()), 1.0),
        ])
        .err()
        .expect("duplicate name");

        assert!(err
            .to_string()
            .contains("duplicate weighted ensemble member name"));
    }

    #[test]
    fn weighted_ensemble_aligns_panel_forecasts() {
        let rows = vec![
            ForecastRow::new("PU1->DO2", ts(1), 10.0),
            ForecastRow::new("PU1->DO2", ts(2), 12.0),
            ForecastRow::new("PU1->DO2", ts(3), 14.0),
            ForecastRow::new("PU9->DO8", ts(1), 30.0),
            ForecastRow::new("PU9->DO8", ts(2), 28.0),
            ForecastRow::new("PU9->DO8", ts(3), 26.0),
        ];
        let frame = ForecastFrame::new(rows, ForecastFrequency::Daily).expect("frame");
        let mut ensemble = WeightedEnsembleForecaster::new(vec![
            ("last".to_string(), Box::new(NaiveForecaster::new()), 0.5),
            (
                "seasonal".to_string(),
                Box::new(SeasonalNaiveForecaster::new(2).expect("seasonal")),
                0.5,
            ),
        ])
        .expect("ensemble");

        ensemble.fit(&frame).expect("fit");
        let result = ensemble.predict(1).expect("predict");
        let predictions = result.predictions();

        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].series_id, "PU1->DO2");
        assert_eq!(predictions[0].mean, 13.0);
        assert_eq!(predictions[1].series_id, "PU9->DO8");
        assert_eq!(predictions[1].mean, 27.0);
    }

    #[test]
    fn weighted_ensemble_rejects_mismatched_forecast_index() {
        let first = FixedForecaster {
            name: "first",
            predictions: vec![prediction("PU1->DO2", 4, 1, 10.0)],
        };
        let second = FixedForecaster {
            name: "second",
            predictions: vec![prediction("PU9->DO8", 4, 1, 20.0)],
        };
        let ensemble = WeightedEnsembleForecaster::new(vec![
            ("first".to_string(), Box::new(first), 1.0),
            ("second".to_string(), Box::new(second), 1.0),
        ])
        .expect("ensemble");

        let err = ensemble.predict(1).expect_err("mismatched index");

        assert!(err.to_string().contains("mismatched forecast index"));
    }

    #[test]
    fn weighted_ensemble_metadata_exposes_normalized_weights() {
        let ensemble = WeightedEnsembleForecaster::new(vec![
            ("last".to_string(), Box::new(NaiveForecaster::new()), 1.0),
            (
                "seasonal".to_string(),
                Box::new(SeasonalNaiveForecaster::new(2).expect("seasonal")),
                3.0,
            ),
        ])
        .expect("ensemble");
        let metadata = ensemble.metadata();

        assert_eq!(metadata["model"], "weighted_ensemble");
        assert_eq!(metadata["weights"]["last"], 0.25);
        assert_eq!(metadata["weights"]["seasonal"], 0.75);
    }

    fn prediction(series_id: &str, day: u32, horizon: usize, mean: f64) -> ForecastPrediction {
        ForecastPrediction {
            series_id: series_id.to_string(),
            timestamp: ts(day),
            horizon,
            model: "fixed".to_string(),
            mean,
        }
    }

    fn ts(day: u32) -> chrono::NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 1, day)
            .expect("date")
            .and_hms_opt(0, 0, 0)
            .expect("time")
    }
}
