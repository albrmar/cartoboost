use crate::forecasting::{ForecastFrame, ForecastPrediction, ForecastResult, Forecaster};
use crate::{CartoBoostError, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub struct WeightedEnsembleForecaster {
    members: Vec<WeightedMember>,
}

struct WeightedMember {
    name: String,
    weight: f64,
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

impl Forecaster for WeightedEnsembleForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        for member in &mut self.members {
            member.forecaster.fit(frame)?;
        }
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let mut weighted: BTreeMap<ForecastKey, f64> = BTreeMap::new();
        let mut expected_keys: Option<Vec<ForecastKey>> = None;
        for member in &self.members {
            let result = member.forecaster.predict(horizon)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::{
        ForecastFrequency, ForecastRow, NaiveForecaster, SeasonalNaiveForecaster,
    };
    use chrono::NaiveDate;

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

    fn ts(day: u32) -> chrono::NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 1, day)
            .expect("date")
            .and_hms_opt(0, 0, 0)
            .expect("time")
    }
}
