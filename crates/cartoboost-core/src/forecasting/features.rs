use super::lag_features::history_by_series;
use crate::forecasting::{ForecastFrame, LagFeatureBuilder, LagFeatureConfig};
use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectFeatureMatrix {
    pub feature_names: Vec<String>,
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
    pub series_ids: Vec<String>,
    pub origin_timestamps: Vec<NaiveDateTime>,
    pub target_timestamps: Vec<NaiveDateTime>,
    pub horizons: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastFeatureFactory {
    lag_builder: LagFeatureBuilder,
    max_horizon: usize,
    include_horizon_feature: bool,
}

impl DirectFeatureMatrix {
    pub fn row_count(&self) -> usize {
        self.targets.len()
    }

    pub fn feature_count(&self) -> usize {
        self.feature_names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    pub fn row_indices_for_horizon(&self, horizon: usize) -> Vec<usize> {
        self.horizons
            .iter()
            .enumerate()
            .filter_map(|(idx, row_horizon)| (*row_horizon == horizon).then_some(idx))
            .collect()
    }
}

impl ForecastFeatureFactory {
    pub fn new(lag_config: LagFeatureConfig, max_horizon: usize) -> Result<Self> {
        Self::from_lag_builder(LagFeatureBuilder::new(lag_config)?, max_horizon)
    }

    pub fn from_lag_builder(lag_builder: LagFeatureBuilder, max_horizon: usize) -> Result<Self> {
        if max_horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "direct feature max_horizon must be positive".to_string(),
            ));
        }
        Ok(Self {
            lag_builder,
            max_horizon,
            include_horizon_feature: true,
        })
    }

    pub fn with_horizon_feature(mut self, include_horizon_feature: bool) -> Self {
        self.include_horizon_feature = include_horizon_feature;
        self
    }

    pub fn lag_builder(&self) -> &LagFeatureBuilder {
        &self.lag_builder
    }

    pub fn max_horizon(&self) -> usize {
        self.max_horizon
    }

    pub fn build_direct_matrix(&self, frame: &ForecastFrame) -> Result<DirectFeatureMatrix> {
        let mut feature_names = self.lag_builder.feature_names().to_vec();
        if self.include_horizon_feature {
            feature_names.push("horizon".to_string());
        }

        let mut matrix = DirectFeatureMatrix {
            feature_names,
            features: Vec::new(),
            targets: Vec::new(),
            series_ids: Vec::new(),
            origin_timestamps: Vec::new(),
            target_timestamps: Vec::new(),
            horizons: Vec::new(),
        };

        for (series_id, history) in history_by_series(frame.rows()) {
            for origin_idx in 0..history.len() {
                let origin_timestamp = history[origin_idx].timestamp;
                let prior = &history[..=origin_idx];
                for horizon in 1..=self.max_horizon {
                    let target_idx = origin_idx + horizon;
                    if target_idx >= history.len() {
                        continue;
                    }
                    let target_row = &history[target_idx];
                    let expected_timestamp =
                        frame.frequency().advance(origin_timestamp, horizon)?;
                    if target_row.timestamp != expected_timestamp {
                        return Err(CartoBoostError::InvalidInput(format!(
                            "direct feature target timestamp mismatch for series {series_id}; expected {expected_timestamp}, observed {}",
                            target_row.timestamp
                        )));
                    }
                    let mut features = match self.lag_builder.transform_next_sorted_prior(
                        &series_id,
                        prior,
                        target_row.timestamp,
                    ) {
                        Ok(features) => features,
                        Err(CartoBoostError::InvalidInput(message))
                            if message.contains("does not have enough prior history") =>
                        {
                            continue;
                        }
                        Err(err) => return Err(err),
                    };
                    if self.include_horizon_feature {
                        features.push(horizon as f64);
                    }
                    matrix.features.push(features);
                    matrix.targets.push(target_row.target);
                    matrix.series_ids.push(series_id.clone());
                    matrix.origin_timestamps.push(origin_timestamp);
                    matrix.target_timestamps.push(target_row.timestamp);
                    matrix.horizons.push(horizon);
                }
            }
        }

        Ok(matrix)
    }
}
