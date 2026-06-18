use crate::booster::{Booster, BoosterConfig};
use crate::data::{Dataset, FeatureKind, FeatureSchema};
use crate::forecasting::lag_features::{history_by_series, LagFeatureBuilder, LagFeatureConfig};
use crate::forecasting::{
    ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use crate::tree::Model;
use crate::{CartoBoostError, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct CartoBoostLagForecaster {
    lag_builder: LagFeatureBuilder,
    booster_config: BoosterConfig,
    fitted: Option<FittedGlobalState>,
}

#[derive(Debug, Clone)]
struct FittedGlobalState {
    frame: ForecastFrame,
    history_by_series: BTreeMap<String, Vec<ForecastRow>>,
    model: Model,
    training_rows: usize,
}

impl CartoBoostLagForecaster {
    pub fn new(lag_config: LagFeatureConfig, booster_config: BoosterConfig) -> Result<Self> {
        Ok(Self {
            lag_builder: LagFeatureBuilder::new(lag_config)?,
            booster_config,
            fitted: None,
        })
    }

    pub fn lag_builder(&self) -> &LagFeatureBuilder {
        &self.lag_builder
    }

    pub fn booster_config(&self) -> &BoosterConfig {
        &self.booster_config
    }

    pub fn model(&self) -> Option<&Model> {
        self.fitted.as_ref().map(|state| &state.model)
    }

    pub fn training_rows(&self) -> Option<usize> {
        self.fitted.as_ref().map(|state| state.training_rows)
    }
}

impl Forecaster for CartoBoostLagForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let feature_rows = self.lag_builder.transform_frame(frame)?;
        if feature_rows.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "not enough history to build lag training rows".to_string(),
            ));
        }
        let feature_count = self.lag_builder.feature_names().len();
        let x = Dataset::from_rows(
            feature_rows
                .iter()
                .map(|row| row.features.clone())
                .collect::<Vec<_>>(),
        )?
        .with_schema(FeatureSchema {
            names: self.lag_builder.feature_names().to_vec(),
            kinds: vec![FeatureKind::Numeric; feature_count],
        })?;
        let y = feature_rows
            .iter()
            .map(|row| row.target)
            .collect::<Vec<_>>();
        let model = Booster::new(self.booster_config.clone()).fit(&x, &y, None)?;
        self.fitted = Some(FittedGlobalState {
            frame: frame.clone(),
            history_by_series: history_by_series(frame.rows()),
            model,
            training_rows: feature_rows.len(),
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mut predictions = Vec::new();
        for (series_id, fitted_history) in &fitted.history_by_series {
            let mut history = fitted_history.clone();
            let last = history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?
                .clone();
            for step in 1..=horizon {
                let timestamp = fitted.frame.frequency().advance(last.timestamp, step)?;
                let features = self
                    .lag_builder
                    .transform_next(series_id, &history, timestamp)?;
                let mean = fitted.model.predict_one(&features);
                predictions.push(ForecastPrediction {
                    series_id: series_id.clone(),
                    timestamp,
                    horizon: step,
                    model: self.model_name().to_string(),
                    mean,
                });
                history.push(ForecastRow::new(series_id.clone(), timestamp, mean));
            }
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "cartoboost_lag"
    }

    fn metadata(&self) -> Value {
        let mut payload = json!({
            "model": self.model_name(),
            "feature_names": self.lag_builder.feature_names(),
            "lag_config": self.lag_builder.config(),
            "booster_config": self.booster_config,
        });
        if let Some(fitted) = &self.fitted {
            payload["training_rows"] = json!(fitted.training_rows);
            payload["series_count"] = json!(fitted.history_by_series.len());
            payload["native_model_metadata"] = json!(fitted.model.metadata);
        }
        payload
    }
}

fn validate_horizon(horizon: usize) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    Ok(())
}

fn not_fitted() -> CartoBoostError {
    CartoBoostError::InvalidInput("forecaster must be fitted before predict".to_string())
}
