use crate::forecasting::{ForecastModelSpec, ForecastRegistry};
use crate::{CartoBoostError, Result};
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastingConfig {
    pub horizon: usize,
    #[serde(default)]
    pub frequency: Option<String>,
    #[serde(default)]
    pub target_column: Option<String>,
    #[serde(default)]
    pub time_column: Option<String>,
    #[serde(default)]
    pub series_id_column: Option<String>,
    #[serde(default)]
    pub models: Vec<ForecastModelConfig>,
    #[serde(default)]
    pub feature_config: Map<String, Value>,
    #[serde(default)]
    pub artifact: Map<String, Value>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastModelConfig {
    pub name: String,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl ForecastingConfig {
    pub fn from_toml_str(input: &str) -> Result<Self> {
        let config: Self = toml::from_str(input).map_err(|err| {
            CartoBoostError::InvalidInput(format!("invalid forecasting TOML config: {err}"))
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecasting config horizon must be positive".to_string(),
            ));
        }
        if self.models.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "forecasting config requires at least one model".to_string(),
            ));
        }
        let registry = ForecastRegistry::with_defaults()?;
        for model in &self.models {
            if !registry.contains(&model.name) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "forecasting config references unimplemented Rust model '{}'",
                    model.name
                )));
            }
        }
        Ok(())
    }

    pub fn model_specs(&self) -> Result<Vec<ForecastModelSpec>> {
        self.models
            .iter()
            .map(|model| Ok(ForecastModelSpec::new(&model.name)?.with_params(model.params.clone())))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strict_toml_for_implemented_models() {
        let config = ForecastingConfig::from_toml_str(
            r#"
            horizon = 2
            frequency = "1d"
            target_column = "pickup_demand"

            [[models]]
            name = "seasonal_naive"

            [models.params]
            season_length = 7
            "#,
        )
        .expect("config");

        assert_eq!(config.horizon, 2);
        assert_eq!(config.models[0].params["season_length"], Value::from(7));
    }

    #[test]
    fn rejects_unknown_config_fields() {
        let err = ForecastingConfig::from_toml_str(
            r#"
            horizon = 2
            untracked = true

            [[models]]
            name = "naive"
            "#,
        )
        .expect_err("unknown field");

        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn rejects_unimplemented_model_names() {
        let err = ForecastingConfig::from_toml_str(
            r#"
            horizon = 2

            [[models]]
            name = "auto_arima"
            "#,
        )
        .expect_err("unimplemented model");

        assert!(err.to_string().contains("unimplemented Rust model"));
    }
}
