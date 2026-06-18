use crate::booster::BoosterConfig;
use crate::forecasting::{
    CartoBoostLagForecaster, Forecaster, LagFeatureConfig, NaiveForecaster, SeasonalNaiveForecaster,
};
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub type ForecastFactory = fn(&ForecastModelSpec) -> Result<Box<dyn Forecaster>>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastModelSpec {
    pub name: String,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone)]
pub struct RegisteredForecastModel {
    spec: ForecastModelSpec,
    factory: ForecastFactory,
}

#[derive(Clone, Default)]
pub struct ForecastRegistry {
    models: BTreeMap<String, RegisteredForecastModel>,
}

impl ForecastModelSpec {
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = normalize_name(name.into())?;
        Ok(Self {
            name,
            params: Map::new(),
            metadata: Map::new(),
        })
    }

    pub fn with_params(mut self, params: Map<String, Value>) -> Self {
        self.params = params;
        self
    }
}

impl RegisteredForecastModel {
    pub fn new(spec: ForecastModelSpec, factory: ForecastFactory) -> Self {
        Self { spec, factory }
    }

    pub fn spec(&self) -> &ForecastModelSpec {
        &self.spec
    }

    pub fn create(&self) -> Result<Box<dyn Forecaster>> {
        (self.factory)(&self.spec)
    }
}

impl ForecastRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_defaults() -> Result<Self> {
        let mut registry = Self::new();
        registry.register(
            ForecastModelSpec::new("naive")?,
            create_naive,
            RegisterMode::RejectDuplicate,
        )?;
        let mut seasonal_params = Map::new();
        seasonal_params.insert("season_length".to_string(), Value::from(1_u64));
        registry.register(
            ForecastModelSpec::new("seasonal_naive")?.with_params(seasonal_params),
            create_seasonal_naive,
            RegisterMode::RejectDuplicate,
        )?;
        let mut lag_params = Map::new();
        lag_params.insert(
            "lag_config".to_string(),
            serde_json::to_value(LagFeatureConfig::default())?,
        );
        lag_params.insert(
            "booster_config".to_string(),
            serde_json::to_value(BoosterConfig::default())?,
        );
        registry.register(
            ForecastModelSpec::new("cartoboost_lag")?.with_params(lag_params),
            create_cartoboost_lag,
            RegisterMode::RejectDuplicate,
        )?;
        Ok(registry)
    }

    pub fn register(
        &mut self,
        spec: ForecastModelSpec,
        factory: ForecastFactory,
        mode: RegisterMode,
    ) -> Result<()> {
        let name = normalize_name(&spec.name)?;
        if self.models.contains_key(&name) && mode == RegisterMode::RejectDuplicate {
            return Err(CartoBoostError::InvalidInput(format!(
                "forecast model '{name}' is already registered"
            )));
        }
        self.models
            .insert(name, RegisteredForecastModel::new(spec, factory));
        Ok(())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.models.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<&RegisteredForecastModel> {
        self.models.get(name).ok_or_else(|| {
            let known = self.names().join(", ");
            CartoBoostError::InvalidInput(format!(
                "forecast model '{name}' is not registered; known models: {known}"
            ))
        })
    }

    pub fn create(&self, name: &str) -> Result<Box<dyn Forecaster>> {
        self.get(name)?.create()
    }

    pub fn names(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterMode {
    RejectDuplicate,
    Override,
}

fn create_naive(_: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    Ok(Box::new(NaiveForecaster::new()))
}

fn create_seasonal_naive(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let season_length = spec
        .params
        .get("season_length")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "seasonal_naive requires integer param 'season_length'".to_string(),
            )
        })?;
    let season_length = usize::try_from(season_length).map_err(|_| {
        CartoBoostError::InvalidInput("season_length is too large for this platform".to_string())
    })?;
    Ok(Box::new(SeasonalNaiveForecaster::new(season_length)?))
}

fn create_cartoboost_lag(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let lag_config = match spec.params.get("lag_config") {
        Some(value) => serde_json::from_value::<LagFeatureConfig>(value.clone())?,
        None => LagFeatureConfig::default(),
    };
    let booster_config = match spec.params.get("booster_config") {
        Some(value) => serde_json::from_value::<BoosterConfig>(value.clone())?,
        None => BoosterConfig::default(),
    };
    Ok(Box::new(CartoBoostLagForecaster::new(
        lag_config,
        booster_config,
    )?))
}

fn normalize_name(name: impl AsRef<str>) -> Result<String> {
    let name = name.as_ref().trim();
    if name.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "forecast model name must be non-empty".to_string(),
        ));
    }
    Ok(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_only_contains_implemented_rust_models() {
        let registry = ForecastRegistry::with_defaults().expect("defaults");

        assert_eq!(
            registry.names(),
            vec!["cartoboost_lag", "naive", "seasonal_naive"]
        );
    }

    #[test]
    fn registry_rejects_duplicates_unless_override() {
        let mut registry = ForecastRegistry::new();
        registry
            .register(
                ForecastModelSpec::new("naive").expect("spec"),
                create_naive,
                RegisterMode::RejectDuplicate,
            )
            .expect("first register");

        let duplicate = registry.register(
            ForecastModelSpec::new("naive").expect("spec"),
            create_naive,
            RegisterMode::RejectDuplicate,
        );

        assert!(duplicate.is_err());
        registry
            .register(
                ForecastModelSpec::new("naive").expect("spec"),
                create_naive,
                RegisterMode::Override,
            )
            .expect("override");
    }

    #[test]
    fn seasonal_naive_requires_season_length_param() {
        let spec = ForecastModelSpec::new("seasonal_naive").expect("spec");

        let err = create_seasonal_naive(&spec).err().expect("missing param");

        assert!(err
            .to_string()
            .contains("seasonal_naive requires integer param"));
    }

    #[test]
    fn registry_constructs_cartoboost_lag_from_default_params() {
        let registry = ForecastRegistry::with_defaults().expect("defaults");

        let model = registry.create("cartoboost_lag").expect("lag model");

        assert_eq!(model.model_name(), "cartoboost_lag");
    }
}
