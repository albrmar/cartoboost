use crate::booster::BoosterConfig;
use crate::forecasting::{
    ArimaForecaster, AutoARIMAForecaster, AutoKalmanForecaster, AutoLocalLevelKalmanForecaster,
    CartoBoostLagForecaster, ETSForecaster, Forecaster, KalmanForecaster, LagFeatureConfig,
    LocalLevelKalmanForecaster, NaiveForecaster, OptimizedThetaForecaster, SeasonalNaiveForecaster,
    ThetaForecaster,
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
        let mut theta_params = Map::new();
        theta_params.insert("theta".to_string(), Value::from(2.0));
        theta_params.insert("alpha".to_string(), Value::from(0.5));
        registry.register(
            ForecastModelSpec::new("theta")?.with_params(theta_params),
            create_theta,
            RegisterMode::RejectDuplicate,
        )?;
        let mut optimized_theta_params = Map::new();
        optimized_theta_params.insert("theta_grid".to_string(), serde_json::json!([1.0, 2.0]));
        optimized_theta_params.insert("alpha_grid".to_string(), serde_json::json!([0.2, 0.5, 0.8]));
        registry.register(
            ForecastModelSpec::new("optimized_theta")?.with_params(optimized_theta_params),
            create_optimized_theta,
            RegisterMode::RejectDuplicate,
        )?;
        let mut ets_params = Map::new();
        ets_params.insert("alpha".to_string(), Value::from(0.5));
        ets_params.insert("beta".to_string(), Value::from(0.1));
        registry.register(
            ForecastModelSpec::new("ets")?.with_params(ets_params),
            create_ets,
            RegisterMode::RejectDuplicate,
        )?;
        let mut arima_params = Map::new();
        arima_params.insert("p".to_string(), Value::from(1_u64));
        arima_params.insert("d".to_string(), Value::from(0_u64));
        arima_params.insert("q".to_string(), Value::from(0_u64));
        registry.register(
            ForecastModelSpec::new("arima")?.with_params(arima_params),
            create_arima,
            RegisterMode::RejectDuplicate,
        )?;
        let mut auto_arima_params = Map::new();
        auto_arima_params.insert("max_p".to_string(), Value::from(3_u64));
        auto_arima_params.insert("max_d".to_string(), Value::from(1_u64));
        auto_arima_params.insert("max_q".to_string(), Value::from(2_u64));
        registry.register(
            ForecastModelSpec::new("auto_arima")?.with_params(auto_arima_params),
            create_auto_arima,
            RegisterMode::RejectDuplicate,
        )?;
        let mut kalman_params = Map::new();
        kalman_params.insert("level_process_variance".to_string(), Value::from(0.05));
        kalman_params.insert("trend_process_variance".to_string(), Value::from(0.005));
        kalman_params.insert("observation_variance".to_string(), Value::from(1.0));
        registry.register(
            ForecastModelSpec::new("kalman")?.with_params(kalman_params),
            create_kalman,
            RegisterMode::RejectDuplicate,
        )?;
        let mut local_level_kalman_params = Map::new();
        local_level_kalman_params.insert("level_process_variance".to_string(), Value::from(0.05));
        local_level_kalman_params.insert("observation_variance".to_string(), Value::from(1.0));
        registry.register(
            ForecastModelSpec::new("local_level_kalman")?.with_params(local_level_kalman_params),
            create_local_level_kalman,
            RegisterMode::RejectDuplicate,
        )?;
        let mut auto_kalman_params = Map::new();
        auto_kalman_params.insert(
            "level_process_variance_grid".to_string(),
            serde_json::json!([0.001, 0.01, 0.05, 0.1]),
        );
        auto_kalman_params.insert(
            "trend_process_variance_grid".to_string(),
            serde_json::json!([0.0001, 0.001, 0.005, 0.01]),
        );
        auto_kalman_params.insert(
            "observation_variance_grid".to_string(),
            serde_json::json!([0.1, 0.5, 1.0, 2.0]),
        );
        registry.register(
            ForecastModelSpec::new("auto_kalman")?.with_params(auto_kalman_params),
            create_auto_kalman,
            RegisterMode::RejectDuplicate,
        )?;
        let mut auto_local_level_kalman_params = Map::new();
        auto_local_level_kalman_params.insert(
            "level_process_variance_grid".to_string(),
            serde_json::json!([0.001, 0.01, 0.05, 0.1]),
        );
        auto_local_level_kalman_params.insert(
            "observation_variance_grid".to_string(),
            serde_json::json!([0.1, 0.5, 1.0, 2.0]),
        );
        registry.register(
            ForecastModelSpec::new("auto_local_level_kalman")?
                .with_params(auto_local_level_kalman_params),
            create_auto_local_level_kalman,
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
        let mut spec = spec;
        let name = normalize_name(&spec.name)?;
        spec.name = name.clone();
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
        normalize_name(name)
            .map(|name| self.models.contains_key(&name))
            .unwrap_or(false)
    }

    pub fn get(&self, name: &str) -> Result<&RegisteredForecastModel> {
        let name = normalize_name(name)?;
        self.models.get(&name).ok_or_else(|| {
            let known = self.names().join(", ");
            CartoBoostError::InvalidInput(format!(
                "forecast model '{name}' is not registered; known models: {known}"
            ))
        })
    }

    pub fn create(&self, name: &str) -> Result<Box<dyn Forecaster>> {
        self.get(name)?.create()
    }

    pub fn create_from_spec(&self, spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
        let name = normalize_name(&spec.name)?;
        let registered = self.get(&name)?;
        let mut spec = spec.clone();
        spec.name = name;
        (registered.factory)(&spec)
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
    let season_length = required_usize_param(spec, "season_length")?;
    Ok(Box::new(SeasonalNaiveForecaster::new(season_length)?))
}

fn create_theta(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let theta = optional_f64_param(spec, "theta")?.unwrap_or(2.0);
    let alpha = optional_f64_param(spec, "alpha")?.unwrap_or(0.5);
    Ok(Box::new(ThetaForecaster::new(theta, alpha)?))
}

fn create_optimized_theta(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let theta_grid = optional_f64_vec_param(spec, "theta_grid")?.unwrap_or_else(|| vec![1.0, 2.0]);
    let alpha_grid =
        optional_f64_vec_param(spec, "alpha_grid")?.unwrap_or_else(|| vec![0.2, 0.5, 0.8]);
    Ok(Box::new(OptimizedThetaForecaster::new(
        theta_grid, alpha_grid,
    )?))
}

fn create_ets(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let alpha = optional_f64_param(spec, "alpha")?.unwrap_or(0.5);
    let beta = optional_f64_param(spec, "beta")?.unwrap_or(0.1);
    let gamma = optional_f64_param(spec, "gamma")?;
    let season_length = optional_usize_param(spec, "season_length")?;
    Ok(Box::new(ETSForecaster::with_additive_seasonality(
        alpha,
        beta,
        gamma,
        season_length,
    )?))
}

fn create_arima(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let p = optional_usize_param(spec, "p")?.unwrap_or(1);
    let d = optional_usize_param(spec, "d")?.unwrap_or(0);
    let q = optional_usize_param(spec, "q")?.unwrap_or(0);
    Ok(Box::new(ArimaForecaster::new(p, d, q)?))
}

fn create_auto_arima(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let max_p = optional_usize_param(spec, "max_p")?.unwrap_or(3);
    let max_d = optional_usize_param(spec, "max_d")?.unwrap_or(1);
    let max_q = optional_usize_param(spec, "max_q")?.unwrap_or(2);
    Ok(Box::new(AutoARIMAForecaster::with_max_order(
        max_p, max_d, max_q,
    )?))
}

fn create_kalman(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let level_process_variance =
        optional_f64_param(spec, "level_process_variance")?.unwrap_or(0.05);
    let trend_process_variance =
        optional_f64_param(spec, "trend_process_variance")?.unwrap_or(0.005);
    let observation_variance = optional_f64_param(spec, "observation_variance")?.unwrap_or(1.0);
    Ok(Box::new(KalmanForecaster::new(
        level_process_variance,
        trend_process_variance,
        observation_variance,
    )?))
}

fn create_local_level_kalman(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let level_process_variance =
        optional_f64_param(spec, "level_process_variance")?.unwrap_or(0.05);
    let observation_variance = optional_f64_param(spec, "observation_variance")?.unwrap_or(1.0);
    Ok(Box::new(LocalLevelKalmanForecaster::new(
        level_process_variance,
        observation_variance,
    )?))
}

fn create_auto_kalman(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let level_process_variance_grid = optional_f64_vec_param(spec, "level_process_variance_grid")?
        .unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]);
    let trend_process_variance_grid = optional_f64_vec_param(spec, "trend_process_variance_grid")?
        .unwrap_or_else(|| vec![0.0001, 0.001, 0.005, 0.01]);
    let observation_variance_grid = optional_f64_vec_param(spec, "observation_variance_grid")?
        .unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]);
    let validation_window = optional_usize_param(spec, "validation_window")?;
    Ok(Box::new(AutoKalmanForecaster::with_grids(
        level_process_variance_grid,
        trend_process_variance_grid,
        observation_variance_grid,
        validation_window,
    )?))
}

fn create_auto_local_level_kalman(spec: &ForecastModelSpec) -> Result<Box<dyn Forecaster>> {
    let level_process_variance_grid = optional_f64_vec_param(spec, "level_process_variance_grid")?
        .unwrap_or_else(|| vec![0.001, 0.01, 0.05, 0.1]);
    let observation_variance_grid = optional_f64_vec_param(spec, "observation_variance_grid")?
        .unwrap_or_else(|| vec![0.1, 0.5, 1.0, 2.0]);
    let validation_window = optional_usize_param(spec, "validation_window")?;
    Ok(Box::new(AutoLocalLevelKalmanForecaster::with_grids(
        level_process_variance_grid,
        observation_variance_grid,
        validation_window,
    )?))
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

fn required_usize_param(spec: &ForecastModelSpec, param: &str) -> Result<usize> {
    let value = spec
        .params
        .get(param)
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!("{} requires integer param '{param}'", spec.name))
        })?;
    usize::try_from(value).map_err(|_| {
        CartoBoostError::InvalidInput(format!("{param} is too large for this platform"))
    })
}

fn optional_f64_param(spec: &ForecastModelSpec, param: &str) -> Result<Option<f64>> {
    match spec.params.get(param) {
        Some(value) => value.as_f64().map(Some).ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "{} param '{param}' must be a finite number",
                spec.name
            ))
        }),
        None => Ok(None),
    }
}

fn optional_usize_param(spec: &ForecastModelSpec, param: &str) -> Result<Option<usize>> {
    let Some(value) = spec.params.get(param) else {
        return Ok(None);
    };
    let value = value.as_u64().ok_or_else(|| {
        CartoBoostError::InvalidInput(format!("{} param '{param}' must be an integer", spec.name))
    })?;
    usize::try_from(value).map(Some).map_err(|_| {
        CartoBoostError::InvalidInput(format!("{param} is too large for this platform"))
    })
}

fn optional_f64_vec_param(spec: &ForecastModelSpec, param: &str) -> Result<Option<Vec<f64>>> {
    let Some(value) = spec.params.get(param) else {
        return Ok(None);
    };
    let values = value.as_array().ok_or_else(|| {
        CartoBoostError::InvalidInput(format!("{} param '{param}' must be an array", spec.name))
    })?;
    let mut parsed = Vec::with_capacity(values.len());
    for value in values {
        let number = value.as_f64().ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "{} param '{param}' must contain only finite numbers",
                spec.name
            ))
        })?;
        parsed.push(number);
    }
    Ok(Some(parsed))
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
            vec![
                "arima",
                "auto_arima",
                "auto_kalman",
                "auto_local_level_kalman",
                "cartoboost_lag",
                "ets",
                "kalman",
                "local_level_kalman",
                "naive",
                "optimized_theta",
                "seasonal_naive",
                "theta"
            ]
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

    #[test]
    fn registry_constructs_theta_from_configured_spec() {
        let registry = ForecastRegistry::with_defaults().expect("defaults");
        let mut params = Map::new();
        params.insert("theta".to_string(), Value::from(1.5));
        params.insert("alpha".to_string(), Value::from(0.3));
        let spec = ForecastModelSpec::new(" theta ")
            .expect("spec")
            .with_params(params);

        let model = registry.create_from_spec(&spec).expect("theta model");

        assert_eq!(model.model_name(), "theta");
    }

    #[test]
    fn registry_rejects_unknown_model_names_clearly() {
        let registry = ForecastRegistry::with_defaults().expect("defaults");

        let err = match registry.create("foundation_model") {
            Ok(_) => panic!("unknown model should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("not registered; known models"));
    }
}
