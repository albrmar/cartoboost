use crate::booster::BoosterConfig;
use crate::forecasting::lag_features::history_by_series;
use crate::forecasting::{
    CalendarFeature, ClassicalExpertBank, ExpertScore, ForecastActual, ForecastFrame,
    ForecastObjective, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
    IntermittentDemandConfig, IntermittentDemandForecaster, LagFeatureConfig, LagPlusConfig,
    LagPlusForecaster, LocalStandardScaledForecaster, Log1pForecaster,
    RectifiedRecursiveForecaster, RuleBasedGating, RuleBasedGatingGuardrails, ValidationScoreTable,
};
use crate::forecasting::{
    CartoBoostDirectForecaster, CartoBoostLagForecaster, GlobalForecastTargetMode,
};
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const LAG_EXPERT: &str = "cartoboost_lag";
const SCALED_LAG_EXPERT: &str = "scaled_lag";
const DELTA_LAG_EXPERT: &str = "delta_lag";
const SCALED_DELTA_LAG_EXPERT: &str = "scaled_delta_lag";
const DIRECT_EXPERT: &str = "cartoboost_direct";
const RECTIFIED_RECURSIVE_EXPERT: &str = "cartoboost_rectified_recursive";
const LOG1P_SCALED_LAG_EXPERT: &str = "log1p_scaled_lag";
const LAG_PLUS_EXPERT: &str = "lag_plus";
const INTERMITTENT_DEMAND_EXPERT: &str = "intermittent_demand";
const CLASSICAL_EXPERT: &str = "classical_expert_bank";

pub type AutoForecastObjective = ForecastObjective;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoForecastConfig {
    pub lag_config: LagFeatureConfig,
    pub booster_config: BoosterConfig,
    pub target_mode: GlobalForecastTargetMode,
    pub season_length: usize,
    pub validation_window: Option<usize>,
    pub objective: AutoForecastObjective,
    pub baseline_displacement_gain: f64,
    pub hard_winner_relative_gain: f64,
    pub min_blend_weight: f64,
    pub max_blend_weight: f64,
    pub max_direct_horizon: usize,
}

impl Default for AutoForecastConfig {
    fn default() -> Self {
        Self {
            lag_config: LagFeatureConfig {
                lags: vec![1, 2, 3, 7, 14, 28],
                rolling_mean_windows: vec![7, 14, 28],
                calendar_features: vec![
                    CalendarFeature::DayOfWeek,
                    CalendarFeature::Month,
                    CalendarFeature::Day,
                ],
                difference_lags: vec![2, 3, 7, 14, 28],
                rolling_trend_windows: vec![7, 14, 28],
            },
            booster_config: BoosterConfig::default(),
            target_mode: GlobalForecastTargetMode::Level,
            season_length: 7,
            validation_window: None,
            objective: ForecastObjective::Rmse,
            baseline_displacement_gain: 0.03,
            hard_winner_relative_gain: 0.05,
            min_blend_weight: 0.15,
            max_blend_weight: 0.85,
            max_direct_horizon: 28,
        }
    }
}

pub struct AutoForecastModel {
    config: AutoForecastConfig,
    fitted: Option<FittedAutoForecastModel>,
}

struct FittedAutoForecastModel {
    members: Vec<FittedMember>,
    weights: BTreeMap<String, f64>,
    horizon_weights: BTreeMap<usize, BTreeMap<String, f64>>,
    series_weights: BTreeMap<String, BTreeMap<String, f64>>,
    validation_scores: Vec<ExpertScore>,
    validation_window: usize,
    member_metadata: BTreeMap<String, Value>,
    nonnegative_output: bool,
}

struct FittedMember {
    name: String,
    forecaster: Box<dyn Forecaster>,
}

enum DirectAutoMember {
    Direct(Box<CartoBoostDirectForecaster>),
    Rectified(Box<RectifiedRecursiveForecaster>),
}

struct FixedHorizonDirectForecaster {
    member: DirectAutoMember,
    fit_horizon: usize,
    model_name: &'static str,
}

impl FixedHorizonDirectForecaster {
    fn new(member: DirectAutoMember, fit_horizon: usize, model_name: &'static str) -> Result<Self> {
        if fit_horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "auto direct fit horizon must be positive".to_string(),
            ));
        }
        Ok(Self {
            member,
            fit_horizon,
            model_name,
        })
    }
}

impl Forecaster for FixedHorizonDirectForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        match &mut self.member {
            DirectAutoMember::Direct(model) => model.fit_horizon(frame, self.fit_horizon),
            DirectAutoMember::Rectified(model) => model.fit_horizon(frame, self.fit_horizon),
        }
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        match &self.member {
            DirectAutoMember::Direct(model) => model.predict(horizon),
            DirectAutoMember::Rectified(model) => model.predict(horizon),
        }
    }

    fn model_name(&self) -> &'static str {
        self.model_name
    }

    fn metadata(&self) -> Value {
        let inner = match &self.member {
            DirectAutoMember::Direct(model) => model.metadata(),
            DirectAutoMember::Rectified(model) => model.metadata(),
        };
        json!({
            "model": self.model_name(),
            "fit_horizon": self.fit_horizon,
            "inner": inner,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ForecastKey {
    series_id: String,
    timestamp: chrono::NaiveDateTime,
    horizon: usize,
}

impl AutoForecastModel {
    pub fn new(config: AutoForecastConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            config,
            fitted: None,
        })
    }

    pub fn config(&self) -> &AutoForecastConfig {
        &self.config
    }

    pub fn weights(&self) -> Option<&BTreeMap<String, f64>> {
        self.fitted.as_ref().map(|state| &state.weights)
    }
}

impl Forecaster for AutoForecastModel {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let validation_window = effective_validation_window(frame, self.config.validation_window);
        let nonnegative_output = frame_is_nonnegative(frame.rows());
        let split = split_validation_frame(frame, validation_window)?;
        let mut scores = Vec::new();
        for name in [
            LAG_EXPERT,
            SCALED_LAG_EXPERT,
            DELTA_LAG_EXPERT,
            SCALED_DELTA_LAG_EXPERT,
            DIRECT_EXPERT,
            RECTIFIED_RECURSIVE_EXPERT,
            LOG1P_SCALED_LAG_EXPERT,
            LAG_PLUS_EXPERT,
            INTERMITTENT_DEMAND_EXPERT,
            CLASSICAL_EXPERT,
        ] {
            if !candidate_is_eligible(name, frame) {
                continue;
            }
            if let Ok(candidate_scores) =
                score_candidate(name, &self.config, &split, nonnegative_output)
            {
                scores.extend(candidate_scores);
            }
        }
        if scores.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "no auto forecast model candidate could be validated".to_string(),
            ));
        }
        let baseline = scores
            .iter()
            .any(|score| score.expert == LAG_EXPERT)
            .then(|| LAG_EXPERT.to_string());
        let gating = RuleBasedGating::with_guardrails(
            self.config.objective.as_str(),
            ValidationScoreTable::new(scores.clone())?,
            RuleBasedGatingGuardrails {
                top_k: Some(2),
                hard_winner_relative_gain: Some(self.config.hard_winner_relative_gain),
                weight_bounds: Some((self.config.min_blend_weight, self.config.max_blend_weight)),
                baseline,
                baseline_displacement_gain: Some(self.config.baseline_displacement_gain)
                    .filter(|_| scores.iter().any(|score| score.expert == LAG_EXPERT)),
                ..RuleBasedGatingGuardrails::default()
            },
        )?;
        let weights = gating.weights_for(None, None)?;
        let horizon_weights = weights_by_horizon(&gating, validation_window)?;
        let series_weights = weights_by_series(&gating, frame)?;
        let selected_names = selected_member_names(&weights, &horizon_weights, &series_weights);
        let mut members = Vec::new();
        let mut member_metadata = BTreeMap::new();
        for name in selected_names {
            let mut forecaster = build_candidate(&name, &self.config)?;
            forecaster.fit(frame)?;
            member_metadata.insert(name.clone(), forecaster.metadata());
            members.push(FittedMember {
                name: name.clone(),
                forecaster,
            });
        }
        self.fitted = Some(FittedAutoForecastModel {
            members,
            weights,
            horizon_weights,
            series_weights,
            validation_scores: scores,
            validation_window,
            member_metadata,
            nonnegative_output,
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            CartoBoostError::InvalidInput("auto forecast model must be fitted".to_string())
        })?;
        let mut weighted: BTreeMap<ForecastKey, f64> = BTreeMap::new();
        let mut expected_keys: Option<Vec<ForecastKey>> = None;
        for member in &fitted.members {
            let result = member.forecaster.predict(horizon)?;
            let mut current_keys = Vec::with_capacity(result.predictions().len());
            for prediction in result.predictions() {
                let weight = member_weight_for_prediction(
                    fitted,
                    &member.name,
                    &prediction.series_id,
                    prediction.horizon,
                );
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
                        "auto forecast member '{}' produced a mismatched forecast index",
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
                mean: maybe_clamp_nonnegative(mean, fitted.nonnegative_output),
            })
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "cartoboost_auto_forecast"
    }

    fn metadata(&self) -> Value {
        let fitted = self.fitted.as_ref();
        json!({
            "model": self.model_name(),
            "objective": self.config.objective.as_str(),
            "season_length": self.config.season_length,
            "validation_window": fitted.map(|state| state.validation_window),
            "baseline": LAG_EXPERT,
            "baseline_displacement_gain": self.config.baseline_displacement_gain,
            "hard_winner_relative_gain": self.config.hard_winner_relative_gain,
            "max_direct_horizon": self.config.max_direct_horizon,
            "nonnegative_output": fitted.map(|state| state.nonnegative_output),
            "blend_bounds": {
                "min": self.config.min_blend_weight,
                "max": self.config.max_blend_weight,
            },
            "weights": fitted.map(|state| &state.weights),
            "horizon_weights": fitted.map(|state| &state.horizon_weights),
            "series_weights": fitted.map(|state| &state.series_weights),
            "validation_scores": fitted.map(|state| &state.validation_scores),
            "members": fitted.map(|state| &state.member_metadata),
        })
    }
}

struct ValidationSplit {
    train: ForecastFrame,
    validation: Vec<ForecastRow>,
}

fn build_candidate(name: &str, config: &AutoForecastConfig) -> Result<Box<dyn Forecaster>> {
    build_candidate_with_direct_horizon(name, config, config.max_direct_horizon)
}

fn build_candidate_with_direct_horizon(
    name: &str,
    config: &AutoForecastConfig,
    direct_horizon: usize,
) -> Result<Box<dyn Forecaster>> {
    match name {
        LAG_EXPERT => Ok(Box::new(CartoBoostLagForecaster::new_with_target_mode(
            config.lag_config.clone(),
            config.booster_config.clone(),
            config.target_mode,
        )?)),
        SCALED_LAG_EXPERT => Ok(Box::new(LocalStandardScaledForecaster::new(
            Box::new(CartoBoostLagForecaster::new_with_target_mode(
                config.lag_config.clone(),
                config.booster_config.clone(),
                config.target_mode,
            )?),
            1e-6,
            SCALED_LAG_EXPERT,
        )?)),
        DELTA_LAG_EXPERT => Ok(Box::new(CartoBoostLagForecaster::new_with_target_mode(
            config.lag_config.clone(),
            config.booster_config.clone(),
            GlobalForecastTargetMode::DeltaFromLast,
        )?)),
        SCALED_DELTA_LAG_EXPERT => Ok(Box::new(LocalStandardScaledForecaster::new(
            Box::new(CartoBoostLagForecaster::new_with_target_mode(
                config.lag_config.clone(),
                config.booster_config.clone(),
                GlobalForecastTargetMode::DeltaFromLast,
            )?),
            1e-6,
            SCALED_DELTA_LAG_EXPERT,
        )?)),
        DIRECT_EXPERT => Ok(Box::new(FixedHorizonDirectForecaster::new(
            DirectAutoMember::Direct(Box::new(CartoBoostDirectForecaster::new(
                config.lag_config.clone(),
                config.booster_config.clone(),
            )?)),
            direct_horizon,
            DIRECT_EXPERT,
        )?)),
        RECTIFIED_RECURSIVE_EXPERT => Ok(Box::new(FixedHorizonDirectForecaster::new(
            DirectAutoMember::Rectified(Box::new(RectifiedRecursiveForecaster::new(
                config.lag_config.clone(),
                config.booster_config.clone(),
            )?)),
            direct_horizon,
            RECTIFIED_RECURSIVE_EXPERT,
        )?)),
        LOG1P_SCALED_LAG_EXPERT => Ok(Box::new(Log1pForecaster::new(
            Box::new(LocalStandardScaledForecaster::new(
                Box::new(CartoBoostLagForecaster::new_with_target_mode(
                    config.lag_config.clone(),
                    config.booster_config.clone(),
                    config.target_mode,
                )?),
                1e-6,
                "log1p_scaled_lag_transformed",
            )?),
            LOG1P_SCALED_LAG_EXPERT,
        ))),
        LAG_PLUS_EXPERT => Ok(Box::new(LagPlusForecaster::new(LagPlusConfig {
            lag_config: config.lag_config.clone(),
            booster_config: config.booster_config.clone(),
            target_mode: config.target_mode,
            validation_window: config.validation_window,
            objective: config.objective,
            shrinkage_strength: 4.0,
            seasonal_bucket_period: Some(config.season_length),
        })?)),
        INTERMITTENT_DEMAND_EXPERT => Ok(Box::new(IntermittentDemandForecaster::new(
            IntermittentDemandConfig {
                validation_window: config.validation_window,
                objective: config.objective,
                ..IntermittentDemandConfig::default()
            },
        )?)),
        CLASSICAL_EXPERT => Ok(Box::new(ClassicalExpertBank::with_validation_window(
            ClassicalExpertBank::default_for_season_length(config.season_length)?
                .experts()
                .to_vec(),
            Some(config.validation_window.unwrap_or(1).max(1)),
        )?)),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown auto forecast candidate '{other}'"
        ))),
    }
}

fn candidate_is_eligible(name: &str, frame: &ForecastFrame) -> bool {
    match name {
        LOG1P_SCALED_LAG_EXPERT => frame_is_nonnegative(frame.rows()),
        INTERMITTENT_DEMAND_EXPERT => {
            frame_is_nonnegative(frame.rows()) && zero_fraction(frame.rows()) >= 0.25
        }
        DIRECT_EXPERT | RECTIFIED_RECURSIVE_EXPERT => zero_fraction(frame.rows()) < 0.25,
        _ => true,
    }
}

fn zero_fraction(rows: &[ForecastRow]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter().filter(|row| row.target == 0.0).count() as f64 / rows.len() as f64
}

fn frame_is_nonnegative(rows: &[ForecastRow]) -> bool {
    rows.iter().all(|row| row.target >= 0.0)
}

fn weights_by_series(
    gating: &RuleBasedGating,
    frame: &ForecastFrame,
) -> Result<BTreeMap<String, BTreeMap<String, f64>>> {
    let mut by_series = BTreeMap::new();
    if !frame.is_panel() {
        return Ok(by_series);
    }
    for series_id in frame.series_ids() {
        by_series.insert(
            series_id.clone(),
            gating.weights_for(Some(&series_id), None)?,
        );
    }
    Ok(by_series)
}

fn weights_by_horizon(
    gating: &RuleBasedGating,
    validation_window: usize,
) -> Result<BTreeMap<usize, BTreeMap<String, f64>>> {
    let mut by_horizon = BTreeMap::new();
    for horizon in 1..=validation_window {
        by_horizon.insert(horizon, gating.weights_for(None, Some(horizon))?);
    }
    Ok(by_horizon)
}

fn selected_member_names(
    weights: &BTreeMap<String, f64>,
    horizon_weights: &BTreeMap<usize, BTreeMap<String, f64>>,
    series_weights: &BTreeMap<String, BTreeMap<String, f64>>,
) -> Vec<String> {
    let mut selected = BTreeSet::new();
    for name in weights.keys() {
        selected.insert(name.clone());
    }
    for weights in horizon_weights.values() {
        for name in weights.keys() {
            selected.insert(name.clone());
        }
    }
    for weights in series_weights.values() {
        for name in weights.keys() {
            selected.insert(name.clone());
        }
    }
    selected.into_iter().collect()
}

fn member_weight_for_prediction(
    fitted: &FittedAutoForecastModel,
    member_name: &str,
    series_id: &str,
    horizon: usize,
) -> f64 {
    if let Some(weights) = fitted.series_weights.get(series_id) {
        return weights.get(member_name).copied().unwrap_or(0.0);
    }
    if let Some(weights) = fitted.horizon_weights.get(&horizon) {
        return weights.get(member_name).copied().unwrap_or(0.0);
    }
    fitted.weights.get(member_name).copied().unwrap_or(0.0)
}

fn score_candidate(
    name: &str,
    config: &AutoForecastConfig,
    split: &ValidationSplit,
    nonnegative_output: bool,
) -> Result<Vec<ExpertScore>> {
    let horizon = validation_horizon(&split.validation);
    let mut candidate = build_candidate_with_direct_horizon(name, config, horizon)?;
    candidate.fit(&split.train)?;
    let predictions = maybe_clamp_result(candidate.predict(horizon)?, nonnegative_output)?;
    let actuals = validation_actuals(&split.validation);
    let metrics = crate::forecasting::evaluate_forecast(&predictions, &actuals)?;
    let mut scores = vec![ExpertScore::global(
        name,
        config.objective.as_str(),
        config.objective.metric_value(&metrics),
    )];
    scores.extend(horizon_scores(
        name,
        config.objective,
        &predictions,
        &actuals,
    )?);
    scores.extend(series_scores(
        name,
        config.objective,
        &predictions,
        &actuals,
    )?);
    Ok(scores)
}

fn horizon_scores(
    name: &str,
    objective: ForecastObjective,
    predictions: &ForecastResult,
    actuals: &[ForecastActual],
) -> Result<Vec<ExpertScore>> {
    let mut actuals_by_horizon: BTreeMap<usize, Vec<ForecastActual>> = BTreeMap::new();
    for actual in actuals {
        actuals_by_horizon
            .entry(actual.horizon)
            .or_default()
            .push(actual.clone());
    }
    let mut predictions_by_horizon: BTreeMap<usize, Vec<ForecastPrediction>> = BTreeMap::new();
    for prediction in predictions.predictions() {
        predictions_by_horizon
            .entry(prediction.horizon)
            .or_default()
            .push(prediction.clone());
    }
    let mut scores = Vec::with_capacity(actuals_by_horizon.len());
    for (horizon, horizon_actuals) in actuals_by_horizon {
        let Some(horizon_predictions) = predictions_by_horizon.get(&horizon) else {
            continue;
        };
        let metrics = crate::forecasting::evaluate_forecast(
            &ForecastResult::new(horizon_predictions.clone())?,
            &horizon_actuals,
        )?;
        scores.push(ExpertScore::for_horizon(
            name,
            horizon,
            objective.as_str(),
            objective.metric_value(&metrics),
        ));
    }
    Ok(scores)
}

fn maybe_clamp_result(result: ForecastResult, nonnegative_output: bool) -> Result<ForecastResult> {
    if !nonnegative_output {
        return Ok(result);
    }
    let predictions = result
        .predictions()
        .iter()
        .map(|prediction| ForecastPrediction {
            series_id: prediction.series_id.clone(),
            timestamp: prediction.timestamp,
            horizon: prediction.horizon,
            model: prediction.model.clone(),
            mean: maybe_clamp_nonnegative(prediction.mean, true),
        })
        .collect();
    ForecastResult::new(predictions)
}

fn maybe_clamp_nonnegative(value: f64, nonnegative_output: bool) -> f64 {
    if nonnegative_output {
        value.max(0.0)
    } else {
        value
    }
}

fn series_scores(
    name: &str,
    objective: ForecastObjective,
    predictions: &ForecastResult,
    actuals: &[ForecastActual],
) -> Result<Vec<ExpertScore>> {
    let mut actuals_by_series: BTreeMap<String, Vec<ForecastActual>> = BTreeMap::new();
    for actual in actuals {
        actuals_by_series
            .entry(actual.series_id.clone())
            .or_default()
            .push(actual.clone());
    }
    if actuals_by_series.len() <= 1 {
        return Ok(Vec::new());
    }
    let mut predictions_by_series: BTreeMap<String, Vec<ForecastPrediction>> = BTreeMap::new();
    for prediction in predictions.predictions() {
        predictions_by_series
            .entry(prediction.series_id.clone())
            .or_default()
            .push(prediction.clone());
    }
    let mut scores = Vec::with_capacity(actuals_by_series.len());
    for (series_id, series_actuals) in actuals_by_series {
        let Some(series_predictions) = predictions_by_series.get(&series_id) else {
            continue;
        };
        let metrics = crate::forecasting::evaluate_forecast(
            &ForecastResult::new(series_predictions.clone())?,
            &series_actuals,
        )?;
        scores.push(ExpertScore::for_series(
            name,
            series_id,
            objective.as_str(),
            objective.metric_value(&metrics),
        ));
    }
    Ok(scores)
}

fn split_validation_frame(
    frame: &ForecastFrame,
    validation_window: usize,
) -> Result<ValidationSplit> {
    let mut train_rows = Vec::new();
    let mut validation_rows = Vec::new();
    for (_, rows) in history_by_series(frame.rows()) {
        if rows.len() <= validation_window {
            return Err(CartoBoostError::InvalidInput(
                "not enough history for auto forecast validation".to_string(),
            ));
        }
        let split_at = rows.len() - validation_window;
        train_rows.extend(rows[..split_at].iter().cloned());
        validation_rows.extend(rows[split_at..].iter().cloned());
    }
    Ok(ValidationSplit {
        train: ForecastFrame::with_metadata(
            train_rows,
            frame.frequency(),
            frame.metadata().clone(),
        )?,
        validation: validation_rows,
    })
}

fn validation_actuals(validation: &[ForecastRow]) -> Vec<ForecastActual> {
    let mut actuals = Vec::with_capacity(validation.len());
    for (_, rows) in history_by_series(validation) {
        for (index, row) in rows.iter().enumerate() {
            actuals.push(ForecastActual {
                series_id: row.series_id.clone(),
                timestamp: row.timestamp,
                horizon: index + 1,
                actual: row.target,
            });
        }
    }
    actuals
}

fn validation_horizon(validation: &[ForecastRow]) -> usize {
    history_by_series(validation)
        .values()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
}

fn effective_validation_window(frame: &ForecastFrame, configured: Option<usize>) -> usize {
    if let Some(window) = configured {
        return window;
    }
    let min_history = history_by_series(frame.rows())
        .values()
        .map(Vec::len)
        .min()
        .unwrap_or(0);
    (min_history / 5).clamp(1, 8)
}

fn validate_config(config: &AutoForecastConfig) -> Result<()> {
    if config.season_length == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast season_length must be positive".to_string(),
        ));
    }
    if matches!(config.validation_window, Some(0)) {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast validation_window must be positive".to_string(),
        ));
    }
    if config.max_direct_horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast max_direct_horizon must be positive".to_string(),
        ));
    }
    if !config.baseline_displacement_gain.is_finite()
        || !(0.0..=1.0).contains(&config.baseline_displacement_gain)
    {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast baseline_displacement_gain must be between 0 and 1".to_string(),
        ));
    }
    if !config.hard_winner_relative_gain.is_finite()
        || !(0.0..=1.0).contains(&config.hard_winner_relative_gain)
    {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast hard_winner_relative_gain must be between 0 and 1".to_string(),
        ));
    }
    if !config.min_blend_weight.is_finite()
        || !config.max_blend_weight.is_finite()
        || config.min_blend_weight < 0.0
        || config.max_blend_weight > 1.0
        || config.min_blend_weight > config.max_blend_weight
    {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast blend weights must satisfy 0 <= min <= max <= 1".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_weights_are_authoritative_when_present() {
        let fitted = FittedAutoForecastModel {
            members: Vec::new(),
            weights: BTreeMap::from([
                ("cartoboost_lag".to_string(), 0.6),
                ("cartoboost_direct".to_string(), 0.4),
            ]),
            horizon_weights: BTreeMap::from([(
                2,
                BTreeMap::from([("cartoboost_direct".to_string(), 1.0)]),
            )]),
            series_weights: BTreeMap::from([(
                "pickup_zone_1".to_string(),
                BTreeMap::from([("cartoboost_lag".to_string(), 1.0)]),
            )]),
            validation_scores: Vec::new(),
            validation_window: 2,
            member_metadata: BTreeMap::new(),
            nonnegative_output: true,
        };

        assert_eq!(
            member_weight_for_prediction(&fitted, "cartoboost_lag", "pickup_zone_1", 2),
            1.0
        );
        assert_eq!(
            member_weight_for_prediction(&fitted, "cartoboost_direct", "pickup_zone_1", 2),
            0.0
        );
        assert_eq!(
            member_weight_for_prediction(&fitted, "cartoboost_direct", "unknown_zone", 2),
            1.0
        );
        assert_eq!(
            member_weight_for_prediction(&fitted, "cartoboost_direct", "unknown_zone", 9),
            0.4
        );
    }
}
