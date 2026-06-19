use crate::forecasting::lag_features::history_by_series;
use crate::forecasting::local::{
    AutoARIMAForecaster, AutoKalmanForecaster, AutoLocalLevelKalmanForecaster, ETSForecaster,
    KalmanForecaster, LocalLevelKalmanForecaster, NaiveForecaster, SeasonalNaiveForecaster,
    ThetaForecaster,
};
use crate::forecasting::{
    ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use crate::{CartoBoostError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum ClassicalExpert {
    Naive,
    SeasonalNaive { season_length: usize },
    Theta { theta: f64, alpha: f64 },
    ETS { alpha: f64, beta: f64 },
    AutoARIMA { max_p: usize, max_d: usize },
    LocalLevelKalman,
    Kalman,
    AutoLocalLevelKalman,
    AutoKalman,
}

pub struct ClassicalExpertBank {
    experts: Vec<ClassicalExpert>,
    validation_window: Option<usize>,
    fitted: Option<FittedClassicalBankState>,
}

struct FittedClassicalBankState {
    selected: ClassicalExpert,
    fitted: Box<dyn Forecaster>,
    scores: Vec<ClassicalExpertScore>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassicalExpertScore {
    pub expert: ClassicalExpert,
    pub mse: f64,
    pub validation_rows: usize,
}

impl ClassicalExpert {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::SeasonalNaive { .. } => "seasonal_naive",
            Self::Theta { .. } => "theta",
            Self::ETS { .. } => "ets",
            Self::AutoARIMA { .. } => "auto_arima",
            Self::LocalLevelKalman => "local_level_kalman",
            Self::Kalman => "kalman",
            Self::AutoLocalLevelKalman => "auto_local_level_kalman",
            Self::AutoKalman => "auto_kalman",
        }
    }

    pub fn build(&self) -> Result<Box<dyn Forecaster>> {
        match *self {
            Self::Naive => Ok(Box::new(NaiveForecaster::new())),
            Self::SeasonalNaive { season_length } => {
                Ok(Box::new(SeasonalNaiveForecaster::new(season_length)?))
            }
            Self::Theta { theta, alpha } => Ok(Box::new(ThetaForecaster::new(theta, alpha)?)),
            Self::ETS { alpha, beta } => Ok(Box::new(ETSForecaster::new(alpha, beta)?)),
            Self::AutoARIMA { max_p, max_d } => {
                Ok(Box::new(AutoARIMAForecaster::new(max_p, max_d)?))
            }
            Self::LocalLevelKalman => Ok(Box::new(LocalLevelKalmanForecaster::default())),
            Self::Kalman => Ok(Box::new(KalmanForecaster::default())),
            Self::AutoLocalLevelKalman => Ok(Box::new(AutoLocalLevelKalmanForecaster::default())),
            Self::AutoKalman => Ok(Box::new(AutoKalmanForecaster::default())),
        }
    }

    pub fn metadata(&self) -> Value {
        match self {
            Self::Naive => json!({"model": self.name()}),
            Self::SeasonalNaive { season_length } => {
                json!({"model": self.name(), "season_length": season_length})
            }
            Self::Theta { theta, alpha } => {
                json!({"model": self.name(), "theta": theta, "alpha": alpha})
            }
            Self::ETS { alpha, beta } => {
                json!({"model": self.name(), "alpha": alpha, "beta": beta})
            }
            Self::AutoARIMA { max_p, max_d } => {
                json!({"model": self.name(), "max_p": max_p, "max_d": max_d})
            }
            Self::LocalLevelKalman
            | Self::Kalman
            | Self::AutoLocalLevelKalman
            | Self::AutoKalman => json!({"model": self.name()}),
        }
    }
}

impl ClassicalExpertBank {
    pub fn new(experts: Vec<ClassicalExpert>) -> Result<Self> {
        Self::with_validation_window(experts, None)
    }

    pub fn with_validation_window(
        experts: Vec<ClassicalExpert>,
        validation_window: Option<usize>,
    ) -> Result<Self> {
        if experts.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "classical expert bank requires at least one expert".to_string(),
            ));
        }
        if matches!(validation_window, Some(0)) {
            return Err(CartoBoostError::InvalidInput(
                "validation_window must be positive when provided".to_string(),
            ));
        }
        Ok(Self {
            experts,
            validation_window,
            fitted: None,
        })
    }

    pub fn default_for_season_length(season_length: usize) -> Result<Self> {
        Self::with_validation_window(default_experts(season_length), None)
    }

    pub fn experts(&self) -> &[ClassicalExpert] {
        &self.experts
    }

    pub fn selected_expert(&self) -> Option<&ClassicalExpert> {
        self.fitted.as_ref().map(|state| &state.selected)
    }

    pub fn validation_scores(&self) -> &[ClassicalExpertScore] {
        self.fitted
            .as_ref()
            .map(|state| state.scores.as_slice())
            .unwrap_or(&[])
    }
}

impl Forecaster for ClassicalExpertBank {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let validation_window = self.effective_validation_window(frame);
        let mut scores = Vec::new();
        if validation_window > 0 {
            let split = split_validation_frame(frame, validation_window)?;
            for expert in &self.experts {
                if let Ok(score) = score_expert(expert, &split.train, &split.validation) {
                    scores.push(score);
                }
            }
        }
        if scores.is_empty() {
            for expert in &self.experts {
                if let Ok(mut candidate) = expert.build() {
                    if candidate.fit(frame).is_ok() {
                        self.fitted = Some(FittedClassicalBankState {
                            selected: expert.clone(),
                            fitted: candidate,
                            scores,
                        });
                        return Ok(());
                    }
                }
            }
            return Err(CartoBoostError::InvalidInput(
                "no classical expert could fit the forecast frame".to_string(),
            ));
        }
        scores.sort_by_key(|score| {
            (
                OrderedF64(score.mse),
                expert_rank(&score.expert),
                score.expert.name(),
            )
        });
        let selected = scores[0].expert.clone();
        let mut fitted = selected.build()?;
        fitted.fit(frame)?;
        self.fitted = Some(FittedClassicalBankState {
            selected,
            fitted,
            scores,
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.fitted.predict(horizon)?;
        let predictions = result
            .predictions()
            .iter()
            .map(|prediction| ForecastPrediction {
                series_id: prediction.series_id.clone(),
                timestamp: prediction.timestamp,
                horizon: prediction.horizon,
                model: self.model_name().to_string(),
                mean: prediction.mean,
            })
            .collect::<Vec<_>>();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "classical_expert_bank"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "validation_window": self.validation_window,
            "experts": self.experts.iter().map(ClassicalExpert::metadata).collect::<Vec<_>>(),
            "selected_expert": self.selected_expert().map(ClassicalExpert::metadata),
            "validation_scores": self.validation_scores().iter().map(|score| {
                json!({
                    "expert": score.expert.metadata(),
                    "mse": score.mse,
                    "validation_rows": score.validation_rows,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Default for ClassicalExpertBank {
    fn default() -> Self {
        Self::default_for_season_length(24).expect("default classical bank is valid")
    }
}

struct ValidationSplit {
    train: ForecastFrame,
    validation: Vec<ForecastRow>,
}

fn default_experts(season_length: usize) -> Vec<ClassicalExpert> {
    vec![
        ClassicalExpert::Naive,
        ClassicalExpert::SeasonalNaive { season_length },
        ClassicalExpert::Theta {
            theta: 2.0,
            alpha: 0.3,
        },
        ClassicalExpert::ETS {
            alpha: 0.3,
            beta: 0.1,
        },
        ClassicalExpert::AutoARIMA { max_p: 2, max_d: 1 },
        ClassicalExpert::LocalLevelKalman,
        ClassicalExpert::Kalman,
        ClassicalExpert::AutoLocalLevelKalman,
        ClassicalExpert::AutoKalman,
    ]
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
                "not enough history for classical bank validation".to_string(),
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

fn score_expert(
    expert: &ClassicalExpert,
    train: &ForecastFrame,
    validation: &[ForecastRow],
) -> Result<ClassicalExpertScore> {
    let horizon = validation_horizon(validation);
    let mut model = expert.build()?;
    model.fit(train)?;
    let predictions = model.predict(horizon)?;
    let mut squared_error = 0.0;
    let mut count = 0usize;
    for actual in validation {
        let train_len = train.rows_for_series(&actual.series_id).len();
        let validation_idx = validation
            .iter()
            .filter(|row| row.series_id == actual.series_id && row.timestamp <= actual.timestamp)
            .count();
        let expected_horizon = validation_idx;
        if train_len == 0 || expected_horizon == 0 {
            continue;
        }
        if let Some(prediction) = predictions.predictions().iter().find(|prediction| {
            prediction.series_id == actual.series_id && prediction.horizon == expected_horizon
        }) {
            let err = prediction.mean - actual.target;
            squared_error += err * err;
            count += 1;
        }
    }
    if count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "classical expert produced no comparable validation predictions".to_string(),
        ));
    }
    Ok(ClassicalExpertScore {
        expert: expert.clone(),
        mse: squared_error / count as f64,
        validation_rows: count,
    })
}

fn validation_horizon(validation: &[ForecastRow]) -> usize {
    history_by_series(validation)
        .values()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
}

impl ClassicalExpertBank {
    fn effective_validation_window(&self, frame: &ForecastFrame) -> usize {
        if let Some(window) = self.validation_window {
            return window;
        }
        let min_history = history_by_series(frame.rows())
            .values()
            .map(Vec::len)
            .min()
            .unwrap_or(0);
        if min_history < 4 {
            0
        } else {
            (min_history / 5).clamp(1, 8)
        }
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

fn expert_rank(expert: &ClassicalExpert) -> usize {
    match expert {
        ClassicalExpert::Naive => 0,
        ClassicalExpert::SeasonalNaive { .. } => 1,
        ClassicalExpert::Theta { .. } => 2,
        ClassicalExpert::ETS { .. } => 3,
        ClassicalExpert::AutoARIMA { .. } => 4,
        ClassicalExpert::LocalLevelKalman => 5,
        ClassicalExpert::Kalman => 6,
        ClassicalExpert::AutoLocalLevelKalman => 7,
        ClassicalExpert::AutoKalman => 8,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
