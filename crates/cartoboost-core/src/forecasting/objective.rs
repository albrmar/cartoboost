use crate::forecasting::ForecastMetricSet;
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastObjective {
    Rmse,
    Wape,
}

impl ForecastObjective {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "rmse" => Ok(Self::Rmse),
            "wape" => Ok(Self::Wape),
            other => Err(CartoBoostError::InvalidInput(format!(
                "unknown forecast objective {other:?}; expected 'rmse' or 'wape'"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rmse => "rmse",
            Self::Wape => "wape",
        }
    }

    pub fn metric_value(&self, metrics: &ForecastMetricSet) -> f64 {
        match self {
            Self::Rmse => metrics.rmse,
            Self::Wape => metrics.wape,
        }
    }
}
