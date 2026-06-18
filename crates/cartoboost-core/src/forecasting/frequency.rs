use crate::{CartoBoostError, Result};
use chrono::{Duration, NaiveDateTime};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastFrequency {
    Hourly,
    Daily,
    Weekly,
}

impl ForecastFrequency {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "h" | "H" | "hour" | "hourly" => Ok(Self::Hourly),
            "d" | "D" | "day" | "daily" => Ok(Self::Daily),
            "w" | "W" | "week" | "weekly" => Ok(Self::Weekly),
            other => Err(CartoBoostError::InvalidInput(format!(
                "unsupported forecast frequency {other:?}; expected H, D, or W"
            ))),
        }
    }

    pub fn step(self) -> Duration {
        match self {
            Self::Hourly => Duration::hours(1),
            Self::Daily => Duration::days(1),
            Self::Weekly => Duration::weeks(1),
        }
    }

    pub fn advance(self, timestamp: NaiveDateTime, steps: usize) -> Result<NaiveDateTime> {
        let step_count = i32::try_from(steps).map_err(|_| {
            CartoBoostError::InvalidInput("forecast horizon is too large".to_string())
        })?;
        timestamp
            .checked_add_signed(self.step() * step_count)
            .ok_or_else(|| CartoBoostError::InvalidInput("forecast timestamp overflow".to_string()))
    }
}
