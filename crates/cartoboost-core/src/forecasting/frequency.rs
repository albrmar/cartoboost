use crate::{CartoBoostError, Result};
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime};
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

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hourly => "hourly",
            Self::Daily => "daily",
            Self::Weekly => "weekly",
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

pub fn parse_forecast_timestamp(value: &str) -> Result<NaiveDateTime> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "forecast timestamp must not be empty".to_string(),
        ));
    }
    for format in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    ] {
        if let Ok(timestamp) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Ok(timestamp);
        }
    }
    if let Ok(date) = NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        return date.and_hms_opt(0, 0, 0).ok_or_else(|| {
            CartoBoostError::InvalidInput(format!("invalid forecast timestamp {value:?}"))
        });
    }
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(trimmed) {
        return Ok(timestamp.naive_utc());
    }
    Err(CartoBoostError::InvalidInput(format!(
        "forecast timestamp {value:?} is not parseable"
    )))
}
