use crate::forecasting::ForecastFrequency;
use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const SINGLE_SERIES_ID: &str = "__single__";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastRow {
    pub series_id: String,
    pub timestamp: NaiveDateTime,
    pub target: f64,
}

impl ForecastRow {
    pub fn new(series_id: impl Into<String>, timestamp: NaiveDateTime, target: f64) -> Self {
        let series_id = series_id.into();
        Self {
            series_id: if series_id.is_empty() {
                SINGLE_SERIES_ID.to_string()
            } else {
                series_id
            },
            timestamp,
            target,
        }
    }

    pub fn single(timestamp: NaiveDateTime, target: f64) -> Self {
        Self::new(SINGLE_SERIES_ID, timestamp, target)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastFrame {
    rows: Vec<ForecastRow>,
    frequency: ForecastFrequency,
}

impl ForecastFrame {
    pub fn new(mut rows: Vec<ForecastRow>, frequency: ForecastFrequency) -> Result<Self> {
        if rows.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "forecast frame must contain at least one row".to_string(),
            ));
        }
        for row in &mut rows {
            if row.series_id.is_empty() {
                row.series_id = SINGLE_SERIES_ID.to_string();
            }
            if !row.target.is_finite() {
                return Err(CartoBoostError::InvalidInput(
                    "forecast targets must be finite".to_string(),
                ));
            }
        }
        rows.sort_by(|a, b| {
            a.series_id
                .cmp(&b.series_id)
                .then_with(|| a.timestamp.cmp(&b.timestamp))
        });

        let mut seen = HashSet::with_capacity(rows.len());
        for row in &rows {
            if !seen.insert((row.series_id.clone(), row.timestamp)) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "duplicate forecast timestamp {} for series {}",
                    row.timestamp, row.series_id
                )));
            }
        }
        validate_regular_frequency(&rows, frequency)?;
        Ok(Self { rows, frequency })
    }

    pub fn rows(&self) -> &[ForecastRow] {
        &self.rows
    }

    pub fn frequency(&self) -> ForecastFrequency {
        self.frequency
    }

    pub fn series_ids(&self) -> Vec<String> {
        let mut ids = Vec::new();
        let mut last: Option<&str> = None;
        for row in &self.rows {
            if last != Some(row.series_id.as_str()) {
                ids.push(row.series_id.clone());
                last = Some(row.series_id.as_str());
            }
        }
        ids
    }

    pub fn rows_for_series(&self, series_id: &str) -> Vec<&ForecastRow> {
        self.rows
            .iter()
            .filter(|row| row.series_id == series_id)
            .collect()
    }
}

fn validate_regular_frequency(rows: &[ForecastRow], frequency: ForecastFrequency) -> Result<()> {
    let expected = frequency.step();
    let mut previous: Option<&ForecastRow> = None;
    for row in rows {
        if let Some(prev) = previous {
            if prev.series_id == row.series_id {
                let observed = row.timestamp - prev.timestamp;
                if observed != expected {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "irregular forecast frequency for series {} between {} and {}; expected {:?}, observed {:?}",
                        row.series_id, prev.timestamp, row.timestamp, expected, observed
                    )));
                }
            }
        }
        previous = Some(row);
    }
    Ok(())
}
