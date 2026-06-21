use crate::forecasting::{parse_forecast_timestamp, ForecastFrequency};
use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

pub const SINGLE_SERIES_ID: &str = "__single__";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastRow {
    pub series_id: String,
    pub timestamp: NaiveDateTime,
    pub target: f64,
    #[serde(default)]
    pub covariates: BTreeMap<String, f64>,
}

impl ForecastRow {
    pub fn new(series_id: impl Into<String>, timestamp: NaiveDateTime, target: f64) -> Self {
        Self::with_covariates(series_id, timestamp, target, BTreeMap::new())
    }

    pub fn with_covariates(
        series_id: impl Into<String>,
        timestamp: NaiveDateTime,
        target: f64,
        covariates: BTreeMap<String, f64>,
    ) -> Self {
        let series_id = series_id.into();
        Self {
            series_id: if series_id.is_empty() {
                SINGLE_SERIES_ID.to_string()
            } else {
                series_id
            },
            timestamp,
            target,
            covariates,
        }
    }

    pub fn single(timestamp: NaiveDateTime, target: f64) -> Self {
        Self::new(SINGLE_SERIES_ID, timestamp, target)
    }

    pub fn from_timestamp_str(
        series_id: impl Into<String>,
        timestamp: &str,
        target: f64,
    ) -> Result<Self> {
        Ok(Self::with_covariates(
            series_id,
            parse_forecast_timestamp(timestamp)?,
            target,
            BTreeMap::new(),
        ))
    }

    pub fn from_timestamp_str_with_covariates(
        series_id: impl Into<String>,
        timestamp: &str,
        target: f64,
        covariates: BTreeMap<String, f64>,
    ) -> Result<Self> {
        Ok(Self::with_covariates(
            series_id,
            parse_forecast_timestamp(timestamp)?,
            target,
            covariates,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastFrame {
    rows: Vec<ForecastRow>,
    frequency: ForecastFrequency,
    metadata: ForecastFrameMetadata,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastFrameMetadata {
    pub timestamp_col: Option<String>,
    pub target_col: Option<String>,
    pub series_id_col: Option<String>,
    pub static_covariates: Vec<String>,
    pub known_future_covariates: Vec<String>,
    pub historical_covariates: Vec<String>,
}

impl ForecastFrame {
    pub fn new(rows: Vec<ForecastRow>, frequency: ForecastFrequency) -> Result<Self> {
        Self::with_metadata(rows, frequency, ForecastFrameMetadata::default())
    }

    pub fn with_metadata(
        mut rows: Vec<ForecastRow>,
        frequency: ForecastFrequency,
        metadata: ForecastFrameMetadata,
    ) -> Result<Self> {
        if rows.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "forecast frame must contain at least one row".to_string(),
            ));
        }
        validate_metadata(&metadata)?;
        for row in &mut rows {
            if row.series_id.is_empty() {
                row.series_id = SINGLE_SERIES_ID.to_string();
            }
            if !row.target.is_finite() {
                return Err(CartoBoostError::InvalidInput(
                    "forecast targets must be finite".to_string(),
                ));
            }
            for (name, value) in &row.covariates {
                if name.is_empty() {
                    return Err(CartoBoostError::InvalidInput(
                        "forecast covariate names must not be empty".to_string(),
                    ));
                }
                if !value.is_finite() {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "forecast covariate {name:?} for series {} at {} must be finite",
                        row.series_id, row.timestamp
                    )));
                }
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
        Ok(Self {
            rows,
            frequency,
            metadata,
        })
    }

    pub fn from_string_rows(
        rows: Vec<(String, String, f64)>,
        frequency: ForecastFrequency,
        metadata: ForecastFrameMetadata,
    ) -> Result<Self> {
        let parsed = rows
            .into_iter()
            .map(|(series_id, timestamp, target)| {
                ForecastRow::from_timestamp_str(series_id, &timestamp, target)
            })
            .collect::<Result<Vec<_>>>()?;
        Self::with_metadata(parsed, frequency, metadata)
    }

    pub fn from_string_rows_with_covariates(
        rows: Vec<(String, String, f64, BTreeMap<String, f64>)>,
        frequency: ForecastFrequency,
        metadata: ForecastFrameMetadata,
    ) -> Result<Self> {
        let parsed = rows
            .into_iter()
            .map(|(series_id, timestamp, target, covariates)| {
                ForecastRow::from_timestamp_str_with_covariates(
                    series_id, &timestamp, target, covariates,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Self::with_metadata(parsed, frequency, metadata)
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

    pub fn metadata(&self) -> &ForecastFrameMetadata {
        &self.metadata
    }

    pub fn metadata_value(&self) -> Value {
        json!({
            "timestamp_col": self.metadata.timestamp_col,
            "target_col": self.metadata.target_col,
            "series_id_col": self.metadata.series_id_col,
            "frequency": self.frequency.as_str(),
            "is_panel": self.is_panel(),
            "n_rows": self.rows.len(),
            "series_ids": self.series_ids(),
            "static_covariates": self.metadata.static_covariates,
            "known_future_covariates": self.metadata.known_future_covariates,
            "historical_covariates": self.metadata.historical_covariates,
        })
    }

    pub fn metadata_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.metadata_value()).map_err(CartoBoostError::from)
    }

    pub fn is_panel(&self) -> bool {
        self.series_ids()
            .iter()
            .any(|series_id| series_id != SINGLE_SERIES_ID)
    }
}

fn validate_metadata(metadata: &ForecastFrameMetadata) -> Result<()> {
    let mut seen = HashSet::new();
    for (label, values) in [
        ("static_covariates", &metadata.static_covariates),
        ("known_future_covariates", &metadata.known_future_covariates),
        ("historical_covariates", &metadata.historical_covariates),
    ] {
        let mut role_seen = HashSet::new();
        for value in values {
            if value.is_empty() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "{label} must not contain empty column names"
                )));
            }
            if !role_seen.insert(value) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "{label} must not contain duplicate column names"
                )));
            }
            if !seen.insert(value) {
                return Err(CartoBoostError::InvalidInput(
                    "forecast covariate columns must belong to only one role".to_string(),
                ));
            }
        }
    }
    Ok(())
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
