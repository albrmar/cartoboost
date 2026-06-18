use crate::forecasting::ForecastResult;
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

pub const FORECAST_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_MANIFEST_FILE: &str = "manifest.json";
pub const DEFAULT_FORECAST_FILE: &str = "forecast.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastArtifactManifest {
    pub schema_version: u32,
    pub model_name: String,
    pub horizon: usize,
    pub forecast_path: String,
    #[serde(default)]
    pub frequency: Option<String>,
    #[serde(default)]
    pub target_column: Option<String>,
    #[serde(default)]
    pub time_column: Option<String>,
    #[serde(default)]
    pub series_id_column: Option<String>,
    #[serde(default)]
    pub feature_config: Map<String, Value>,
    #[serde(default)]
    pub params: Map<String, Value>,
    #[serde(default)]
    pub metrics: Map<String, Value>,
    #[serde(default)]
    pub interval_metadata: Map<String, Value>,
    #[serde(default)]
    pub ensemble_metadata: Map<String, Value>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastArtifact {
    pub manifest: ForecastArtifactManifest,
    pub forecast: ForecastResult,
}

impl ForecastArtifactManifest {
    pub fn new(
        model_name: impl Into<String>,
        horizon: usize,
        forecast_path: impl Into<String>,
    ) -> Result<Self> {
        let model_name = model_name.into();
        let forecast_path = forecast_path.into();
        if model_name.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "artifact model_name must be non-empty".to_string(),
            ));
        }
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "artifact horizon must be positive".to_string(),
            ));
        }
        if forecast_path.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "artifact forecast_path must be non-empty".to_string(),
            ));
        }
        Ok(Self {
            schema_version: FORECAST_ARTIFACT_SCHEMA_VERSION,
            model_name,
            horizon,
            forecast_path,
            frequency: None,
            target_column: None,
            time_column: None,
            series_id_column: None,
            feature_config: Map::new(),
            params: Map::new(),
            metrics: Map::new(),
            interval_metadata: Map::new(),
            ensemble_metadata: Map::new(),
            metadata: Map::new(),
        })
    }
}

impl ForecastArtifact {
    pub fn new(manifest: ForecastArtifactManifest, forecast: ForecastResult) -> Result<Self> {
        validate_manifest_matches_forecast(&manifest, &forecast)?;
        Ok(Self { manifest, forecast })
    }

    pub fn save_json(&self, directory: impl AsRef<Path>) -> Result<()> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let manifest_path = directory.join(DEFAULT_MANIFEST_FILE);
        let forecast_path = resolve_child_path(directory, &self.manifest.forecast_path)?;
        let manifest_file = BufWriter::new(File::create(manifest_path)?);
        serde_json::to_writer_pretty(manifest_file, &self.manifest)?;
        let forecast_file = BufWriter::new(File::create(forecast_path)?);
        serde_json::to_writer_pretty(forecast_file, &self.forecast)?;
        Ok(())
    }

    pub fn load_json(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let manifest_path = directory.join(DEFAULT_MANIFEST_FILE);
        let manifest_file = BufReader::new(File::open(manifest_path)?);
        let manifest: ForecastArtifactManifest = serde_json::from_reader(manifest_file)?;
        if manifest.schema_version != FORECAST_ARTIFACT_SCHEMA_VERSION {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported forecast artifact schema_version {}",
                manifest.schema_version
            )));
        }
        let forecast_path = resolve_child_path(directory, &manifest.forecast_path)?;
        let forecast_file = BufReader::new(File::open(forecast_path)?);
        let forecast: ForecastResult = serde_json::from_reader(forecast_file)?;
        Self::new(manifest, forecast)
    }
}

fn validate_manifest_matches_forecast(
    manifest: &ForecastArtifactManifest,
    forecast: &ForecastResult,
) -> Result<()> {
    if forecast
        .predictions()
        .iter()
        .any(|prediction| prediction.horizon > manifest.horizon)
    {
        return Err(CartoBoostError::InvalidInput(
            "artifact forecast contains predictions beyond manifest horizon".to_string(),
        ));
    }
    Ok(())
}

fn resolve_child_path(directory: &Path, child: &str) -> Result<PathBuf> {
    let path = Path::new(child);
    if path.is_absolute() || child.contains("..") {
        return Err(CartoBoostError::InvalidInput(
            "artifact child paths must be relative and stay inside the artifact directory"
                .to_string(),
        ));
    }
    Ok(directory.join(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::{ForecastPrediction, ForecastResult};
    use chrono::NaiveDate;

    #[test]
    fn artifact_manifest_and_forecast_round_trip_as_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let forecast = ForecastResult::new(vec![ForecastPrediction {
            series_id: "PULocationID=100".to_string(),
            timestamp: ts(),
            horizon: 1,
            model: "naive".to_string(),
            mean: 42.0,
        }])
        .expect("forecast");
        let mut manifest =
            ForecastArtifactManifest::new("naive", 1, DEFAULT_FORECAST_FILE).expect("manifest");
        manifest.target_column = Some("pickup_demand".to_string());
        manifest
            .metadata
            .insert("split".to_string(), Value::from("taxi_holdout"));
        let artifact = ForecastArtifact::new(manifest, forecast).expect("artifact");

        artifact.save_json(dir.path()).expect("save");
        let restored = ForecastArtifact::load_json(dir.path()).expect("load");

        assert_eq!(restored, artifact);
    }

    #[test]
    fn artifact_rejects_parent_path_forecast_file() {
        let err = resolve_child_path(Path::new("/tmp/artifact"), "../forecast.json")
            .expect_err("invalid path");

        assert!(err.to_string().contains("relative"));
    }

    fn ts() -> chrono::NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 1, 1)
            .expect("date")
            .and_hms_opt(0, 0, 0)
            .expect("time")
    }
}
