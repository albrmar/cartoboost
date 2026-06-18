use crate::forecasting::{ForecastFrame, ForecastRow};
use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastWindow {
    Expanding,
    Sliding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollingOriginSplitter {
    pub horizon: usize,
    pub step: usize,
    pub min_train_size: usize,
    pub max_train_size: Option<usize>,
    pub n_splits: Option<usize>,
    pub window: ForecastWindow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastFold {
    pub fold_id: String,
    pub train_indices: Vec<usize>,
    pub validation_indices: Vec<usize>,
    pub train_start: NaiveDateTime,
    pub train_end: NaiveDateTime,
    pub validation_start: NaiveDateTime,
    pub validation_end: NaiveDateTime,
    pub horizon: usize,
    pub step: usize,
    pub metadata: ForecastFoldMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastFoldMetadata {
    pub origin_timestamp: NaiveDateTime,
    pub train_size: usize,
    pub validation_size: usize,
    pub train_timestamp_count: usize,
    pub validation_timestamp_count: usize,
    pub series_count: usize,
}

impl RollingOriginSplitter {
    pub fn expanding(horizon: usize, min_train_size: usize) -> Result<Self> {
        Self::new(
            horizon,
            1,
            min_train_size,
            None,
            None,
            ForecastWindow::Expanding,
        )
    }

    pub fn sliding(horizon: usize, min_train_size: usize, max_train_size: usize) -> Result<Self> {
        Self::new(
            horizon,
            1,
            min_train_size,
            Some(max_train_size),
            None,
            ForecastWindow::Sliding,
        )
    }

    pub fn new(
        horizon: usize,
        step: usize,
        min_train_size: usize,
        max_train_size: Option<usize>,
        n_splits: Option<usize>,
        window: ForecastWindow,
    ) -> Result<Self> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        if step == 0 {
            return Err(CartoBoostError::InvalidInput(
                "rolling-origin step must be positive".to_string(),
            ));
        }
        if min_train_size == 0 {
            return Err(CartoBoostError::InvalidInput(
                "min_train_size must be positive".to_string(),
            ));
        }
        if let Some(max_train_size) = max_train_size {
            if max_train_size < min_train_size {
                return Err(CartoBoostError::InvalidInput(
                    "max_train_size must be greater than or equal to min_train_size".to_string(),
                ));
            }
        }
        if matches!(window, ForecastWindow::Sliding) && max_train_size.is_none() {
            return Err(CartoBoostError::InvalidInput(
                "sliding forecast windows require max_train_size".to_string(),
            ));
        }
        if n_splits == Some(0) {
            return Err(CartoBoostError::InvalidInput(
                "n_splits must be positive when provided".to_string(),
            ));
        }
        Ok(Self {
            horizon,
            step,
            min_train_size,
            max_train_size,
            n_splits,
            window,
        })
    }

    pub fn split(&self, frame: &ForecastFrame) -> Result<Vec<ForecastFold>> {
        let timestamps = unique_timestamps(frame.rows());
        if timestamps.len() <= self.horizon {
            return Ok(Vec::new());
        }
        let mut candidates = Vec::new();
        let last_origin = timestamps.len() - self.horizon;
        let mut cutoff = self.min_train_size - 1;
        while cutoff < last_origin {
            let train_start_pos = self.max_train_size.map_or(0, |max_train_size| {
                (cutoff + 1).saturating_sub(max_train_size)
            });
            let train_times = &timestamps[train_start_pos..=cutoff];
            let validation_times = &timestamps[cutoff + 1..=cutoff + self.horizon];

            let train_indices = indices_for_timestamps(frame.rows(), train_times);
            let validation_indices = indices_for_timestamps(frame.rows(), validation_times);
            if train_times.len() >= self.min_train_size
                && !train_indices.is_empty()
                && !validation_indices.is_empty()
            {
                let train_end = *train_times
                    .last()
                    .expect("non-empty train timestamp window");
                let validation_start = validation_times[0];
                if train_end >= validation_start {
                    return Err(CartoBoostError::InvalidInput(
                        "forecast split leakage: max(train timestamp) must be < min(validation timestamp)"
                            .to_string(),
                    ));
                }
                let series_count = validation_indices
                    .iter()
                    .map(|index| frame.rows()[*index].series_id.as_str())
                    .collect::<BTreeSet<_>>()
                    .len();
                candidates.push(ForecastFold {
                    fold_id: String::new(),
                    train_indices,
                    validation_indices,
                    train_start: train_times[0],
                    train_end,
                    validation_start,
                    validation_end: *validation_times
                        .last()
                        .expect("non-empty validation timestamp window"),
                    horizon: self.horizon,
                    step: self.step,
                    metadata: ForecastFoldMetadata {
                        origin_timestamp: timestamps[cutoff],
                        train_size: 0,
                        validation_size: 0,
                        train_timestamp_count: train_times.len(),
                        validation_timestamp_count: validation_times.len(),
                        series_count,
                    },
                });
            }
            cutoff += self.step;
        }
        if let Some(n_splits) = self.n_splits {
            let keep_from = candidates.len().saturating_sub(n_splits);
            candidates = candidates.split_off(keep_from);
        }
        for (index, fold) in candidates.iter_mut().enumerate() {
            fold.fold_id = format!("fold_{index:04}");
            fold.metadata.train_size = fold.train_indices.len();
            fold.metadata.validation_size = fold.validation_indices.len();
        }
        Ok(candidates)
    }
}

pub(crate) fn frame_from_indices(
    frame: &ForecastFrame,
    indices: &[usize],
) -> Result<ForecastFrame> {
    let rows = indices
        .iter()
        .map(|index| frame.rows()[*index].clone())
        .collect::<Vec<_>>();
    ForecastFrame::new(rows, frame.frequency())
}

fn unique_timestamps(rows: &[ForecastRow]) -> Vec<NaiveDateTime> {
    rows.iter()
        .map(|row| row.timestamp)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn indices_for_timestamps(rows: &[ForecastRow], timestamps: &[NaiveDateTime]) -> Vec<usize> {
    let timestamp_set = timestamps.iter().copied().collect::<BTreeSet<_>>();
    rows.iter()
        .enumerate()
        .filter_map(|(index, row)| timestamp_set.contains(&row.timestamp).then_some(index))
        .collect()
}
