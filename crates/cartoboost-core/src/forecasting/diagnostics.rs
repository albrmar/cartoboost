use crate::forecasting::ForecastFrame;
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastDiagnostics {
    pub n_rows: usize,
    pub n_series: usize,
    pub zero_count: usize,
    pub zero_fraction: f64,
    pub intermittent_series_count: usize,
    pub series: Vec<SeriesDiagnostics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesDiagnostics {
    pub series_id: String,
    pub n_rows: usize,
    pub start_timestamp: NaiveDateTime,
    pub end_timestamp: NaiveDateTime,
    pub min_target: f64,
    pub max_target: f64,
    pub mean_target: f64,
    pub zero_count: usize,
    pub nonzero_count: usize,
    pub zero_fraction: f64,
    pub intermittency_ratio: Option<f64>,
    pub mean_nonzero_interval: Option<f64>,
    pub max_zero_run: usize,
    pub is_intermittent: bool,
}

impl ForecastDiagnostics {
    pub fn from_frame(frame: &ForecastFrame) -> Self {
        let series = frame
            .series_ids()
            .into_iter()
            .map(|series_id| {
                let rows = frame.rows_for_series(&series_id);
                SeriesDiagnostics::from_rows(series_id, &rows)
            })
            .collect::<Vec<_>>();
        let n_rows = frame.rows().len();
        let zero_count = series.iter().map(|diag| diag.zero_count).sum::<usize>();
        let zero_fraction = fraction(zero_count, n_rows);
        let intermittent_series_count = series.iter().filter(|diag| diag.is_intermittent).count();
        Self {
            n_rows,
            n_series: series.len(),
            zero_count,
            zero_fraction,
            intermittent_series_count,
            series,
        }
    }

    pub fn series(&self, series_id: &str) -> Option<&SeriesDiagnostics> {
        self.series
            .iter()
            .find(|diagnostics| diagnostics.series_id == series_id)
    }
}

impl SeriesDiagnostics {
    fn from_rows(series_id: String, rows: &[&crate::forecasting::ForecastRow]) -> Self {
        let n_rows = rows.len();
        let start_timestamp = rows
            .first()
            .map(|row| row.timestamp)
            .expect("ForecastFrame guarantees non-empty series");
        let end_timestamp = rows
            .last()
            .map(|row| row.timestamp)
            .expect("ForecastFrame guarantees non-empty series");
        let mut min_target = f64::INFINITY;
        let mut max_target = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut zero_count = 0;
        let mut max_zero_run = 0;
        let mut current_zero_run = 0;
        let mut nonzero_positions = Vec::new();

        for (idx, row) in rows.iter().enumerate() {
            min_target = min_target.min(row.target);
            max_target = max_target.max(row.target);
            sum += row.target;
            if row.target == 0.0 {
                zero_count += 1;
                current_zero_run += 1;
                max_zero_run = max_zero_run.max(current_zero_run);
            } else {
                current_zero_run = 0;
                nonzero_positions.push(idx);
            }
        }

        let nonzero_count = n_rows - zero_count;
        let zero_fraction = fraction(zero_count, n_rows);
        let intermittency_ratio = if nonzero_count == 0 {
            None
        } else {
            Some(zero_count as f64 / nonzero_count as f64)
        };
        let mean_nonzero_interval = if nonzero_positions.len() < 2 {
            None
        } else {
            let first = *nonzero_positions.first().expect("non-empty positions") as f64;
            let last = *nonzero_positions.last().expect("non-empty positions") as f64;
            Some((last - first) / (nonzero_positions.len() - 1) as f64)
        };

        Self {
            series_id,
            n_rows,
            start_timestamp,
            end_timestamp,
            min_target,
            max_target,
            mean_target: sum / n_rows as f64,
            zero_count,
            nonzero_count,
            zero_fraction,
            intermittency_ratio,
            mean_nonzero_interval,
            max_zero_run,
            is_intermittent: zero_count > 0,
        }
    }
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}
