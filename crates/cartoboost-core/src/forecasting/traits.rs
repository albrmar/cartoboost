use crate::forecasting::{ForecastFrame, ForecastResult};
use crate::Result;
use serde_json::Value;

pub trait Forecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()>;
    fn predict(&self, horizon: usize) -> Result<ForecastResult>;
    fn model_name(&self) -> &'static str;
    fn metadata(&self) -> Value;
}
