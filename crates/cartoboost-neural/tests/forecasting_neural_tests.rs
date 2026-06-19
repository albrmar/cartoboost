pub use cartoboost_neural::{NeuralError, Result};

#[allow(dead_code, unused_imports)]
#[path = "../src/forecasting/mod.rs"]
mod forecasting;

use cartoboost_core::forecasting::{ForecastFrame, ForecastFrequency, ForecastRow, Forecaster};
use forecasting::{NBeatsConfig, NBeatsForecaster, NHiTSConfig, NHiTSForecaster, StandardScaler};

#[test]
fn nbeats_forecaster_is_deterministic_on_cpu() {
    let frame = taxi_frame();
    let config = NBeatsConfig {
        input_size: 4,
        hidden_size: 6,
        epochs: 30,
        learning_rate: 0.01,
    };
    let mut first = NBeatsForecaster::new(config.clone()).expect("first model");
    let mut second = NBeatsForecaster::new(config).expect("second model");

    first.fit(&frame).expect("first fit");
    second.fit(&frame).expect("second fit");

    let first_predictions = first.predict(3).expect("first predict");
    let second_predictions = second.predict(3).expect("second predict");

    assert_eq!(first_predictions, second_predictions);
    assert_eq!(first_predictions.predictions().len(), 6);
    assert!(first_predictions
        .predictions()
        .iter()
        .all(|prediction| prediction.mean.is_finite()));
}

#[test]
fn nhits_forecaster_handles_panel_taxi_series() {
    let frame = taxi_frame();
    let config = NHiTSConfig {
        input_size: 4,
        hidden_size: 6,
        epochs: 30,
        learning_rate: 0.01,
        pooling_size: 2,
    };
    let mut model = NHiTSForecaster::new(config).expect("model");

    model.fit(&frame).expect("fit");
    let predictions = model.predict(2).expect("predict");

    assert_eq!(predictions.predictions().len(), 4);
    assert_eq!(predictions.predictions()[0].series_id, "PU1->DO2");
    assert_eq!(predictions.predictions()[0].horizon, 1);
    assert_eq!(predictions.predictions()[1].horizon, 2);
    assert_eq!(predictions.predictions()[2].series_id, "PU3->DO4");
}

#[test]
fn scaler_round_trips_constant_series() {
    let scaler = StandardScaler::fit(&[42.0, 42.0, 42.0]).expect("scaler");

    assert_eq!(scaler.transform(42.0), 0.0);
    assert_eq!(scaler.inverse_transform(0.0), 42.0);
    assert!(scaler.scale() > 0.0);
}

fn taxi_frame() -> ForecastFrame {
    let rows = (1..=10)
        .flat_map(|hour| {
            [
                ForecastRow::from_timestamp_str(
                    "PU1->DO2",
                    &timestamp(hour),
                    10.0 + hour as f64 * 1.5,
                )
                .expect("row"),
                ForecastRow::from_timestamp_str(
                    "PU3->DO4",
                    &timestamp(hour),
                    25.0 - hour as f64 * 0.75,
                )
                .expect("row"),
            ]
        })
        .collect();
    ForecastFrame::new(rows, ForecastFrequency::Hourly).expect("frame")
}

fn timestamp(hour: u32) -> String {
    format!("2024-01-01T{hour:02}:00:00")
}
