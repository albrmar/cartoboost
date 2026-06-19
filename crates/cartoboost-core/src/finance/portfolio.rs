use crate::{CartoBoostError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioSignal {
    pub asset_id: String,
    pub expected_return: f64,
}

impl PortfolioSignal {
    pub fn new(asset_id: impl Into<String>, expected_return: f64) -> Result<Self> {
        let asset_id = asset_id.into();
        if asset_id.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "asset_id must not be empty".to_string(),
            ));
        }
        if !expected_return.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "expected_return must be finite".to_string(),
            ));
        }
        Ok(Self {
            asset_id,
            expected_return,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioPosition {
    pub asset_id: String,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Portfolio {
    pub positions: Vec<PortfolioPosition>,
}

impl Portfolio {
    pub fn deterministic_long_short(
        signals: &[PortfolioSignal],
        long_count: usize,
        short_count: usize,
    ) -> Result<Self> {
        if signals.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "signals must contain at least one asset".to_string(),
            ));
        }
        if long_count == 0 && short_count == 0 {
            return Err(CartoBoostError::InvalidInput(
                "at least one long or short position is required".to_string(),
            ));
        }
        if long_count + short_count > signals.len() {
            return Err(CartoBoostError::InvalidInput(
                "long_count plus short_count must not exceed asset count".to_string(),
            ));
        }
        let mut ranked = signals.to_vec();
        ranked.sort_by(|left, right| {
            right
                .expected_return
                .total_cmp(&left.expected_return)
                .then_with(|| left.asset_id.cmp(&right.asset_id))
        });

        let long_weight = if long_count == 0 {
            0.0
        } else {
            1.0 / long_count as f64
        };
        let short_weight = if short_count == 0 {
            0.0
        } else {
            -1.0 / short_count as f64
        };
        let mut positions = Vec::with_capacity(long_count + short_count);
        positions.extend(
            ranked
                .iter()
                .take(long_count)
                .map(|signal| PortfolioPosition {
                    asset_id: signal.asset_id.clone(),
                    weight: long_weight,
                }),
        );
        positions.extend(
            ranked
                .iter()
                .rev()
                .take(short_count)
                .map(|signal| PortfolioPosition {
                    asset_id: signal.asset_id.clone(),
                    weight: short_weight,
                }),
        );
        positions.sort_by(|left, right| left.asset_id.cmp(&right.asset_id));
        Ok(Self { positions })
    }
}
