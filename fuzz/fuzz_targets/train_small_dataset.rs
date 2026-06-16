#![no_main]

use cartoboost_core::tree::SplitterKind;
use cartoboost_core::{Booster, BoosterConfig, Dataset};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    let rows = 2 + (data[0] as usize % 6);
    let cols = 1 + (data[1] as usize % 4);
    let needed = rows * cols + rows;
    if data.len() < needed + 2 {
        return;
    }
    let mut offset = 2;
    let mut x = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            row.push((data[offset] as f64 / 16.0) - 8.0);
            offset += 1;
        }
        x.push(row);
    }
    let y = (0..rows)
        .map(|_| {
            let value = (data[offset] as f64 / 16.0) - 8.0;
            offset += 1;
            value
        })
        .collect::<Vec<_>>();
    let Ok(dataset) = Dataset::from_rows(x) else {
        return;
    };
    let config = BoosterConfig {
        n_estimators: 1,
        learning_rate: 0.1,
        max_depth: 2,
        min_samples_leaf: 1,
        min_gain: 0.0,
        splitters: vec![SplitterKind::Axis],
        ..BoosterConfig::default()
    };
    let _ = Booster::new(config).fit(&dataset, &y, None);
});
