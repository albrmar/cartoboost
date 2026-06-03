#![no_main]

use geoboost_core::data::SparseSetColumn;
use geoboost_core::tree::{FuzzyKernel, Node, Split, Tree, MODEL_ARTIFACT_VERSION};
use geoboost_core::{Dataset, Model};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    let rows = 2 + (data[0] as usize % 6);
    let cols = 1 + (data[1] as usize % 3);
    let mut offset = 2;
    let needed = rows * cols + rows * 2 + 8;
    if data.len() < needed {
        return;
    }

    let mut dense = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            row.push((data[offset] as f64 / 16.0) - 8.0);
            offset += 1;
        }
        dense.push(row);
    }

    let mut sparse_rows = Vec::with_capacity(rows);
    for _ in 0..rows {
        sparse_rows.push(vec![
            u64::from(data[offset] % 16),
            u64::from(data[offset + 1] % 16),
        ]);
        offset += 2;
    }

    let Ok(dataset) = Dataset::from_rows(dense)
        .and_then(|x| x.with_sparse_sets(vec![SparseSetColumn::new(sparse_rows)]))
    else {
        return;
    };

    let axis = Split::Axis {
        feature: (data[offset] as usize) % cols,
        threshold: (data[offset + 1] as f64 / 16.0) - 8.0,
        missing_goes_left: data[offset + 2] % 2 == 0,
    };
    let sparse = Split::SparseListContainsAny {
        sparse_feature: 0,
        ids: vec![
            u64::from(data[offset + 3] % 16),
            u64::from(data[offset + 4] % 16),
        ],
        missing_goes_left: data[offset + 5] % 2 == 0,
    };
    let split = match data[offset + 6] % 3 {
        0 => axis.clone(),
        1 => sparse,
        _ => Split::Fuzzy {
            base: Box::new(axis),
            bandwidth: 0.1 + f64::from(data[offset + 7] % 8),
            kernel: FuzzyKernel::Linear,
        },
    };

    let model = Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 0.0,
        learning_rate: 0.5,
        feature_count: cols,
        feature_schema: Some(dataset.feature_schema_or_default()),
        target_name: None,
        training_config: None,
        trees: vec![Tree {
            root: Node::Branch {
                split,
                left: Box::new(Node::Leaf {
                    value: -1.0,
                    sample_weight_sum: 1.0,
                    training_loss: 0.0,
                }),
                right: Box::new(Node::Leaf {
                    value: 1.0,
                    sample_weight_sum: 1.0,
                    training_loss: 0.0,
                }),
                gain: 0.0,
                sample_weight_sum: rows as f64,
            },
        }],
    };

    for row in 0..dataset.n_rows() {
        let _ = model.try_predict_dataset_row(&dataset, row);
    }
});
