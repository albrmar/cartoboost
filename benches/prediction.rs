use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use geoboost_core::tree::{Node, Tree, MODEL_ARTIFACT_VERSION};
use geoboost_core::{Dataset, Model};

fn synthetic_dataset(rows: usize, cols: usize) -> Dataset {
    let values = (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            (row as f64 * 0.031 + col as f64 * 0.17).sin()
        })
        .collect();
    Dataset::from_flat(rows, cols, values).expect("synthetic prediction data is valid")
}

fn synthetic_model(trees: usize) -> Model {
    let trees = (0..trees)
        .map(|index| Tree {
            root: Node::Leaf {
                value: (index as f64 * 0.013).cos() * 0.01,
                sample_weight_sum: 1.0,
                training_loss: 0.0,
            },
        })
        .collect();

    Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: 0.25,
        learning_rate: 0.1,
        feature_count: 8,
        feature_schema: None,
        target_name: None,
        training_config: None,
        trees,
    }
}

fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");
    for &(rows, cols, trees) in &[(128, 4, 4), (1_024, 8, 16)] {
        let dataset = synthetic_dataset(rows, cols);
        let model = synthetic_model(trees);
        group.bench_with_input(
            BenchmarkId::new("model_predict", format!("{rows}x{cols}_{trees}trees")),
            &(model, dataset),
            |bench, (model, dataset)| {
                bench.iter(|| black_box(model).predict(black_box(dataset)));
            },
        );
    }
    group.finish();
}

criterion_group!(prediction_benches, bench_prediction);
criterion_main!(prediction_benches);
