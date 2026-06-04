use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use geoboost_core::tree::SplitterKind;
use geoboost_core::{Booster, BoosterConfig, Dataset};

fn training_fixture(rows: usize, cols: usize) -> (Dataset, Vec<f64>) {
    let values = (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            (row as f64 * 0.07 + col as f64 * 0.19).sin()
        })
        .collect::<Vec<_>>();
    let target = (0..rows)
        .map(|row| {
            let first = values[row * cols];
            let last = values[row * cols + cols - 1];
            if first + 0.25 * last > 0.0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();

    (
        Dataset::from_flat(rows, cols, values).expect("synthetic training data is valid"),
        target,
    )
}

fn booster_config(n_estimators: usize, max_depth: usize) -> BoosterConfig {
    BoosterConfig {
        n_estimators,
        learning_rate: 0.2,
        max_depth,
        min_samples_leaf: 2,
        min_gain: 0.0,
        splitters: vec![SplitterKind::Axis],
        ..BoosterConfig::default()
    }
}

fn histogram_booster_config(n_estimators: usize, max_depth: usize) -> BoosterConfig {
    BoosterConfig {
        splitters: vec![SplitterKind::AxisHistogram { bins: 64 }],
        ..booster_config(n_estimators, max_depth)
    }
}

fn bench_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("training");
    for &(rows, cols, estimators, depth) in &[(32, 3, 1, 1), (96, 4, 3, 2)] {
        let (dataset, target) = training_fixture(rows, cols);
        let booster = Booster::new(booster_config(estimators, depth));
        group.bench_with_input(
            BenchmarkId::new(
                "booster_fit",
                format!("{rows}x{cols}_{estimators}trees_depth{depth}"),
            ),
            &(booster, dataset, target),
            |bench, (booster, dataset, target)| {
                bench.iter(|| {
                    black_box(booster)
                        .fit(black_box(dataset), black_box(target), None)
                        .expect("benchmark fixture trains")
                });
            },
        );
    }
    for &(rows, cols, estimators, depth) in &[(25_000, 8, 100, 4), (25_000, 3, 100, 4)] {
        let (dataset, target) = training_fixture(rows, cols);
        let booster = Booster::new(histogram_booster_config(estimators, depth));
        group.bench_with_input(
            BenchmarkId::new(
                "booster_fit_axis_histogram",
                format!("{rows}x{cols}_{estimators}trees_depth{depth}"),
            ),
            &(booster, dataset, target),
            |bench, (booster, dataset, target)| {
                bench.iter(|| {
                    black_box(booster)
                        .fit(black_box(dataset), black_box(target), None)
                        .expect("benchmark fixture trains")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(training_benches, bench_training);
criterion_main!(training_benches);
