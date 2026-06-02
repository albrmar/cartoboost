use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[derive(Clone)]
struct SyntheticForest {
    trees: usize,
    depth: usize,
    thresholds: Vec<f32>,
    leaves: Vec<f32>,
}

impl SyntheticForest {
    fn new(trees: usize, depth: usize) -> Self {
        let split_count = trees * ((1 << depth) - 1);
        let leaf_count = trees * (1 << depth);
        let thresholds = (0..split_count)
            .map(|index| (index as f32 * 0.013).sin())
            .collect();
        let leaves = (0..leaf_count)
            .map(|index| (index as f32 * 0.007).cos() * 0.1)
            .collect();
        Self {
            trees,
            depth,
            thresholds,
            leaves,
        }
    }
}

fn synthetic_features(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| ((index as f32 * 0.017).sin() + (index as f32 * 0.003).cos()) * 0.5)
        .collect()
}

fn predict_placeholder(forest: &SyntheticForest, features: &[f32], rows: usize, cols: usize) -> f32 {
    let split_stride = (1 << forest.depth) - 1;
    let leaf_stride = 1 << forest.depth;
    let mut checksum = 0.0_f32;

    for row in 0..rows {
        let row_offset = row * cols;
        let mut prediction = 0.0_f32;
        for tree in 0..forest.trees {
            let mut node = 0;
            for level in 0..forest.depth {
                let feature = features[row_offset + ((tree + level) % cols)];
                let threshold = forest.thresholds[tree * split_stride + node];
                node = node * 2 + 1 + usize::from(feature > threshold);
            }
            let leaf = node - split_stride;
            prediction += forest.leaves[tree * leaf_stride + leaf];
        }
        checksum += prediction;
    }

    checksum
}

fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");
    for &(rows, cols, trees, depth) in &[(1_000, 16, 100, 4), (50_000, 32, 300, 6)] {
        let forest = SyntheticForest::new(trees, depth);
        let features = synthetic_features(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("batch_placeholder", format!("{rows}x{cols}_{trees}x{depth}")),
            &(forest, features),
            |bench, (forest, features)| {
                bench.iter(|| {
                    predict_placeholder(
                        black_box(forest),
                        black_box(features),
                        black_box(rows),
                        black_box(cols),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(prediction_benches, bench_prediction);
criterion_main!(prediction_benches);
