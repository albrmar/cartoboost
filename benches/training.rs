use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[derive(Clone)]
struct SyntheticMatrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
    target: Vec<f32>,
}

impl SyntheticMatrix {
    fn new(rows: usize, cols: usize) -> Self {
        let mut state = 0x5eed_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 32) as f32 / u32::MAX as f32) - 0.5
        };

        let values = (0..rows * cols).map(|_| next()).collect::<Vec<_>>();
        let target = (0..rows)
            .map(|row| {
                let offset = row * cols;
                values[offset] * 1.7 + values[offset + 1] * -0.6 + next() * 0.1
            })
            .collect();

        Self {
            rows,
            cols,
            values,
            target,
        }
    }

    fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.values[start..start + self.cols]
    }
}

fn histogram_training_placeholder(data: &SyntheticMatrix, bins: usize, rounds: usize) -> f32 {
    let mut prediction = vec![0.0_f32; data.rows];
    let mut gain_total = 0.0_f32;

    for _ in 0..rounds {
        for col in 0..data.cols {
            let mut histogram = vec![0.0_f32; bins];
            for row in 0..data.rows {
                let gradient = data.target[row] - prediction[row];
                let value = data.row(row)[col];
                let bin = ((value.abs() * bins as f32) as usize).min(bins - 1);
                histogram[bin] += gradient;
            }
            gain_total += histogram.iter().map(|value| value.abs()).sum::<f32>();
        }

        for row_prediction in &mut prediction {
            *row_prediction += 0.01 * gain_total.signum();
        }
    }

    gain_total
}

fn bench_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("training");
    for &(rows, cols) in &[(1_000, 16), (10_000, 32)] {
        let data = SyntheticMatrix::new(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("histogram_placeholder", format!("{rows}x{cols}")),
            &data,
            |bench, data| {
                bench.iter(|| {
                    histogram_training_placeholder(black_box(data), black_box(64), black_box(10))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(training_benches, bench_training);
criterion_main!(training_benches);
