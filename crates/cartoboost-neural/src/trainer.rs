use crate::artifact::{build_embedding_table_artifact, ArtifactFallbackKind, EmbeddingRow};
use crate::error::{NeuralError, Result};
use std::collections::HashMap;

const DEFAULT_TRAINING_ITERS: usize = 24;
const DEFAULT_HEAD_REGULARIZATION: f64 = 1e-2;
const DEFAULT_PRIOR_STRENGTH: f64 = 1.0;

struct IdState {
    id: u64,
    mean: f64,
    count: f64,
    prior: Vec<f32>,
    embedding: Vec<f32>,
}

pub fn fit_embedding_table(
    dim: usize,
    ids: &[u64],
    target: &[f32],
    fallback: ArtifactFallbackKind,
    random_state: Option<u64>,
) -> Result<crate::artifact::EmbeddingTable> {
    if dim == 0 {
        return Err(NeuralError::InvalidArgument(
            "embedding dimension must be positive".to_string(),
        ));
    }

    if ids.len() != target.len() {
        return Err(NeuralError::InvalidArgument(
            "ids and target must have the same length".to_string(),
        ));
    }

    let mut stats: HashMap<u64, (f64, usize)> = HashMap::new();
    for (&id, &value) in ids.iter().zip(target.iter()) {
        let entry = stats.entry(id).or_insert((0.0, 0));
        entry.0 += f64::from(value);
        entry.1 += 1;
    }

    let mut states: Vec<IdState> = Vec::with_capacity(stats.len());
    let seed = random_state.unwrap_or(0x6a09e667f3bcc909);
    let total_count: f64 = ids.len() as f64;

    for (id, (sum, count)) in stats {
        let mean = if count == 0 { 0.0 } else { sum / count as f64 };
        let count = count as f64;
        let prior = build_prior_vector(id, dim, seed);
        let embedding = prior.clone();
        states.push(IdState {
            id,
            mean,
            count,
            prior,
            embedding,
        });
    }

    if states.is_empty() {
        let artifact = build_embedding_table_artifact(dim, Vec::new(), fallback)?;
        return crate::artifact::EmbeddingTable::from_artifact(artifact);
    }

    let mut head = initial_head_vector(dim, &states, &seed);
    let mut bias = states
        .iter()
        .map(|state| state.mean * state.count)
        .sum::<f64>()
        / total_count;

    for _ in 0..DEFAULT_TRAINING_ITERS {
        for state in &mut states {
            state.embedding = updated_embedding(state.mean, state.count, &state.prior, &head, bias);
        }

        let bias_numerator = states.iter().fold(0.0_f64, |acc, state| {
            acc + state.count * (state.mean - dot_f64(&head, &state.embedding))
        });
        if total_count > 0.0 {
            bias = bias_numerator / total_count;
        }

        if let Some(next_head) = solve_head(dim, &states, bias) {
            head = next_head;
        }
    }

    let mut rows: Vec<EmbeddingRow> = states
        .into_iter()
        .map(|state| EmbeddingRow {
            id: state.id,
            values: state
                .embedding
                .into_iter()
                .map(|value| value.clamp(-1.0, 1.0))
                .collect(),
        })
        .collect();
    rows.sort_by_key(|row| row.id);

    let artifact = build_embedding_table_artifact(dim, rows, fallback)?;
    crate::artifact::EmbeddingTable::from_artifact(artifact)
}

fn initial_head_vector(dim: usize, states: &[IdState], seed: &u64) -> Vec<f32> {
    let mut head = vec![0.0_f32; dim];
    let mut total_count = 0.0_f64;

    for state in states {
        total_count += state.count;
        for (index, value) in state.prior.iter().enumerate() {
            head[index] += value * state.count as f32;
        }
    }

    if total_count > 0.0 {
        for value in &mut head {
            *value /= total_count as f32;
        }
    }

    if dot_sq(&head) < 1e-12 {
        head = vec![0.0_f32; dim];
        head[0] = 1.0;
    }

    let mut state = *seed;
    let mut state = splitmix(&mut state);
    let jitter = splitmix_normalized(&mut state) as f32 * 0.05_f32;
    for (index, value) in head.iter_mut().enumerate() {
        *value += jitter / (index.max(1) as f32);
    }

    normalize(&mut head);
    head
}

fn updated_embedding(mean: f64, count: f64, prior: &[f32], head: &[f32], bias: f64) -> Vec<f32> {
    let dim = head.len();
    let mut rhs = vec![0.0_f64; dim];
    let residual = mean - bias;
    let head_norm_sq = dot_sq(head);

    for (index, value) in prior.iter().enumerate() {
        rhs[index] = DEFAULT_PRIOR_STRENGTH * f64::from(*value);
    }

    for (index, value) in head.iter().enumerate() {
        rhs[index] += count * residual * f64::from(*value);
    }

    let projected = dot_f64_f64(&rhs, head);
    let denom = if head_norm_sq > 0.0 {
        DEFAULT_PRIOR_STRENGTH * (DEFAULT_PRIOR_STRENGTH + count * head_norm_sq)
    } else {
        DEFAULT_PRIOR_STRENGTH * DEFAULT_PRIOR_STRENGTH
    };

    rhs.iter()
        .enumerate()
        .map(|(index, value)| {
            let value = value / DEFAULT_PRIOR_STRENGTH
                - (count * projected * f64::from(head[index])) / denom;
            value as f32
        })
        .collect()
}

#[allow(clippy::needless_range_loop)]
fn solve_head(dim: usize, states: &[IdState], bias: f64) -> Option<Vec<f32>> {
    let mut a = vec![vec![0.0_f64; dim]; dim];
    let mut b = vec![0.0_f64; dim];

    for state in states {
        let residual = state.mean - bias;
        for row in 0..dim {
            let row_value = f64::from(state.embedding[row]);
            for col in row..dim {
                a[row][col] += state.count * row_value * f64::from(state.embedding[col]);
            }
            b[row] += state.count * residual * row_value;
        }
    }

    for index in 0..dim {
        for col in 0..index {
            a[index][col] = a[col][index];
        }
        a[index][index] += DEFAULT_HEAD_REGULARIZATION;
    }

    let solution = solve_linear_system(a, b)?;
    Some(solution.into_iter().map(|value| value as f32).collect())
}

#[allow(clippy::needless_range_loop)]
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let dim = b.len();
    let eps = 1e-12;

    for col in 0..dim {
        let mut pivot = col;
        let mut pivot_mag = a[col][col].abs();
        for row in (col + 1)..dim {
            let candidate = a[row][col].abs();
            if candidate > pivot_mag {
                pivot = row;
                pivot_mag = candidate;
            }
        }

        if pivot_mag <= eps {
            return None;
        }

        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }

        let pivot_value = a[col][col];
        for index in col..dim {
            a[col][index] /= pivot_value;
        }
        b[col] /= pivot_value;

        for row in 0..dim {
            if row == col {
                continue;
            }

            let factor = a[row][col];
            if factor.abs() <= eps {
                continue;
            }

            for index in col..dim {
                a[row][index] -= factor * a[col][index];
            }
            b[row] -= factor * b[col];
        }
    }

    Some(b)
}

fn build_prior_vector(id: u64, dim: usize, random_state: u64) -> Vec<f32> {
    let mut state = id ^ random_state;
    let mut state = splitmix(&mut state);
    let mut output = Vec::with_capacity(dim);
    for _ in 0..dim {
        output.push((splitmix_normalized(&mut state) as f32 - 0.5) * 0.2);
    }
    output
}

fn splitmix_normalized(state: &mut u64) -> f64 {
    let mut u = 0.0_f64;
    for _ in 0..2 {
        let next = splitmix(state);
        u = (next as f64) / ((u64::MAX as f64) + 1.0);
        if u > 0.0 {
            break;
        }
    }
    u
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z = z ^ (z >> 31);
    z
}

fn dot_sq(values: &[f32]) -> f64 {
    values.iter().map(|value| f64::from(*value * *value)).sum()
}

fn dot_f64(values: &[f32], other: &[f32]) -> f64 {
    values
        .iter()
        .zip(other)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}

fn dot_f64_f64(values: &[f64], other: &[f32]) -> f64 {
    values
        .iter()
        .zip(other)
        .map(|(a, b)| (*a) * f64::from(*b))
        .sum()
}

fn normalize(values: &mut [f32]) {
    let norm = dot_sq(values).sqrt();
    if norm <= 0.0 {
        return;
    }

    let scale = 1.0_f64 / norm;
    for value in values {
        *value = (*value as f64 * scale) as f32;
    }
}
