# Neural Embedding Benchmark

## Research Question

When does neural residual embedding improve prediction quality, and when does
it fail because validation IDs were not observed during training?

## Dataset

The benchmark uses a deterministic synthetic geographic-ID regression fixture:

- Seed: `42`
- Rows: `2000`
- Dense features: `8`
- ID cells: `128`
- Embedding dimension: `16`
- Train fraction: `0.8`
- Split modes: random, temporal blocked, geographic blocked, tail, cold origin,
  and cold destination.

## Target

The target is a continuous regression outcome with dense feature signal plus an
ID-specific residual component. The residual component is intentionally useful
when IDs repeat and unreliable when IDs are cold.

## Features

The baseline model receives dense numeric features. The hybrid model receives
the same dense features plus neural residual embeddings keyed by origin or
destination ID depending on the scenario.

## Command

```sh
uv run python scripts/run_neural_embedding_benchmark.py \
  --output target/validation/neural_benchmark.json
```

## Metrics

MAE is the primary quality metric. Fit and prediction milliseconds are reported
as operating-cost context.

## Results

| Scenario | ID key | Train | Test | Base MAE | Hybrid MAE | MAE improvement |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cold_destination | destination | 1597 | 403 | 0.6952 | 0.6887 | +0.0065 |
| cold_origin | origin | 1603 | 397 | 0.7810 | 0.7967 | -0.0157 |
| geo_blocked | origin | 1620 | 380 | 0.6724 | 0.6403 | +0.0320 |
| random | origin | 1600 | 400 | 0.6741 | 0.4812 | +0.1929 |
| tail | origin | 1600 | 400 | 0.6873 | 0.4551 | +0.2322 |
| temporal_blocked | origin | 1600 | 400 | 0.6861 | 0.4570 | +0.2291 |

Aggregate summary:

- Mean baseline MAE: `0.6993`
- Mean hybrid MAE: `0.5865`
- Mean MAE improvement: `0.1128`
- Improved scenarios: `5 / 6`
- Best improvement: `0.2322`
- Worst change: `-0.0157`

## Interpretation

The hybrid model improves strongly when validation rows reuse useful ID
structure: random, tail, temporal-blocked, and geo-blocked splits. The cold
origin split is the warning case. When the validation IDs have no training
history, embeddings can underperform the structured baseline.

The correct conclusion is split-specific: neural residual embeddings are useful
for repeated-ID residual signal, not a guarantee of cold-start generalization.

## Limitations

- The dataset is synthetic.
- The benchmark isolates ID residual behavior and should not be generalized to
  all geographic prediction tasks.
- Timing is machine-dependent and should not be compared without hardware and
  dependency context.
