# Neural Embedding Benchmark (latest run)

## Command

`uv run --group dev python scripts/run_neural_embedding_benchmark.py --output target/validation/neural_benchmark.json`

Seed: `42`

Configuration: rows=2000, features=8, cells=128, neural_dim=16, train_frac=0.8, split_mode=all

## Scenario results

| Scenario        | ID key     | Train | Test | Base MAE | Hybrid MAE | ΔMAE | Base fit (ms) | Hybrid fit (ms) | Base predict (ms) | Hybrid predict (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cold_destination | destination | 1597 | 403 | 0.6952 | 0.6887 | +0.0065 | 23.9 | 413.5 | 1.176 | 1.121 |
| cold_origin | origin | 1603 | 397 | 0.7810 | 0.7967 | -0.0157 | 23.1 | 412.4 | 0.335 | 0.889 |
| geo_blocked | origin | 1620 | 380 | 0.6724 | 0.6403 | +0.0320 | 23.2 | 420.0 | 0.273 | 1.039 |
| random | origin | 1600 | 400 | 0.6741 | 0.4812 | +0.1929 | 23.9 | 426.6 | 0.384 | 0.899 |
| tail | origin | 1600 | 400 | 0.6873 | 0.4551 | +0.2322 | 23.8 | 422.2 | 0.207 | 1.017 |
| temporal_blocked | origin | 1600 | 400 | 0.6861 | 0.4570 | +0.2291 | 24.0 | 427.5 | 0.358 | 1.364 |

## Aggregate summary

- Mean baseline MAE: `0.6993`
- Mean hybrid MAE: `0.5865`
- Mean MAE delta (hybrid - baseline): `0.1128`
- Scenarios with improvement: `5 / 6`
- Best improvement: `0.2322`
- Worst improvement: `-0.0157`
