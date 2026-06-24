# Neural Embedding Benchmark

## Bottom Line

The neural embedding benchmark is a synthetic repeated-ID diagnostic. It shows
the intended pattern: residual embeddings help strongly when IDs recur, help
modestly on some geographic/tail splits, and can hurt when the tested origin
IDs are cold.

## Reproduce

```sh
uv run python scripts/run_neural_embedding_benchmark.py \
  --output target/validation/neural_benchmark.json
```

## Data And Setup

| Field | Value |
| --- | ---: |
| Seed | 42 |
| Rows | 2,000 |
| Dense features | 8 |
| ID cells | 128 |
| Embedding dimension | 16 |
| Train fraction | 0.8 |

## Scenario Breakdown

| Scenario | ID key | Base MAE | Hybrid MAE | MAE improvement | Read |
| --- | --- | ---: | ---: | ---: | --- |
| Random | origin | 0.6741 | 0.4812 | +0.1929 | Strong repeated-ID gain. |
| Tail | origin | 0.6873 | 0.4551 | +0.2322 | Strong rare/tail gain in fixture. |
| Temporal blocked | origin | 0.6861 | 0.4570 | +0.2291 | Reused IDs over time help. |
| Geographic blocked | origin | 0.6724 | 0.6403 | +0.0320 | Smaller spatial gain. |
| Cold destination | destination | 0.6952 | 0.6887 | +0.0065 | Near tie. |
| Cold origin | origin | 0.7810 | 0.7967 | -0.0157 | Warning case: cold IDs can hurt. |

## Interpretation

This is exactly why neural rows must be split-specific. A random repeated-ID
gain does not imply cold-start generalization. Use neural residual embeddings
only when production IDs recur and the deployment split also improves.

