# Neural Embedding Benchmark

## What It Tests

The neural embedding benchmark is a synthetic repeated-ID diagnostic. It
isolates when residual embeddings help and when they fail because validation
IDs were not observed during training.

It is not a standalone public accuracy benchmark.

## Fixture

Record:

- generator version and seed list;
- row count and dense feature count;
- ID cardinality and embedding dimension;
- train fraction and split modes;
- target construction, including the ID residual component;
- model settings for the base and hybrid rows.

Split modes:

- random repeated-ID;
- temporal blocked;
- geographic blocked;
- tail or rare-ID;
- cold origin;
- cold destination.

## Metrics

MAE is the primary diagnostic metric. Also include fit time, prediction time,
and support slices when available. Timing is only useful when hardware metadata
is recorded.

## Reproduce

```sh
uv run python scripts/run_neural_embedding_benchmark.py \
  --output target/validation/neural_benchmark.json
```

Write temporary outputs under `target/`. Commit generated artifacts only when
the benchmark page is updated in the same change.

## Interpretation

The expected pattern is split-specific:

- repeated-ID splits may pass when ID residual signal is stable;
- tail splits may pass if rare-ID shrinkage is effective;
- cold-origin or cold-destination splits may fail because embeddings have no
  train-side support.

That pattern is useful because it shows where a neural row deserves inclusion
in real taxi benchmarks.

## Claim Boundary

Use this diagnostic to decide whether a neural row belongs in the real taxi
benchmark. Do not use it to claim real geographic accuracy or production
generalization.
