# Neural Features (Phase 1)

Phase 1 introduces `NeuralEmbeddingFeatures` for **embedding-table** generators only.
This keeps CartoBoost as the runtime scorer and keeps inference in Rust.

## Principle

Neural output is materialized as regular dense columns before scoring.
CartoBoost never calls a neural model at serve time.

- Dense input rows remain unchanged in row count.
- Generated features are appended to the dense feature matrix.
- Row order is preserved.

## Runtime split

- Python owns:
  - neural training/optimization loops
  - fitting an embedding table
  - writing embedding-table artifacts
  - offline/online export helpers
- Rust owns:
  - artifact loading and schema validation
  - fallback behavior
  - lookup speed and feature append into dense matrices
  - final CartoBoost tree model scoring

## Supported type in this PR

### Embedding table

Input: categorical IDs (for example H3/S2 cell IDs) → learned vector per ID.

Artifact fields required:

- artifact version
- embedding dimension
- id type
- row count
- checksum/hash

Fallbacks are implemented as:

- `zero_vector`
- `global_mean_vector`
- `parent_cell` placeholder (callback hook in Rust)

At inference each ID becomes columns:

- `neural.<name>_00`
- `neural.<name>_01`
- ...
- `neural.<name>_<d-1>`

## Why this is first

Embedding-table inference is deterministic, fast, and Rust-native.
It avoids ONNX in this phase and keeps the model artifact flow straightforward.

## Future PRs

ONNX encoders (lane residual encoders, temporal encoders, calibration heads) are
planned for later phases and are not part of Phase 1.

## Benchmarking the phase-1 path

You can compare structured-only CartoBoost against the hybrid residual embedding
pipeline with one command:

```bash
uv run --group dev python scripts/run_neural_embedding_benchmark.py \
  --n-rows 2000 \
  --n-features 8 \
  --n-cells 128 \
  --n-neural-dim 16 \
  --include-sklearn \
  --output target/validation/neural_embedding_benchmark.json
```

The output JSON reports `mae`, `fit_ms`, and `predict_ms` for each model family:

- `cartoboost`: structured-only baseline
- `neural_embedding_hybrid`: residual embedding-table + CartoBoost
- `sklearn_gbr` (when `--include-sklearn` is set): sklearn baseline for comparison
