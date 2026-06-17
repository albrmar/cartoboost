# Neural Features (Phase 1): Embedding-Table → CartoBoost Hybrid

This document defines the complete hybrid pattern used in this repository:

1. Train neural-style embeddings offline (Python).
2. Serialize embeddings into a versioned artifact.
3. Load and validate embeddings at inference (Rust).
4. Append generated dense vectors to the input matrix.
5. Keep `CartoBoostRegressor` as the final scorer.

This keeps all gradient-boosting behavior in the existing CartoBoost runtime while
adding learned dense context through generated columns.

## 1) Why this architecture

A hybrid is intentionally conservative in phase 1.

- **No neural runtime inside scoring path**: inference remains deterministic Rust
  and fast.
- **No API split in the final model**: the booster still sees only numeric dense
  + sparse inputs.
- **Stable ordering and contracts**: generated feature names and counts are fixed
  by the artifact.

The pattern is:

```text
raw row/context
    ↓
existing feature assembler
    ↓
structured dense + sparse
    ↓
NeuralEmbeddingFeatures lookup
    ↓
append generated dense embedding columns
    ↓
CartoBoost predict
```

## 2) What is an embedding feature here?

For this phase, a **neural feature** is a deterministic vector lookup keyed by an
ID:

- **Input**: one or more IDs per row (e.g. cell IDs).
- **Output**: `dim` floating columns for each logical feature.

Example for `dim=16` and origin cell embedding:

- `neural.origin_h3_r7_emb_00`
- `neural.origin_h3_r7_emb_01`
- `...`
- `neural.origin_h3_r7_emb_15`

Important: these are just dense feature columns from the perspective of the
booster.

## 3) Implementation surface (Phase 1)

### Python ownership

- `python/cartoboost/neural/features.py`
  - `NeuralEmbeddingFeatures`
  - Fits per-ID vectors from an ID/value stream.
  - Exports a JSON artifact with version/metadata and checksum.
  - Loads artifact for parity checks.

- `python/cartoboost/neural/pipeline.py`
  - `NeuralEmbeddingRegressor`
  - Orchestrates residual pipeline and final `CartoBoostRegressor` fit.
  - Appends embedding columns at training and inference time.

- `scripts/run_neural_embedding_benchmark.py`
  - Quick structured-only vs hybrid comparison on synthetic data.

### Rust ownership

- `crates/cartoboost-neural/`
  - `EmbeddingTable` artifact parsing/validation
  - fallback logic (`zero`, `global_mean`, parent placeholder)
  - encoder trait and dense block assembly utilities
  - deterministic feature name generation (`name_00`, `name_01`, ...)

## 4) Full training flow (residual mode)

By default, `NeuralEmbeddingRegressor` uses residual training:

1. Fit a baseline CartoBoost model with only structured inputs.
2. Compute residuals:

```python
residual = y - baseline.predict(X_structured)
```

3. Fit neural embedding table on `(ids, residual)`.
4. Transform `ids` into embedding matrix:
   `Z_train = embedding_transformer.transform(ids_train)`.
5. Concatenate:
   `X_aug = [X_structured, Z_train]`.
6. Fit final `CartoBoostRegressor` on `X_aug` and `y`.

### Why residual mode first

The residual model focuses on what the structured model misses. Instead of directly
adding neural residuals into the final output, we expose learned representation to
CartoBoost and let trees decide when that signal is useful.

### Optional non-residual mode

`use_residual=False` trains embeddings on raw target `y` directly. This is
available but less common in this phase.

## 4.5) Transformer setup and internals (`NeuralEmbeddingFeatures`)

Phase 1 currently uses a deterministic embedding-table transformer, not a deep
model runtime.

### What this transformer is

- It is keyed by one or more IDs (`u64` keys).
- It learns one fixed vector per ID in `fit()`.
- It exports rows to JSON artifact for Rust-native lookup.
- At inference, it emits a dense matrix of shape `(n_rows, dim)`.

### How it is trained in code (`fit`)

From `python/cartoboost/neural/features.py`:

1. Cast `ids` to `uint64` and `target` to float32.
2. Aggregate per ID:
   - `sum[target][id] += residual[id]`
   - `count[id] += 1`
3. Compute per-ID mean target:
   - `mean_id = sum[id] / count[id]`
4. Create per-ID vector using seeded noise:
   - `z ~ Normal(0, 0.1^2)` with seed from `blake2b(id, random_state)`
   - `vector = z`
   - `vector[0] += mean_id`
5. Store each vector in memory map `_embeddings[id]`.

So today’s phase 1 learner is: deterministic seed + first-dimension residual
signal injection, then full vector reuse at serve time.

### How lookup works at transform time (`transform`)

For each row id:

- If `id` exists in map: return learned vector.
- Else:
  - `zero_vector`: all zeros
  - `global_mean_vector`: use global mean of learned vectors
  - `parent_cell`: deferred callback hook in Rust path

The output has:

- `rows = n_rows`
- `cols = dim`
- dtype `float32`

### Artifact and schema details

`export(path)` writes:

- `metadata.artifact_type = "cartoboost.neural.embedding_table"`
- `metadata.artifact_version = 1`
- `metadata.dim`
- `metadata.id_type = "u64"`
- `metadata.row_count`
- `metadata.fallback` with strategy fields
- `metadata.checksum` over sorted rows + metadata
- `rows[]` entries with `{id, values}`

`from_artifact(path)` validates:

- artifact type/version
- row count
- per-row width (`len(values) == dim`)
- checksum integrity

### Why this is called a “transformer” despite being a table

Because it transforms raw row context into new dense columns before the booster
consumes them. It is a feature-generation primitive and is intentionally
deterministic in phase 1 to de-risk serving.

## 5) Inference flow

Given input rows and IDs:

1. Assemble dense features.
2. Build/validate ID list from:
   - explicit `ids` argument, or
   - `id_column` reference.
3. Convert IDs to embedding matrix via loaded embedding table.
4. Append embedding columns to dense matrix.
5. Call final CartoBoost model `predict`.

Row count is preserved through the process.

## 6) API contracts

### Required artifact metadata (Phase 1)

- `artifact_type`
- `artifact_version`
- `dim`
- `id_type` (currently `u64`)
- `row_count`
- `checksum`/`hash`
- `fallback` strategy

Checksum is computed over sorted rows and metadata and validated on load.

### Fallback strategies

Missing IDs are resolved by strategy in this order (implemented in Rust):

- `zero_vector`
- `global_mean_vector`
- `parent_cell` (placeholder callback path present; parent resolution currently via callback)

### Feature naming and order

For prefix `name = neural.origin_cell`, `dim = 4`:

- `neural.origin_cell_00`
- `neural.origin_cell_01`
- `neural.origin_cell_02`
- `neural.origin_cell_03`

The booster input matrix for each row is:

```text
[original_dense_features..., neural feature block]
```

## 7) End-to-end code example

```python
import numpy as np
from cartoboost import NeuralEmbeddingRegressor

rng = np.random.default_rng(0)
rows = 1000
ids = rng.integers(1, 200, size=rows, dtype=np.uint64)
X = rng.normal(size=(rows, 8))
y = 1.2 * X[:, 0] - 0.8 * X[:, 1] + (ids % 7) * 0.1 + rng.normal(0.0, 0.1, size=rows)

model = NeuralEmbeddingRegressor(
    dim=16,
    id_column=None,
    final_model_kwargs={
        "n_estimators": 80,
        "learning_rate": 0.08,
        "max_depth": 4,
        "min_gain": 0.0,
    },
)

# ids provided directly
model.fit(X, y, ids=ids)
pred = model.predict(X, ids=ids)
print(pred[:3])
```

Benchmark helper for quick smoke checks:

```python
from cartoboost import benchmark_neural_vs_cartoboost

results = benchmark_neural_vs_cartoboost(
    X,
    y,
    ids=ids,
    split_ratio=0.8,
    cartoboost_kwargs={"n_estimators": 40, "learning_rate": 0.08, "max_depth": 4},
)
print(results)
```

## 8) Benchmarking and acceptance

Use the repository script:

```bash
uv run --group dev python scripts/run_neural_embedding_benchmark.py \
  --n-rows 2000 \
  --n-features 8 \
  --n-cells 128 \
  --n-neural-dim 16 \
  --include-sklearn \
  --output target/validation/neural_embedding_benchmark.json
```

The JSON report contains MAE and latency timing for:

- structured-only `cartoboost`
- `neural_embedding_hybrid`
- optional `sklearn_gbr`

A useful check is to require improvement on blocked/shifted splits, not only
random splits.

## 9) Failure modes and guardrails

- **ID drift**: ID column encoding must stay stable between training/inference.
- **Row ordering**: IDs and rows must align (same order) before transform.
- **Sparse splits**: ensure sparse-set feature configuration is identical across
  baseline and hybrid training.
- **Non-finite IDs**: missing/infinite IDs cannot be encoded.
- **Dimension mismatch**: embedding `dim` in transformer and model must match.

## 10) What changed in this implementation

- Added hybrid estimator + benchmark helper:
  - `python/cartoboost/neural/pipeline.py`
- Exported at package root:
  - `cartoboost.NeuralEmbeddingRegressor`
  - `cartoboost.benchmark_neural_vs_cartoboost`
- Added Python regression tests + benchmark script.
- Added Python API docs for this feature in:
  - `docs/reference/python-api.md`
