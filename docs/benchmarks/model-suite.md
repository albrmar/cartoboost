# Model Benchmark Suite

## What It Tests

The model benchmark suite verifies CartoBoost mechanisms on controlled
workloads before using those mechanisms in real-data benchmarks.

Do not use this page for broad model-superiority claims. Synthetic fixtures
answer mechanism questions, not deployment questions.

## Diagnostic Questions

| Workload | Question |
| --- | --- |
| Dense numeric regression | Does the base booster handle ordinary nonlinear tabular signal? |
| Repeated-ID regression | Do residual embeddings help only when IDs recur, and do they degrade honestly on cold IDs? |
| Directed graph regression | Do graph-derived features help when source-target topology is part of the data-generating mechanism? |

## Report

Report:

- generator version and seed list;
- row count, target definition, and feature families;
- split type, including random, grouped, cold-ID, or cold-edge variants;
- every model family attempted and every skipped optional dependency;
- MAE, RMSE, R2, fit time, prediction time, and prediction throughput;
- repeated seeds when a result is used to justify a design decision.

Do not compare the best CartoBoost-family row against a single baseline row as
a public claim. If a report includes a "best-of-family" summary, label it as
model exploration and also report the individual rows.

## Reproduce

```sh
uv run --group bench python scripts/run_model_benchmark_suite.py \
  --output-dir docs/assets/model_benchmarks
```

Commit generated artifacts only when they are intentional evidence. Temporary
runs should write to `target/`.

## Allowed Conclusions

Allowed:

- "The repeated-ID fixture shows embedding gains only on repeated-ID splits."
- "The graph fixture detects whether train-side topology features are wired."
- "The dense fixture catches regressions in ordinary tabular behavior."

Not allowed:

- "CartoBoost is generally preferable to LightGBM or XGBoost."
- "The best CartoBoost-family row proves a public benchmark claim."
- "Synthetic timing predicts production throughput."

Use this suite to decide which model families deserve real-data runs. Do not
use it as evidence for real-world accuracy.
