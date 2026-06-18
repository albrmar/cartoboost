# Model Benchmark Suite

The model benchmark suite is the maintained synthetic comparison for dense,
neural-ID, and graph-feature regression workloads. It complements the NYC taxi
benchmarks by keeping the data generation deterministic and small enough for
local iteration while still exercising CartoBoost, XGBoost, LightGBM, neural
embedding augmentation, and graph augmentation. The graph workload includes
separate CartoBoost rows for node2vec, GraphSAGE, HeteroGraphSAGE, and HinSAGE.

## Latest Run

Command:

```sh
uv run --group dev --group bench python scripts/run_model_benchmark_suite.py \
  --output-dir docs/assets/model_benchmarks
```

Generated evidence:

- [Results JSON](../assets/model_benchmarks/results.json)
- [Generated table report](../assets/model_benchmarks/results.md)
- `docs/assets/model_benchmarks/mae_by_model.png`
- `docs/assets/model_benchmarks/train_time_by_model.png`
- `docs/assets/model_benchmarks/prediction_throughput_by_model.png`

## Workloads

| Workload | Purpose | Split coverage |
| --- | --- | --- |
| Normal dense | IID numeric regression with nonlinear dense feature interactions. | Random |
| Neural ID | Dense regression with repeated cell IDs and an ID-specific residual signal. | Random and cold group holdout |
| Graph source-target | Directed source-target regression where node features and topology carry signal. | Random and cold source holdout |

The suite reports MAE, RMSE, R2, training seconds, and prediction rows per
second for each model that can run in the current environment. XGBoost and
LightGBM are optional benchmark dependencies; missing packages are recorded as
skipped rows rather than causing the whole suite to fail.

## Result Images

![MAE by model, workload, and split](../assets/model_benchmarks/mae_by_model.png)

![Training time by model, workload, and split](../assets/model_benchmarks/train_time_by_model.png)

![Prediction throughput by model, workload, and split](../assets/model_benchmarks/prediction_throughput_by_model.png)

## Interpretation

On the latest full run, the best CartoBoost-family row beats LightGBM on RMSE
and R2 for every workload/split in the suite:

| workload/split | best CartoBoost-family row | CartoBoost RMSE | LightGBM RMSE | RMSE delta | R2 delta |
| --- | --- | ---: | ---: | ---: | ---: |
| normal/random | `cartoboost` | 0.4625 | 0.5080 | -0.0455 | 0.0095 |
| neural/random | `cartoboost_neural` | 0.4584 | 0.5368 | -0.0783 | 0.0265 |
| neural/group_holdout | `cartoboost` | 0.5387 | 0.5677 | -0.0290 | 0.0112 |
| graph/random | `graphsage_regressor` | 0.4495 | 0.4987 | -0.0493 | 0.0196 |
| graph/group_holdout | `cartoboost` | 0.5210 | 0.5343 | -0.0132 | 0.0057 |

The normal dense workload is a baseline sanity check. CartoBoost wins there
without relying on graph or neural inputs, so the structured rows are not hiding
a regression on ordinary numeric data.

The neural-ID workload separates repeated-ID learning from cold-ID deployment.
`cartoboost_neural` improves the random split because validation rows reuse IDs
seen during training. On the group holdout split, the embedding branch falls
back to the base CartoBoost behavior instead of pretending unseen IDs can be
recovered from an embedding table. Treat this as a guardrail: neural ID features
should be reported with the split protocol, not as a universal quality
improvement.

The maintained `cartoboost_neural` row uses out-of-fold residual embeddings and
support-aware shrinkage, and the API also supports hierarchical `fallback_ids`,
multi-key 2D `ids`, and graph-aware `neighbor_ids`. Any claimed improvement
should still be validated separately on repeated-ID and cold-ID splits because
those are different deployment promises.

The graph workload has two surfaces. Augmented CartoBoost rows fit node2vec,
GraphSAGE, HeteroGraphSAGE, and HinSAGE features from train topology, then
append source and target embeddings to CartoBoost inputs. Standalone graph rows
score the graph task directly; `graphsage_regressor` is the best random-split
regression row, and the link predictors report AUC/AP because they rank
candidate source-target edges rather than predict the regression target.

The core difference from LightGBM is representational, not hyperparameter
search. LightGBM receives dense columns. CartoBoost-family rows can add ID
residual structure, graph topology, and source-target structure when the
benchmark exposes those contracts.

## Reproducibility Rules

- Commit `results.json`, `results.md`, and all plot PNGs together.
- State the exact command and row count used for generated evidence.
- Do not compare timing numbers across machines without naming the machine and
  dependency versions.
- Keep skipped XGBoost or LightGBM rows in the report when optional benchmark
  packages are unavailable.
