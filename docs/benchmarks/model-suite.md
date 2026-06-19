# Model Benchmark Suite

## Research Question

On controlled regression problems, which feature families are responsible for
quality gains: ordinary dense numeric features, repeated-ID residual structure,
or graph source-target topology?

This study is not a production benchmark. It is a deterministic laboratory
fixture designed to isolate data conditions that are difficult to separate in
real taxi data.

## Dataset

The suite generates three synthetic workloads with 2,400 rows each and an
80/20 train/test split. The random seed is `42`.

| Workload | Dataset design | Deployment stress test |
| --- | --- | --- |
| Normal dense | IID dense numeric rows with nonlinear interactions. | Can the model handle ordinary tabular regression without ID or graph help? |
| Neural ID | Dense rows plus repeated cell IDs with residual signal attached to IDs. | Does an embedding help when IDs repeat, and does it stop helping on cold IDs? |
| Graph source-target | Directed source-target rows where node attributes and graph topology carry signal. | Does topology help beyond dense source/target columns, especially on cold source holdouts? |

## Targets

Each workload has a continuous regression target. For graph link-predictor rows,
the target is edge ranking quality, reported as AUC/AP rather than regression
error.

## Features

- Normal dense: numeric columns only.
- Neural ID: numeric columns plus repeated cell identifiers.
- Graph source-target: source node, target node, node attributes, and train
  graph topology.
- Graph augmentation rows add node2vec, GraphSAGE, HeteroGraphSAGE, or HinSAGE
  embeddings learned from train topology.

## Methods

The benchmark compares:

- Mean prediction baseline.
- Base CartoBoost.
- CartoBoost with neural ID residual embeddings.
- CartoBoost with graph-derived node features.
- Standalone neural/graph regressors and link predictors.
- XGBoost and LightGBM when optional benchmark dependencies are installed.

External baselines receive the same generated train/test rows and comparable
dense columns. Graph and embedding rows are evaluated only on workloads where
those inputs exist.

## Command

```sh
uv run --group bench python scripts/run_model_benchmark_suite.py \
  --output-dir docs/assets/model_benchmarks
```

Generated evidence:

- [Results JSON](../assets/model_benchmarks/results.json)
- [Generated table report](../assets/model_benchmarks/results.md)
- `docs/assets/model_benchmarks/mae_by_model.png`
- `docs/assets/model_benchmarks/train_time_by_model.png`
- `docs/assets/model_benchmarks/prediction_throughput_by_model.png`

## Metrics

MAE and RMSE measure error magnitude. R2 measures explained variance. Training
seconds and prediction rows per second are reported for operational context but
are not the primary quality claim.

## Results

In the latest run, the best CartoBoost-family row beats LightGBM on RMSE and R2
for every workload/split where LightGBM ran:

| workload/split | best CartoBoost-family row | CartoBoost RMSE | LightGBM RMSE | RMSE delta | R2 delta |
| --- | --- | ---: | ---: | ---: | ---: |
| normal/random | `cartoboost` | 0.4625 | 0.5080 | -0.0455 | 0.0095 |
| neural/random | `cartoboost_neural` | 0.4584 | 0.5368 | -0.0783 | 0.0265 |
| neural/group_holdout | `cartoboost` | 0.5387 | 0.5677 | -0.0290 | 0.0112 |
| graph/random | `graphsage_regressor` | 0.4495 | 0.4987 | -0.0493 | 0.0196 |
| graph/group_holdout | `cartoboost` | 0.5210 | 0.5343 | -0.0132 | 0.0057 |

![MAE by model, workload, and split](../assets/model_benchmarks/mae_by_model.png)

![Training time by model, workload, and split](../assets/model_benchmarks/train_time_by_model.png)

![Prediction throughput by model, workload, and split](../assets/model_benchmarks/prediction_throughput_by_model.png)

## Interpretation

The dense workload is the control. CartoBoost performs well there without
neural or graph inputs, so the structured rows are not masking a failure on
ordinary numeric regression.

The neural-ID workload separates repeated-ID learning from cold-ID deployment.
`cartoboost_neural` improves the random split because validation rows reuse IDs
seen during training. On group holdout, the embedding branch falls back toward
base behavior because the tested IDs are unseen. That is the desired read:
embedding gains are repeated-ID gains, not automatic cold-start generalization.

The graph workload shows two different uses of topology. Graph-augmented
CartoBoost rows append topology-derived features to a booster. Standalone graph
regressors score the graph task directly. Link predictors report AUC/AP because
they rank candidate edges rather than predict the regression target.

## Limitations

The suite is synthetic and small. It is useful for isolating mechanisms, not
for claiming production accuracy. Timing should not be compared across
machines unless hardware, dependency versions, and command settings are named.

## Reporting Requirements

- Commit `results.json`, `results.md`, and plot PNGs together.
- Keep skipped XGBoost or LightGBM rows visible when optional dependencies are
  unavailable.
- State the workload, split, target, features, and command whenever quoting a
  result.
