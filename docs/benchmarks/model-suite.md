# Model Benchmark Suite

## Bottom Line

The model suite is a synthetic diagnostic. It verifies that dense splitters,
repeated-ID residual features, and graph features are wired and measurable
before spending time on real taxi benchmarks.

The current maintained run uses seed 42, 2,400 rows, and an 80/20 split. It is
not a public model-superiority benchmark because it is synthetic and reports a
best-of-CartoBoost-family comparison.

## Reproduce

```sh
uv run --group bench python scripts/run_model_benchmark_suite.py \
  --output-dir docs/assets/model_benchmarks
```

Artifacts:

- `docs/assets/model_benchmarks/results.json`
- `docs/assets/model_benchmarks/results.md`
- `docs/assets/model_benchmarks/mae_by_model.png`
- `docs/assets/model_benchmarks/train_time_by_model.png`
- `docs/assets/model_benchmarks/prediction_throughput_by_model.png`

## Workload Breakdown

| Workload / split | Best diagnostic row | RMSE | LightGBM RMSE | R2 delta vs LightGBM | Read |
| --- | --- | ---: | ---: | ---: | --- |
| Dense / random | `cartoboost` | 0.4625 | 0.5080 | +0.0095 | Dense numeric control passes. |
| Repeated-ID / random | `cartoboost_neural` | 0.4584 | 0.5368 | +0.0265 | Embeddings help when IDs recur. |
| Repeated-ID / group holdout | `cartoboost` | 0.5387 | 0.5677 | +0.0112 | Cold-ID split favors base behavior. |
| Graph / random | `graphsage_regressor` | 0.4495 | 0.4987 | +0.0196 | Graph signal is learnable in fixture. |
| Graph / group holdout | `cartoboost` | 0.5210 | 0.5343 | +0.0057 | Group holdout reduces graph advantage. |

## Plots

![MAE by model and workload](../assets/model_benchmarks/mae_by_model.png)

![Training time by model and workload](../assets/model_benchmarks/train_time_by_model.png)

![Prediction throughput by model and workload](../assets/model_benchmarks/prediction_throughput_by_model.png)

## Interpretation

Use this page to diagnose mechanisms:

- Dense numeric behavior should be healthy before adding structure.
- Repeated-ID gains should shrink or disappear on cold-ID holdouts.
- Graph rows should only help when the target contains graph topology.

Do not use this suite to claim that CartoBoost is generally better than
LightGBM or XGBoost. That claim requires public datasets, repeated splits,
equal HPO budgets, and uncertainty estimates.

