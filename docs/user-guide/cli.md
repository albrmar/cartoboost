# CLI

The `cartoboost` CLI is a dense numeric CSV interface for quick training,
prediction, evaluation, and artifact inspection. It is useful for simple
baselines and scripted checks. Use the Python API for list-valued sparse route
features, feature schemas, SHAP explanations, and richer temporal-spatial
workflows.

## Commands

Install from PyPI to get both the Python estimator and the `cartoboost` command:

```sh
uv add cartoboost
```

```sh
cartoboost train --data <csv> [--config <toml>] [--model-out <path>] [--output json|csv]
cartoboost predict --model <path> --input <csv> [--predictions-out <path>] [--output json|csv]
cartoboost eval --model <path> --data <csv> [--output json|csv]
cartoboost inspect [--model <path>] [--config <toml>] [--data <csv>] [--output json|csv]
```

## Input CSV

- The first non-empty row is the header.
- All feature cells must parse as `f64`.
- Training data contains a target column.
- Prediction input contains exactly the number of feature columns expected by
  the model.

If `target` is not configured for training, the CLI uses the last CSV column as
the target.

## Configuration

The config file is a simple TOML-like `key = value` file. Supported keys:

| Key | Notes |
| --- | --- |
| `target` or `target_column` | Target column name. |
| `n_estimators` | Boosting rounds. |
| `learning_rate` | Tree shrinkage. |
| `max_depth` | Maximum tree depth. |
| `min_samples_leaf` | Minimum rows per leaf. |
| `min_gain` | Minimum gain required to split. |
| `loss` | `l2`, `squared_error`, `l1`, `mae`, `absolute_error`, `quantile`, or `pinball`. |
| `quantile_alpha` | Quantile alpha in `(0, 1)`. |
| `splitter` or `splitters` | Comma-separated splitter names, including `auto`, `axis`, `axis_histogram:<bins>`, spatial, periodic, and sparse splitters. |
| `leaf_predictor` | `constant` or `linear`. |
| `fuzzy` | Boolean fuzzy routing flag. |
| `fuzzy_bandwidth` | Non-negative fuzzy transition bandwidth. |
| `fuzzy_kernel` | `linear`, `gaussian`, `exponential`, `bisquare`, `epanechnikov`, or `tricube`. |
| `l2_regularization` | Linear-leaf ridge penalty. |
| `monotonic_constraints` | Comma-separated `-1`, `0`, and `1` values. |

Example:

```toml
target = "fare"
n_estimators = 100
learning_rate = 0.05
max_depth = 4
min_samples_leaf = 20
loss = "l2"
splitter = "auto"
leaf_predictor = "constant"
fuzzy_kernel = "linear"
```

For a dense temporal-spatial CSV, include periodic or spatial splitters in the
config:

```toml
target = "demand"
splitter = "axis,diagonal_2d,gaussian_2d,periodic_24"
```

Sparse route-cell columns are not represented in CLI CSV input; train those
models with `CartoBoostRegressor`.

## Output

`--output json` is the default and is intended for scripts. `--output csv`
prints tabular command output. `predict --predictions-out` writes a CSV with
`row,prediction` columns.

## Failure Policy

The CLI exits nonzero for malformed options, unknown config keys, invalid
splitters or leaf predictors, missing targets, numeric parse failures, and
feature-count mismatches. Error messages are prefixed with `cartoboost:`.
