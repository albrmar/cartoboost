# CLI

The `geoboost` CLI is a dense numeric CSV interface for training, prediction,
evaluation, and artifact inspection. Sparse-set route features are supported by
the Python API, not by the CLI v1 path.

## Commands

```sh
geoboost train --data <csv> [--config <toml>] [--model-out <path>] [--output json|csv]
geoboost predict --model <path> --input <csv> [--predictions-out <path>] [--output json|csv]
geoboost eval --model <path> --data <csv> [--output json|csv]
geoboost inspect [--model <path>] [--config <toml>] [--data <csv>] [--output json|csv]
```

During development, run the binary through Cargo:

```sh
cargo run -p geoboost-cli -- train --data train.csv --config config.toml --model-out model.json
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
| `loss` | `l2`, `squared_error`, `quantile`, or `pinball`. |
| `quantile_alpha` | Quantile alpha in `(0, 1)`. |
| `splitter` or `splitters` | Comma-separated splitter names. |
| `leaf_predictor` | `constant` or `linear`. |
| `fuzzy` | Boolean fuzzy routing flag. |
| `fuzzy_bandwidth` | Non-negative fuzzy transition bandwidth. |
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
splitter = "axis,periodic_24"
leaf_predictor = "constant"
```

## Output

`--output json` is the default and is intended for scripts. `--output csv`
prints tabular command output. `predict --predictions-out` writes a CSV with
`row,prediction` columns.

## Failure Policy

The CLI exits nonzero for malformed options, unknown config keys, invalid
splitters or leaf predictors, missing targets, numeric parse failures, and
feature-count mismatches. Error messages are prefixed with `geoboost:`.
