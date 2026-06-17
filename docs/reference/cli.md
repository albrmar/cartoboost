# CLI Reference

## `train`

```sh
cartoboost train --data <csv> [--config <toml>] [--model-out <path>] [--output json|csv]
```

Trains a dense numeric CSV model. If `--model-out` is omitted, the CLI writes
`cartoboost-model.json`.

JSON output:

```json
{"ok": true, "command": "train", "rows": 100, "features": 8, "model_path": "model.json", "trees": 100}
```

## `predict`

```sh
cartoboost predict --model <path> --input <csv> [--predictions-out <path>] [--output json|csv]
```

Loads a model and predicts dense numeric rows. Prediction CSV output uses:

```csv
row,prediction
0,1.25
1,2.5
```

## `eval`

```sh
cartoboost eval --model <path> --data <csv> [--output json|csv]
```

Computes mean absolute error against the target column stored in the model, or
the last data column when the model has no target name.

## `inspect`

```sh
cartoboost inspect [--model <path>] [--config <toml>] [--data <csv>] [--output json|csv]
```

Summarizes model, config, and data inputs without training.

## Accepted Options

| Command | Options |
| --- | --- |
| `train` | `--data`, `--config`, `--model-out`, `--output`, `--help` |
| `predict` | `--model`, `--input`, `--predictions-out`, `--output`, `--help` |
| `eval` | `--model`, `--data`, `--output`, `--help` |
| `inspect` | `--model`, `--config`, `--data`, `--output`, `--help` |

Unknown options fail fast.
