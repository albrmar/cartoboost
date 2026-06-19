# Neural Forecasting Experts

CartoBoost includes Rust-native deterministic CPU forecasting experts modeled after
N-BEATS and N-HiTS. The implementation lives in `crates/cartoboost-neural` and uses
fixed-initialized MLP blocks over lag windows, standard target scaling, and recursive
multi-step prediction.

These experts are intended for panel taxi demand tasks such as pickup/dropoff zone
hourly volume forecasting. Python classes in `cartoboost.forecasting.neural` are thin
configuration wrappers and do not implement model fitting or prediction in Python.

## Models

- `NBeatsForecaster` trains a deterministic MLP over contiguous lag windows.
- `NHiTSForecaster` applies deterministic average pooling to lag windows before the
  MLP block, matching the N-HiTS idea of lower-resolution history blocks.

Both models require:

- `input_size`: number of historical target values per training window.
- `hidden_size`: hidden width for the Rust-native MLP block.
- `epochs`: deterministic full-pass training iterations.
- `learning_rate`: positive gradient descent step size.

`NHiTSForecaster` also requires `pooling_size`, which must be between `1` and
`input_size`.

## Current Binding Status

The Rust implementation is covered by focused crate tests. Python wrappers fail with a
clear `NotImplementedError` if the compiled native binding does not expose the neural
forecasting classes in the current build.

No benchmark quality claims are made here. Reported neural forecasting performance
should come from real taxi-domain benchmark runs with recorded commands, fixed splits,
model settings, timing, and metric tables.
