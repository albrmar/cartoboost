# Probabilistic Forecasting

CartoBoost probabilistic forecasting keeps scoring behavior in Rust and exposes
Python helpers for array conversion and ergonomic validation. Use these helpers
for taxi-demand, fare, duration, and trip-distance forecast distributions when
the experiment needs calibrated intervals or ordered rank probabilities.

## Quantiles

Quantile forecasts must use finite levels in `(0, 1)` and strictly increasing
level order. Predicted quantile values can be repaired with monotone cumulative
maximum logic so lower quantiles do not exceed higher quantiles at the same
horizon:

```python
from cartoboost.forecasting.probabilistic import repair_non_crossing_quantiles

repair_non_crossing_quantiles([12.0, 10.0, 13.0]).tolist()
# [12.0, 12.0, 13.0]
```

Pinball loss is the mean asymmetric quantile loss for aligned actual and
predicted arrays:

```python
from cartoboost.forecasting.probabilistic import pinball_loss

pinball_loss([10.0, 20.0], [11.0, 18.0], 0.5)
# 0.75
```

## Conformal Calibration

Split-conformal intervals require explicit train, calibration, and test
boundaries. CartoBoost rejects overlapping boundaries so calibration residuals
cannot include rows from the training or test window.

```python
from cartoboost.forecasting.probabilistic import ConformalCalibrator

calibrator = ConformalCalibrator(alpha=0.1).fit(
    calibration_actual=[10.0, 20.0],
    calibration_prediction=[11.0, 18.0],
    train_end_exclusive=1000,
    calibration_start=1000,
    calibration_end_exclusive=1200,
    test_start=1200,
)
interval = calibrator.predict_interval([30.0, 40.0], test_start=1200)
```

The returned interval is symmetric around the point prediction using the
finite-sample conformal absolute-residual quantile. The inputs must be finite,
one-dimensional, and exactly aligned.

## Rank Probability Score

Rank probability score evaluates an ordered categorical distribution, such as
forecasted return rank buckets for M6-style tasks. Probabilities must be finite,
non-negative, and sum to one. `observed_rank` is a zero-based index.

```python
from cartoboost.forecasting.probabilistic import rank_probability_score

rank_probability_score([0.2, 0.3, 0.5], observed_rank=2)
# 0.145
```

The implementation scores cumulative probability differences across ordered
rank cutoffs, then averages by the number of cutoffs.
