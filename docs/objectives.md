# Objectives

CartoBoost supports regression objectives for continuous temporal-spatial targets
such as demand, duration, fare, cost, or residual error.

## L2 Regression

L2 squared error is the default objective:

```python
model = CartoBoostRegressor(loss="l2")
```

Use L2 when your primary score is RMSE, R2, or general point-prediction quality.
It supports the current splitters, sample weights, constant leaves, and linear
leaves.

## Quantile Regression

Quantile regression estimates a conditional quantile instead of the conditional
mean:

```python
model = CartoBoostRegressor(
    loss="quantile",
    quantile_alpha=0.9,
    leaf_predictor="constant",
)
```

Use quantile regression when upper or lower tails matter, such as high-delay ETA
risk, high-cost trip estimates, or conservative demand forecasts.

Accepted names:

| Objective | Names |
| --- | --- |
| L2 squared error | `"l2"`, `"squared_error"` |
| Quantile pinball | `"quantile"`, `"pinball"` |

`quantile_alpha` must be in `(0, 1)`. Quantile loss currently requires
constant leaves.

## Evaluation Guidance

- Use RMSE, MAE, and R2 for L2 point prediction.
- Use pinball loss for quantile models.
- Report the holdout type: random, temporal, spatial, grouped, or route-based.
- Compare objectives on the same split and feature set.
- For temporal-spatial work, include residual summaries by time bucket, zone,
  route cell, or lane; aggregate metrics can hide localized failure modes.
