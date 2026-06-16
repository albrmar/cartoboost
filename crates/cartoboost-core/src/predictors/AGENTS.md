# Predictors Agent Guide

## Dev environment tips
- This folder contains leaf predictors, including constant and linear leaves.
- Preserve predictor serialization and prediction semantics so saved artifacts remain loadable.

## Testing instructions
- Verify regularization, feature-index handling, and prediction behavior after linear leaf changes.
- Run Python tests that exercise `linear_leaf_features`.

## PR instructions
- Describe predictor behavior and artifact compatibility impact.
- Mention any changed defaults or configuration requirements.
