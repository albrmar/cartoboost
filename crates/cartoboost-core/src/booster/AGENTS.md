# Booster Agent Guide

## Dev environment tips
- This folder contains gradient boosting orchestration for fitting and prediction.
- Preserve learning-rate semantics, sample-weight handling, fuzzy training behavior, and residual updates.

## Testing instructions
- Add or update focused Rust tests for training behavior changes.
- Run parity or integration tests when Python-facing predictions may change.

## PR instructions
- Explain expected model-output changes.
- Note any impact on validation artifacts, goldens, or parity fixtures.
