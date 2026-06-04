# Python Package Agent Guide

## Dev environment tips
- This folder contains the public Python API, sklearn-compatible regressor, schema helpers, IO helpers, and SHAP integration.
- Keep estimator training and prediction native-extension-only.

## Testing instructions
- Run relevant tests under `tests/python` after estimator, schema, IO, or explanation changes.
- Rebuild native bindings when Python changes depend on `geoboost._native`.

## PR instructions
- Describe changed estimator parameters, fit/predict behavior, persistence, schema handling, or SHAP behavior.
- Update docs and native bindings when needed.
