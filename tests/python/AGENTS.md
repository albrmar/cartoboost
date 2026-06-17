# Python Tests Agent Guide

## Dev environment tips
- This folder contains Python API tests for estimator behavior, schema handling, sparse features, SHAP, NumPy fast paths, validation artifacts, and property checks.
- Keep tests deterministic and avoid requiring optional benchmark dependencies unless the test already marks that need.

## Testing instructions
- Run relevant files with `uv run --group dev pytest tests/python/<file>.py`.
- Rebuild native bindings before tests that depend on changed PyO3 behavior.

## PR instructions
- Describe the Python behavior covered by changed tests.
- Mention native, schema, sparse, or SHAP coverage when relevant.
