# PyO3 Source Agent Guide

## Dev environment tips
- This folder contains PyO3 binding code.
- Keep conversion between NumPy arrays, Python lists, sparse offsets, sparse IDs, feature schemas, and Rust datasets explicit and validated.

## Testing instructions
- From the repository root, rebuild with `uv run maturin develop`.
- Run relevant tests under `tests/python` after method signature or conversion changes.

## PR instructions
- Call out changed native signatures, attributes, or conversion rules.
- Update `python/cartoboost`, tests, and docs for compatibility.
