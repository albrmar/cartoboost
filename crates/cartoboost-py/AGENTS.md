# PyO3 Crate Agent Guide

## Dev environment tips
- This crate exposes the Rust core through PyO3/maturin as `cartoboost._native`.
- Keep Python-facing errors clear and compatible with the Python wrapper expectations.

## Testing instructions
- Run `uv run --group dev maturin develop` after binding changes.
- Run relevant Python tests under `tests/python`.

## PR instructions
- Summarize changed Python-native methods, attributes, or error behavior.
- Mention Python wrapper and docs updates.
