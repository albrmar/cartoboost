# Scripts Agent Guide

## Dev environment tips
- This folder contains validation, benchmark, plotting, synthetic data, and report scripts.
- Keep scripts runnable through `uv run --group dev python scripts/<name>.py` unless a benchmark dependency group is required.
- Avoid hard-coded machine paths.

## Testing instructions
- Run the specific script after changes.
- Run validation script tests under `tests/integration` or `tests/python` when script contracts change.

## PR instructions
- Explain generated outputs and where they are written.
- Mention whether outputs belong under `target/` or committed asset folders.
