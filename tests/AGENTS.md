# Tests Agent Guide

## Dev environment tips
- This folder contains Python, integration, fixture contract, and parity tests.
- Keep tests deterministic and prefer committed fixtures or synthetic data over external downloads.

## Testing instructions
- Run `uv run --group dev pytest` for the full Python test suite.
- For focused work, run the relevant test file first.
- Update tests for public behavior changes.

## PR instructions
- Describe what behavior new or changed tests cover.
- Mention any fixtures, goldens, or parity outputs that changed.
