# Tests Agent Guide

## Dev environment tips
- This folder contains Python, integration, fixture contract, and parity tests.
- Keep tests deterministic and prefer committed fixtures or synthetic data over external downloads.
- Use taxi-style fixture names for public-facing examples and tests that may be copied into docs: pickup/dropoff, taxi trip, fare, duration, and taxi zones.
- Optional dependency tests should use `pytest.importorskip` when the package is not in the relevant dependency group, and should include at least one path that verifies hard-fail behavior when a required optional package is absent.

## Testing instructions
- Run `uv run pytest` for the full Python test suite.
- For focused work, run the relevant test file first.
- Update tests for public behavior changes.
- For input adapter work, compare optional input objects against a NumPy or dict baseline with the same model configuration.
- For benchmark-related changes, tests should verify artifact shape and guardrails, not only that a script exits.

## PR instructions
- Describe what behavior new or changed tests cover.
- Mention any fixtures, goldens, or parity outputs that changed.
