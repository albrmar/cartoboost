# Integration Tests Agent Guide

## Dev environment tips
- This folder contains integration tests for CLI behavior, validation scripts, parity, and cross-surface contracts.
- Keep tests focused on user-visible behavior and failure messages.

## Testing instructions
- Use `uv run pytest tests/integration/<file>.py` while iterating.
- Run broader pytest when contracts change.

## PR instructions
- Describe the cross-surface contract under test.
- Mention whether CLI, Python, Rust, or scripts are involved.
