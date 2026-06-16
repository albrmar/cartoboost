# Metrics Agent Guide

## Dev environment tips
- This folder contains model evaluation metrics.
- Keep metric names, definitions, and edge-case behavior consistent across Rust, CLI output, and Python validation scripts.

## Testing instructions
- Add tests for zero-length inputs, invalid values, and numerical edge cases when metric behavior changes.
- Run validation script tests when metric outputs change.

## PR instructions
- Explain metric definition changes.
- Note downstream updates to reports, docs, or validation scripts.
