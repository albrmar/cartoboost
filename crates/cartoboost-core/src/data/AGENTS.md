# Data Agent Guide

## Dev environment tips
- This folder contains dataset construction, feature schema handling, and sparse-set normalization.
- Keep validation strict and error messages clear because Rust, Python, and CLI training paths depend on this layer.

## Testing instructions
- Test dense, periodic, and sparse feature handling after data-layer changes.
- Update Python tests for native sparse/schema behavior when needed.

## PR instructions
- Describe schema, sparse-set, or dense data contract changes.
- Update schema docs for public behavior changes.
