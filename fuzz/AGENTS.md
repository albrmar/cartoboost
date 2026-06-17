# Fuzz Agent Guide

## Dev environment tips
- This folder contains cargo-fuzz harnesses for model training, prediction, and deserialization.
- Keep fuzz targets small and focused on panic-safety and validation boundaries.

## Testing instructions
- Update fuzz targets when core APIs change so they continue compiling.
- Use `cargo fuzz` workflows from this package when validating harnesses.

## PR instructions
- Explain which input surface the fuzz target covers.
- Mention any corpus or target contract changes.
