# Tree Agent Guide

## Dev environment tips
- This folder contains tree nodes, routing, gain calculation, histograms, and tree building.
- Preserve deterministic split selection, minimum leaf constraints, and prediction routing.

## Testing instructions
- Run focused Rust tree tests after routing or builder changes.
- Run Python parity tests when predictions, serialization, or routing behavior changes.

## PR instructions
- Explain tree-building or routing behavior changes.
- Note expected updates to model outputs, fixtures, or goldens.
