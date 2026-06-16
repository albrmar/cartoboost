# Loss Agent Guide

## Dev environment tips
- This folder contains loss functions.
- Keep gradients, baseline initialization, and metric assumptions numerically stable and deterministic.

## Testing instructions
- Add numerical edge-case tests for changed loss behavior.
- If adding a loss, update configuration parsing and Python validation that assumes L2-only behavior.

## PR instructions
- State how the loss contract changed.
- Mention any docs or examples updated for new loss support.
