# Splitters Agent Guide

## Dev environment tips
- This folder contains splitters for axis, histogram, diagonal 2D, Gaussian/radial 2D, periodic, sparse-set, and fuzzy behavior.
- Keep splitter names and aliases aligned with Python validation and CLI config parsing.

## Testing instructions
- Add targeted tests for routing, gain, schema handling, empty sparse rows, and periodic wraparound.
- Run parity and Python tests when splitter changes alter training outputs.

## PR instructions
- Explain expected model-output changes.
- Mention updates to docs, configs, validation artifacts, or goldens.
