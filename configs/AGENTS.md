# Configs Agent Guide

## Dev environment tips
- This folder contains example TOML training configurations for the CLI and validation scripts.
- Keep config keys aligned with `BoosterConfig`, CLI parsing, and documented examples.

## Testing instructions
- Run CLI integration tests when config behavior changes.
- Update tests and docs when accepted splitters, leaf predictors, or target conventions change.

## PR instructions
- Identify which configuration contract changed.
- Mention affected docs, examples, or validation scripts.
