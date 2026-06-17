# Core Source Agent Guide

## Dev environment tips
- This folder contains the core Rust library modules.
- Keep module boundaries clear across data normalization, booster orchestration, splitters, tree building, prediction, explanation, and serialization.
- Keep public types compatible with CLI and PyO3 callers.

## Testing instructions
- Run `cargo test -p cartoboost-core`.
- Update and run PyO3, CLI, or Python tests when shared public APIs change.

## PR instructions
- State which modules changed and why.
- Mention downstream changes required in CLI, Python bindings, tests, or docs.
