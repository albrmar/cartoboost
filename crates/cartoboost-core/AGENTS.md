# Core Crate Agent Guide

## Dev environment tips
- This crate owns core training, prediction, splitters, tree construction, metrics, serialization, and model artifacts.
- Python and CLI surfaces should delegate advanced behavior here rather than duplicating logic.
- Preserve deterministic behavior and artifact compatibility.

## Testing instructions
- Run `cargo test -p cartoboost-core` for focused checks.
- Run broader workspace and Python parity tests when public behavior changes.

## PR instructions
- Explain prediction, training, serialization, or splitter behavior changes clearly.
- Note whether saved artifacts remain backward compatible.
