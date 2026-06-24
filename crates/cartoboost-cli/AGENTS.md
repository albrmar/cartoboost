# CLI Crate Agent Guide

## Dev environment tips
- This crate provides the dense numeric CSV command-line interface.
- Keep invalid-input handling nonzero with actionable messages.
- Keep output formats predictable JSON or CSV.

## Testing instructions
- Run `cargo test -p cartoboost-cli` for focused Rust checks.
- Run relevant integration tests under `tests/integration` when command behavior changes.

## PR instructions
- Summarize changed commands, options, output fields, or error messages.
- Update examples and docs for user-visible CLI changes.
