# CLI Source Agent Guide

## Dev environment tips
- This folder contains the CLI implementation.
- Keep parsing, validation, file IO, and output formatting simple and explicit.
- Avoid adding dependencies unless they meaningfully reduce risk.

## Testing instructions
- Run `cargo test -p cartoboost-cli`.
- Run `uv run pytest tests/integration/test_cli_train_predict_eval.py tests/integration/test_cli_invalid_inputs.py` for CLI contract changes.

## PR instructions
- Call out changed command behavior and compatibility impact.
- Include before/after output examples when output shape changes.
