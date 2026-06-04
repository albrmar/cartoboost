# GitHub Configuration Agent Guide

## Dev environment tips
- This folder contains repository automation and GitHub configuration.
- Keep workflows aligned with the validation commands documented in the root README and `justfile`.
- Avoid adding secrets, machine-specific paths, or generated artifacts.

## Testing instructions
- Validate workflow YAML syntax after edits.
- Make sure CI commands work from a clean checkout with Rust, Python, `uv`, and maturin setup.

## PR instructions
- Describe which jobs or automation behavior changed.
- Mention whether workflow changes mirror local validation or intentionally differ.
