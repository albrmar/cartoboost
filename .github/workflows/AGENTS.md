# Workflow Agent Guide

## Dev environment tips
- This folder contains GitHub Actions workflows.
- Keep jobs deterministic and mirror local validation where practical.
- Prefer explicit setup for Rust, Python, `uv`, and maturin.

## Testing instructions
- Validate YAML syntax after workflow edits.
- Compare changed commands against the root `justfile` and README validation sequence.

## PR instructions
- Summarize affected jobs and triggers.
- Call out any intentionally skipped local checks or CI-only requirements.
