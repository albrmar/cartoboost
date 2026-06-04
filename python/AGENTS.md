# Python Source Agent Guide

## Dev environment tips
- This folder contains the Python package source root.
- Keep the public package importable through `pythonpath = ["python"]`.
- Keep layout compatible with maturin's `python-source` setting.

## Testing instructions
- Run `uv run --group dev ruff format --check python`.
- Run relevant pytest files after Python package changes.

## PR instructions
- Summarize changed Python API surface.
- Mention whether native binding rebuilds are required.
