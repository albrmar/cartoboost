# Serialization Agent Guide

## Dev environment tips
- This folder contains model artifact serialization and related tests.
- Preserve versioned JSON compatibility unless the task explicitly changes the artifact contract.

## Testing instructions
- Run load/save tests across Rust, CLI, and Python after serialized field changes.
- Update fixture contracts and golden files only for intentional artifact changes.

## PR instructions
- Explain artifact schema changes and compatibility.
- Update `docs/model_artifact.md` for user-visible serialized fields.
