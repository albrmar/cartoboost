# Core Tests Agent Guide

## Dev environment tips
- This folder contains Rust integration and property tests for the core crate.
- Keep tests deterministic and avoid relying on generated data outside committed fixtures.

## Testing instructions
- Use `cargo test -p geoboost-core` for these tests.
- Prefer small fixtures that expose model invariants, routing behavior, serialization, or splitter edge cases.

## PR instructions
- Describe the behavior covered by new or changed tests.
- Mention any known remaining coverage gaps.
