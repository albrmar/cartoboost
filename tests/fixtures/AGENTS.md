# Fixtures Agent Guide

## Dev environment tips
- This folder contains committed test fixtures such as GeoJSON and parity inputs.
- Treat fixtures as API contracts for tests.

## Testing instructions
- Only change fixtures when the expected contract changes.
- Run tests that consume changed fixtures.

## PR instructions
- Explain why fixture shape or values changed.
- Update tests or docs that depend on the old fixture.
