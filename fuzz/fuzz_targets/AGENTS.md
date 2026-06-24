# Fuzz Targets Agent Guide

## Dev environment tips
- This folder contains individual fuzz target entrypoints.
- Avoid expensive setup in fuzz loops.
- Keep generated inputs bounded enough for useful fuzzing.

## Testing instructions
- Compile targets through the `fuzz` package before relying on them.
- Run the relevant `cargo fuzz run <target>` command for behavior checks.

## PR instructions
- Name the fuzz target changed.
- Explain new coverage or validation boundaries.
