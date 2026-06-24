# Benchmarks Agent Guide

## Dev environment tips
- This folder contains Criterion benchmark scaffolding and benchmark summaries.
- Keep benchmark code focused on data loading, training, prediction, and serialization measurements.
- Do not update generated summary images or JSON unless the task is specifically about benchmark artifacts.

## Testing instructions
- Use `cargo bench --workspace --no-run` for a compilation check.
- Run full benchmarks only when benchmark results are part of the task.

## PR instructions
- State whether benchmark code, generated summaries, or both changed.
- Explain any benchmark artifact refresh and the command used to produce it.
