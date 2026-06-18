# Docs Agent Guide

## Dev environment tips
- This folder contains project documentation for API contracts, artifacts, schema behavior, sparse features, tests, benchmarks, and release readiness.
- Keep docs factual and aligned with code and tests.
- Use taxi-domain examples by default: pickup zone, dropoff zone, taxi trip, fare, duration, trip distance, hour, day of week, `PULocationID`, and `DOLocationID`.
- Do not use freight, truckload, carrier, shipper, brokerage, provider, or lane-style terminology in public docs unless a user explicitly asks for that domain.
- Keep optional dependency docs split by extra. For example, document `cartoboost[h3]`, `cartoboost[s2]`, `cartoboost[duckdb]`, and `cartoboost[polars]` separately instead of implying one bundled geo/table extra.
- When documenting benchmark results, distinguish real NYC taxi runs from synthetic smoke or acceptance fixtures. Avoid broad superiority language unless the documented command and metrics support it.

## Testing instructions
- Cross-check examples against current Python, CLI, and Rust contracts.
- Update the specific contract page when implementation behavior changes, not only the README.
- Run `uv run --group dev --group docs mkdocs build --strict` after navigation changes, renamed pages, or public API docs changes.
- Search docs before finishing terminology cleanup, for example with `rg -i "freight|truck|carrier|shipper|brokerage|provider|lane" docs README.md mkdocs.yml`.

## PR instructions
- Identify which public contract or guide changed.
- Mention any code or test changes that keep docs in sync.
- If generated docs assets are refreshed, state the generating command and why the committed outputs changed.
