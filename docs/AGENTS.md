# Docs Agent Guide

## Dev environment tips
- This folder contains project documentation for API contracts, artifacts, schema behavior, sparse features, tests, benchmarks, and release readiness.
- Keep docs factual and aligned with code and tests.
- Use taxi-domain examples by default: pickup zone, dropoff zone, taxi trip, fare, duration, trip distance, hour, day of week, `PULocationID`, and `DOLocationID`.
- Use NYC taxi terminology in public docs unless a user explicitly asks for another domain.
- Keep optional dependency docs split by extra. For example, document `cartoboost[h3]`, `cartoboost[s2]`, `cartoboost[duckdb]`, and `cartoboost[polars]` separately instead of implying one bundled geo/table extra.
- When documenting benchmark results, distinguish real NYC taxi runs from synthetic smoke or acceptance fixtures. Avoid broad superiority language unless the documented command and metrics support it.
- Keep forecast benchmark model tables simple and artifact-specific. For committed M4/M5/M6 CartoBoost tables, show only committed CartoBoost rows relevant to that artifact, sorted by RMSE. Do not include full-roster rows in the same model table and do not show duplicate CartoBoost rows from multiple artifacts.
- Avoid vague public table headers such as `Scope`. Use plain, concrete headers such as `Model`, `RMSE`, `MAE`, `WAPE`, `Read`, `Artifact`, `Details`, and `Result`.
- When benchmark code changes metric output or artifact schema, update public benchmark docs only from freshly run benchmark artifacts, not from stale remembered values.

## Testing instructions
- Cross-check examples against current Python, CLI, and Rust contracts.
- Update the specific contract page when implementation behavior changes, not only the README.
- Run `npm run typecheck` and `npm run build` after navigation changes, renamed pages, public API docs changes, or Docusaurus component edits.
- Search docs before finishing terminology cleanup for old non-taxi terminology in `docs`, `README.md`, `docusaurus.config.ts`, and `sidebars.ts`.

## PR instructions
- Identify which public contract or guide changed.
- Mention any code or test changes that keep docs in sync.
- If generated docs assets are refreshed, state the generating command and why the committed outputs changed.
