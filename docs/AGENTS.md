# Docs Agent Guide

## Dev environment tips
- This folder contains project documentation for API contracts, artifacts, schema behavior, sparse features, tests, benchmarks, and release readiness.
- Keep docs factual and aligned with code and tests.
- Use taxi-domain examples by default: pickup zone, dropoff zone, taxi trip, fare, duration, trip distance, hour, day of week, `PULocationID`, and `DOLocationID`.
- Use NYC taxi terminology in public docs unless a user explicitly asks for another domain.
- Keep optional dependency docs split by extra. For example, document `cartoboost[h3]`, `cartoboost[s2]`, `cartoboost[duckdb]`, and `cartoboost[polars]` separately instead of implying one bundled geo/table extra.
- When documenting benchmark results, distinguish real NYC taxi runs from synthetic smoke or acceptance fixtures. Avoid broad superiority language unless the documented command and metrics support it.
- Keep forecast benchmark tables reader-facing: one ranked model-comparison table per dataset family when possible. If a family includes multiple artifacts, use a `Run` column instead of scattering several small tables.
- Avoid vague public table headers such as `Scope`. Use plain, concrete headers such as `Model`, `RMSE`, `MAE`, `WAPE`, `Read`, `Artifact`, `Details`, and `Result`.
- When benchmark code changes metric output or artifact schema, update public benchmark docs only from freshly run benchmark artifacts, not from stale remembered values.
- Keep public forecasting docs user-facing. Internal implementation rules such as
  Rust-first ownership, Python wrapper boundaries, and benchmark acceptance
  gates belong in AGENTS files unless they are explaining an actual public API
  contract.

## Testing instructions
- Cross-check examples against current Python, CLI, and Rust contracts.
- Update the specific contract page when implementation behavior changes, not only the README.
- Run `npm run typecheck` and `npm run build` after navigation changes, renamed pages, public API docs changes, or Docusaurus component edits.
- Search docs before finishing terminology updates for old non-taxi terminology in `docs`, `README.md`, `docusaurus.config.ts`, and `sidebars.ts`.

## PR instructions
- Identify which public contract or guide changed.
- Mention any code or test changes that keep docs in sync.
- If generated docs assets are refreshed, state the generating command and why the committed outputs changed.
