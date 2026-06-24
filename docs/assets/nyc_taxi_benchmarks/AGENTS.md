# NYC Taxi Benchmark Assets Agent Guide

## Dev environment tips
- This folder contains NYC taxi benchmark reports and summaries.
- Keep result JSON, markdown summaries, and top-level benchmark images aligned.
- Forecast benchmark artifact refreshes must come from the documented scripts. If CartoBoost forecasting code changes, rerun the affected committed CartoBoost benchmarks before updating their JSON, plots, or benchmark markdown.
- Do not mix rows from separate artifacts into one committed CartoBoost model table. Committed M4/M5/M6 model tables should contain only the committed CartoBoost rows for that artifact, sorted by RMSE.

## Testing instructions
- Do not refresh benchmark results casually.
- Use the documented benchmark scripts when benchmark artifacts intentionally change.

## PR instructions
- Explain benchmark environment, command, and reason for refreshed results.
- Note whether plots and summaries were updated together.
