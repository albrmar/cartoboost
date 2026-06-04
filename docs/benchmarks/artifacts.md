# Benchmark Artifact Policy

Generated outputs should make behavior changes visible without turning the
repository into a benchmark cache.

## Commit

Commit these only when the benchmark or validation task explicitly asks for an
artifact refresh:

| Path | Contents |
| --- | --- |
| `docs/assets/segmentation_*.png` | High-level segmentation proof images. |
| `docs/assets/splitter_tests/*.png` | Splitter acceptance proof images. |
| `docs/assets/lane_level_tests/acceptance_metrics.*` | Lane-level acceptance metrics. |
| `docs/assets/lane_level_tests/*.png` | Lane-level diagnostic plots. |
| `docs/assets/nyc_taxi_benchmarks/results.json` | Single-run NYC benchmark machine output. |
| `docs/assets/nyc_taxi_benchmarks/results.md` | Single-run NYC benchmark report. |
| `docs/assets/nyc_taxi_benchmarks/repeated_results.json` | Repeated NYC summary machine output. |
| `docs/assets/nyc_taxi_benchmarks/repeated_results.md` | Repeated NYC summary report. |
| `docs/assets/nyc_taxi_benchmarks/*.png` | Top-level NYC summary charts. |
| `docs/assets/nyc_taxi_benchmarks/plots/*.png` | Per-task NYC plots. |

## Do Not Commit

| Path | Reason |
| --- | --- |
| `target/criterion/` | Raw Criterion output is local and noisy. |
| `target/validation/` | Generated validation workspace. |
| `target/nyc_taxi_repeated/` | Per-run NYC repeated outputs; summaries are committed separately. |
| `data/nyc_taxi/` | Downloaded TLC Parquet cache. |
| `.venv/`, `.pytest_cache/`, `.ruff_cache/` | Local environment and tool caches. |

## Refresh Checklist

1. Run the exact generator documented by the relevant benchmark page.
2. Inspect markdown, JSON, and plots together; do not update only one format.
3. Record the command, data source, sample size, model settings, and reason for
   refresh in the PR.
4. Keep generated directory README files short. Maintained runbooks live under
   `docs/benchmarks/`.

## Artifact Naming

- `results.*` means one benchmark run.
- `repeated_results.*` means a repeated-run aggregate.
- `*_summary.png` means a top-level chart generated from `results.json`.
- `plots/*` means per-task diagnostic plots.
