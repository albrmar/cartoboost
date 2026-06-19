# NYC Taxi Benchmarks

## What It Tests

This benchmark tests taxi-specific structure on real TLC fare, duration, and
demand tasks under splits that resemble deployment.

## Data

Use public NYC TLC trip records and taxi-zone assets. A report should record:

- TLC source URL and extraction date;
- taxi type, year, and months;
- raw row count and cleaned row count;
- filters and dropped-row reasons;
- zone lookup version and join rules;
- target transforms for fare, duration, or demand;
- content hashes for the materialized benchmark tables;
- whether raw files are local-only under `data/` or published as benchmark
  artifacts.

Runs with `--no-download` should fail when required real inputs are missing.
Silent fallback to synthetic data is not acceptable for real-data claims.

## Tasks

| Task | Unit | Primary deployment risk |
| --- | --- | --- |
| Fare regression | Individual taxi trip. | New pickup/dropoff geography and route mix. |
| Duration regression | Individual taxi trip. | New pickup/dropoff geography, hour effects, and congestion regimes. |
| Pickup or lane demand | Zone-time or lane-time bucket. | Future periods, sparse lanes, and cold or rare zones. |

## Splits

Keep split results separate:

- IID or random split for interpolation.
- Spatial/block split for held-out pickup zones or geographic cells.
- Group/cold-route split for unseen lanes or source-target pairs.
- Out-of-time split for future taxi periods.

Model selection should use train/validation data only under the same split
family as the final evaluation. A spatial claim needs spatial validation during
tuning, not only a spatial final test.

## Baselines

At minimum, compare CartoBoost with a GBDT baseline that receives equivalent
train-only spatial features.

Feature access should be symmetric:

- Train-only target encodings are appended to every eligible model family.
- Zone, hour, distance, route, airport, and borough features are made available
  to every model that can consume them.
- Identify Graph or sparse-set features that are CartoBoost-specific and
  evaluate them as model-family capability, not hidden preprocessing.

## Metrics

Primary metrics are RMSE and MAE on the benchmark target. Report R2 only when
it is meaningful for the target and split. Also include:

- borough and airport/non-airport slices;
- rare-zone or rare-lane slices;
- cold-zone or cold-route slices when present;
- fit time, prediction time, prediction throughput, and peak memory;
- repeatability or uncertainty evidence when available.

## Reproduce

Run the current implementation script with:

```sh
PYTHONPATH=python uv run --group bench python \
  scripts/run_nyc_taxi_quality_benchmarks.py \
  --no-download \
  --output-dir docs/assets/nyc_taxi_benchmarks
```

That command alone is not sufficient for a new claim. The report also needs the
split, tuning budget, baselines, slices, and repeatability evidence.

## Reporting Rule

Do not publish a table that compares the best CartoBoost-family variant against
a single LightGBM or XGBoost row. Publish the rows tested, the baseline set,
and the model-selection budget.
