# NYC Taxi Benchmarks

## Research Question

On real NYC taxi data, do geographic and temporal feature families improve
prediction quality for trip duration, fare amount, and pickup-zone demand under
both random validation and pickup-zone spatial holdout?

The study is designed to answer where the signal comes from. Fare and duration
are trip-row targets. Pickup demand is a zone-time target. Those are different
prediction problems and should not be interpreted as one generic benchmark.

## Dataset

The maintained single-run artifact uses cached NYC TLC yellow taxi trip records
for January 2024:

```sh
PYTHONPATH=python uv run --group bench python \
  scripts/run_nyc_taxi_quality_benchmarks.py \
  --no-download \
  --output-dir docs/assets/nyc_taxi_benchmarks
```

The command uses `--no-download`. If the real Parquet file is missing, the run
fails instead of falling back to synthetic data. Raw TLC files are cached under
`data/nyc_taxi/` and are not committed.

## Targets

| Task | Target | Unit of observation |
| --- | --- | --- |
| `duration` | Log trip duration. | Individual taxi trip. |
| `fare` | Log total fare amount. | Individual taxi trip. |
| `pickup_demand` | Log pickup trip count. | Pickup zone x hour x weekday bucket. |

Quality metrics are computed on transformed regression targets.

## Features

All learned models receive comparable cleaned trip, zone, passenger, and time
features. The maintained comparison uses `--zone-treatment target_mean`, which
adds train-only smoothed pickup/dropoff zone target-mean features to every
model, including XGBoost and LightGBM. This prevents the comparison from giving
CartoBoost exclusive access to zone-level target context.

CartoBoost-family rows may additionally use feature families that match the
task structure:

- Periodic hour and weekday features.
- Pickup and dropoff zone membership.
- Pickup/dropoff geometry and route cartometry.
- Sparse zone and lane membership.
- Neural repeated-ID residual features.
- Graph features from observed zone or source-target relationships.

## Split Design

Each task reports:

- Random split: validation rows are sampled from the same overall distribution
  as training rows.
- Spatial holdout: validation rows come from held-out pickup zones.

Pickup-demand spatial holdout intentionally skips learned models. That split
removes the demand history needed to learn zone-level demand, so learned-model
scores would measure fallback priors rather than a valid demand forecast.

## Methods

The requested model set is:

- Mean baseline.
- CartoBoost base candidate.
- CartoBoost reference row.
- CartoBoost neural row.
- CartoBoost graph rows using node2vec, GraphSAGE, HeteroGraphSAGE, and
  HinSAGE-style features.
- LightGBM.
- XGBoost.

The maintained repeated preset uses 100 estimators for both CartoBoost and
tree-boosting baselines, LightGBM/XGBoost max depth 4, CartoBoost max depth 5,
and target-mean zone treatment.

## Metrics

RMSE is the primary comparison metric because the targets are transformed
continuous regressions and larger errors matter. MAE is reported as a secondary
error metric. R2 reports explained variance. Training and prediction speed are
reported separately from quality.

## Current Single-Run Results

On every runnable learned-model split, the best CartoBoost-family row beats
LightGBM on RMSE and R2:

| task/split | best CartoBoost-family row | CartoBoost RMSE | LightGBM RMSE | RMSE delta | R2 delta |
| --- | --- | ---: | ---: | ---: | ---: |
| duration/random | `cartoboost` | 0.278843 | 0.289829 | -0.010986 | 0.012637 |
| duration/spatial_holdout | `cartoboost` | 0.303525 | 0.316291 | -0.012766 | 0.018206 |
| fare/random | `cartoboost` | 0.139640 | 0.143692 | -0.004052 | 0.004239 |
| fare/spatial_holdout | `cartoboost` | 0.148375 | 0.152686 | -0.004311 | 0.007854 |
| pickup_demand/random | `cartoboost_graph_node2vec` | 0.403529 | 0.481552 | -0.078024 | 0.016572 |

Full result tables and plots are stored in
`docs/assets/nyc_taxi_benchmarks/`.

## Interpretation

Fare and duration are geotemporal row-level targets. The winning row is base
CartoBoost, not a neural or graph variant. That means the useful signal is
mostly explained by hour/day periodicity, pickup/dropoff geometry, and zone
membership. LightGBM receives comparable target-mean zone features but still
has to approximate route geometry and cyclic time with axis-aligned tabular
splits.

Pickup demand is a zone-time graph problem. The best random-split row is
`cartoboost_graph_node2vec`, which adds topology learned from observed pickup
zone relationships before modeling hour, weekday, and zone effects. This is the
case where graph structure contributes signal beyond the base feature set.

Neural and graph rows are not universal upgrades. They are useful only when
the target contains repeated-ID residual structure or source-target topology
that ordinary dense columns do not expose.

## Repeated-Run Speed Study

The repeated target runs the same benchmark several times and summarizes
quality and timing relative to XGBoost:

```sh
just nyc-quality-benchmark-repeated
```

The repeated report shows that CartoBoost-family quality is better than XGBoost
on RMSE/R2 for the maintained splits, but the speed gate misses because
CartoBoost trains slower and predicts fewer rows per second than XGBoost under
this preset. That is a quality-vs-throughput tradeoff, not a universal
deployment recommendation.

## Limitations

- The current artifact is January 2024 yellow taxi data, not all TLC history.
- Scores are on transformed targets.
- Timing depends on local hardware and installed optional libraries.
- Pickup-demand spatial holdout is intentionally excluded for learned models.
- Claims should be refreshed when task definitions, split strategy, zone
  treatment, estimator settings, or sample size change.

## Smaller Diagnostics

Run one 25k-row month:

```sh
uv run --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --year 2024 \
  --months 1 \
  --sample-size 25000
```

Run without TLC files:

```sh
uv run --group bench python scripts/run_nyc_taxi_quality_benchmarks.py \
  --synthetic-smoke \
  --models mean
```
