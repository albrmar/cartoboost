# NYC Taxi Model Quality Benchmarks

## Research Question

On real NYC taxi data, do geographic and temporal feature families improve
prediction quality for trip duration, fare amount, and pickup-zone demand when
compared with strong gradient-boosted tabular baselines?

## Dataset

The benchmark uses NYC TLC taxi-derived records. The row-level tasks use trip
records with pickup/dropoff zone context, trip attributes, passenger count, and
time-of-day features. The demand task aggregates pickup activity by zone and
time bucket.

## Targets

Quality metrics are computed on transformed regression targets:

- Trip duration: log trip duration.
- Fare amount: log total amount.
- Pickup-zone demand: log pickup trip count for a zone-time bucket.

## Feature Sets

- Geographic features: pickup zone, dropoff zone, route geometry, and
  zone-level encodings.
- Temporal features: hour, weekday, and periodic time structure.
- Trip features: distance, passenger count, and related trip descriptors.
- Graph features for pickup demand: topology learned from observed pickup-zone
  relationships.

## Comparison Method

The primary `cartoboost` row is compared with the requested external
baselines that finish in the validated environment: XGBoost, optional
LightGBM and CatBoost estimators when available, scikit-learn tree
ensembles, Ridge, and a mean baseline under the same task, split, target
transformation, and global benchmark settings.

- dataset source: nyc_tlc_trip_records
- source URL: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- dataset hash: 741a94b7345cd469a8dc6261b116910f39131f6e1ca0e824dd319e53ef6bd8c8
- sample size: 30000
- task rows: {'duration': 30000, 'fare': 30000, 'pickup_demand': 24650}
- models requested: cartoboost, lightgbm, xgboost, catboost, hist_gradient_boosting, random_forest, extra_trees, ridge, mean
- baseline estimators: 24
- CartoBoost candidate estimators: 24
- baseline max depth: 4
- CartoBoost candidate max depth: 5
- model workers: 1
- zone treatment: target_mean
- command arguments: `scripts/run_nyc_taxi_quality_benchmarks.py --no-download --no-plots --sample-size 30000 --output-dir docs/assets/nyc_taxi_benchmarks --models cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,mean --n-estimators 24 --cartoboost-n-estimators 24 --tasks duration,fare,pickup_demand --model-workers 1`

## Resource Usage

| Field | Value |
| --- | --- |
| cpu | `arm` |
| threads | `10` |
| os | `macOS-26.5.1-arm64-arm-64bit-Mach-O` |
| python | `3.13.12` |
| numpy | `2.4.6` |
| rustc | `rustc 1.94.0 (4a4ef493e 2026-03-02)` |

## Baseline Dependency Status

| Key | Package | Import | Version | Module importable | Required class | Required class available |
| --- | --- | --- | --- | ---: | --- | ---: |
| catboost | catboost | catboost | `None` | False | CatBoostRegressor | False |
| lightgbm | lightgbm | lightgbm | `None` | True | LGBMRegressor | False |
| sklearn | scikit-learn | sklearn | `1.9.0` | True |  |  |
| xgboost | xgboost | xgboost | `3.3.0` | True | XGBRegressor | True |

## Output Artifacts

| Artifact | Size bytes |
| --- | ---: |
| `plots/duration_random_cartoboost_graph_graphsage_predicted_actual.png` | 82615 |
| `plots/duration_random_cartoboost_graph_graphsage_zone_residuals.png` | 33439 |
| `plots/duration_random_cartoboost_graph_hetero_graphsage_predicted_actual.png` | 84062 |
| `plots/duration_random_cartoboost_graph_hetero_graphsage_zone_residuals.png` | 33439 |
| `plots/duration_random_cartoboost_graph_hinsage_predicted_actual.png` | 82410 |
| `plots/duration_random_cartoboost_graph_hinsage_zone_residuals.png` | 33439 |
| `plots/duration_random_cartoboost_graph_node2vec_predicted_actual.png` | 82886 |
| `plots/duration_random_cartoboost_graph_node2vec_zone_residuals.png` | 33439 |
| `plots/duration_random_cartoboost_neural_predicted_actual.png` | 80798 |
| `plots/duration_random_cartoboost_neural_zone_residuals.png` | 33439 |
| `plots/duration_random_cartoboost_predicted_actual.png` | 80411 |
| `plots/duration_random_cartoboost_reference_predicted_actual.png` | 81222 |
| `plots/duration_random_cartoboost_reference_zone_residuals.png` | 33095 |
| `plots/duration_random_cartoboost_zone_residuals.png` | 33439 |
| `plots/duration_random_lightgbm_predicted_actual.png` | 79071 |
| `plots/duration_random_lightgbm_zone_residuals.png` | 33265 |
| `plots/duration_random_mean_predicted_actual.png` | 23830 |
| `plots/duration_random_mean_zone_residuals.png` | 31807 |
| `plots/duration_random_xgboost_predicted_actual.png` | 79883 |
| `plots/duration_random_xgboost_zone_residuals.png` | 33197 |
| `plots/duration_spatial_holdout_cartoboost_graph_graphsage_predicted_actual.png` | 85415 |
| `plots/duration_spatial_holdout_cartoboost_graph_graphsage_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_cartoboost_graph_hetero_graphsage_predicted_actual.png` | 85772 |
| `plots/duration_spatial_holdout_cartoboost_graph_hetero_graphsage_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_cartoboost_graph_hinsage_predicted_actual.png` | 84897 |
| `plots/duration_spatial_holdout_cartoboost_graph_hinsage_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_cartoboost_graph_node2vec_predicted_actual.png` | 85343 |
| `plots/duration_spatial_holdout_cartoboost_graph_node2vec_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_cartoboost_neural_predicted_actual.png` | 83507 |
| `plots/duration_spatial_holdout_cartoboost_neural_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_cartoboost_predicted_actual.png` | 82818 |
| `plots/duration_spatial_holdout_cartoboost_reference_predicted_actual.png` | 84079 |
| `plots/duration_spatial_holdout_cartoboost_reference_zone_residuals.png` | 23245 |
| `plots/duration_spatial_holdout_cartoboost_zone_residuals.png` | 22720 |
| `plots/duration_spatial_holdout_lightgbm_predicted_actual.png` | 82625 |
| `plots/duration_spatial_holdout_lightgbm_zone_residuals.png` | 22766 |
| `plots/duration_spatial_holdout_mean_predicted_actual.png` | 25059 |
| `plots/duration_spatial_holdout_mean_zone_residuals.png` | 23994 |
| `plots/duration_spatial_holdout_xgboost_predicted_actual.png` | 82208 |
| `plots/duration_spatial_holdout_xgboost_zone_residuals.png` | 22787 |
| `plots/fare_random_cartoboost_graph_graphsage_predicted_actual.png` | 61148 |
| `plots/fare_random_cartoboost_graph_graphsage_zone_residuals.png` | 35097 |
| `plots/fare_random_cartoboost_graph_hetero_graphsage_predicted_actual.png` | 62042 |
| `plots/fare_random_cartoboost_graph_hetero_graphsage_zone_residuals.png` | 35097 |
| `plots/fare_random_cartoboost_graph_hinsage_predicted_actual.png` | 60849 |
| `plots/fare_random_cartoboost_graph_hinsage_zone_residuals.png` | 35097 |
| `plots/fare_random_cartoboost_graph_node2vec_predicted_actual.png` | 61325 |
| `plots/fare_random_cartoboost_graph_node2vec_zone_residuals.png` | 35097 |
| `plots/fare_random_cartoboost_neural_predicted_actual.png` | 59314 |
| `plots/fare_random_cartoboost_neural_zone_residuals.png` | 35097 |
| `plots/fare_random_cartoboost_predicted_actual.png` | 58463 |
| `plots/fare_random_cartoboost_reference_predicted_actual.png` | 60547 |
| `plots/fare_random_cartoboost_reference_zone_residuals.png` | 31706 |
| `plots/fare_random_cartoboost_zone_residuals.png` | 35097 |
| `plots/fare_random_lightgbm_predicted_actual.png` | 58606 |
| `plots/fare_random_lightgbm_zone_residuals.png` | 35161 |
| `plots/fare_random_mean_predicted_actual.png` | 26760 |
| `plots/fare_random_mean_zone_residuals.png` | 34116 |
| `plots/fare_random_xgboost_predicted_actual.png` | 59208 |
| `plots/fare_random_xgboost_zone_residuals.png` | 35157 |
| `plots/fare_spatial_holdout_cartoboost_graph_graphsage_predicted_actual.png` | 70861 |
| `plots/fare_spatial_holdout_cartoboost_graph_graphsage_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_cartoboost_graph_hetero_graphsage_predicted_actual.png` | 70792 |
| `plots/fare_spatial_holdout_cartoboost_graph_hetero_graphsage_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_cartoboost_graph_hinsage_predicted_actual.png` | 69929 |
| `plots/fare_spatial_holdout_cartoboost_graph_hinsage_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_cartoboost_graph_node2vec_predicted_actual.png` | 70415 |
| `plots/fare_spatial_holdout_cartoboost_graph_node2vec_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_cartoboost_neural_predicted_actual.png` | 68961 |
| `plots/fare_spatial_holdout_cartoboost_neural_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_cartoboost_predicted_actual.png` | 68242 |
| `plots/fare_spatial_holdout_cartoboost_reference_predicted_actual.png` | 69638 |
| `plots/fare_spatial_holdout_cartoboost_reference_zone_residuals.png` | 22282 |
| `plots/fare_spatial_holdout_cartoboost_zone_residuals.png` | 21755 |
| `plots/fare_spatial_holdout_lightgbm_predicted_actual.png` | 67968 |
| `plots/fare_spatial_holdout_lightgbm_zone_residuals.png` | 21768 |
| `plots/fare_spatial_holdout_mean_predicted_actual.png` | 27023 |
| `plots/fare_spatial_holdout_mean_zone_residuals.png` | 23871 |
| `plots/fare_spatial_holdout_xgboost_predicted_actual.png` | 67825 |
| `plots/fare_spatial_holdout_xgboost_zone_residuals.png` | 21757 |
| `plots/pickup_demand_random_cartoboost_graph_graphsage_predicted_actual.png` | 84853 |
| `plots/pickup_demand_random_cartoboost_graph_graphsage_zone_residuals.png` | 40190 |
| `plots/pickup_demand_random_cartoboost_graph_hetero_graphsage_predicted_actual.png` | 84932 |
| `plots/pickup_demand_random_cartoboost_graph_hetero_graphsage_zone_residuals.png` | 40231 |
| `plots/pickup_demand_random_cartoboost_graph_hinsage_predicted_actual.png` | 83890 |
| `plots/pickup_demand_random_cartoboost_graph_hinsage_zone_residuals.png` | 40231 |
| `plots/pickup_demand_random_cartoboost_graph_node2vec_predicted_actual.png` | 84344 |
| `plots/pickup_demand_random_cartoboost_graph_node2vec_zone_residuals.png` | 37132 |
| `plots/pickup_demand_random_cartoboost_neural_predicted_actual.png` | 85236 |
| `plots/pickup_demand_random_cartoboost_neural_zone_residuals.png` | 39983 |
| `plots/pickup_demand_random_cartoboost_predicted_actual.png` | 84343 |
| `plots/pickup_demand_random_cartoboost_reference_predicted_actual.png` | 91036 |
| `plots/pickup_demand_random_cartoboost_reference_zone_residuals.png` | 34979 |
| `plots/pickup_demand_random_cartoboost_zone_residuals.png` | 39983 |
| `plots/pickup_demand_random_lightgbm_predicted_actual.png` | 87813 |
| `plots/pickup_demand_random_lightgbm_zone_residuals.png` | 39292 |
| `plots/pickup_demand_random_mean_predicted_actual.png` | 26220 |
| `plots/pickup_demand_random_mean_zone_residuals.png` | 40035 |
| `plots/pickup_demand_random_xgboost_predicted_actual.png` | 88929 |
| `plots/pickup_demand_random_xgboost_zone_residuals.png` | 34913 |
| `plots/pickup_demand_spatial_holdout_cartoboost_predicted_actual.png` | 65851 |
| `plots/pickup_demand_spatial_holdout_cartoboost_reference_predicted_actual.png` | 64626 |
| `plots/pickup_demand_spatial_holdout_cartoboost_reference_zone_residuals.png` | 20819 |
| `plots/pickup_demand_spatial_holdout_cartoboost_zone_residuals.png` | 20721 |
| `plots/pickup_demand_spatial_holdout_lightgbm_predicted_actual.png` | 65465 |
| `plots/pickup_demand_spatial_holdout_lightgbm_zone_residuals.png` | 21110 |
| `plots/pickup_demand_spatial_holdout_mean_predicted_actual.png` | 27059 |
| `plots/pickup_demand_spatial_holdout_mean_zone_residuals.png` | 23382 |
| `plots/pickup_demand_spatial_holdout_xgboost_predicted_actual.png` | 64212 |
| `plots/pickup_demand_spatial_holdout_xgboost_zone_residuals.png` | 21078 |
| `prediction_throughput.png` | 119853 |
| `results.json` | 220303 |
| `results.jsonl` | 38376 |
| `results.md` | 25359 |
| `speed_summary.png` | 121081 |

## Selection and Leakage Policy

- global hyperparameters: fixed_before_holdout_scoring; no model family uses test labels for tuning
- primary cartoboost row: single configured cartoboost run; no internal candidate is selected on test metrics
- zone target encoding: fit on training rows for each outer split before transforming holdout rows
- graph feature gate: uses deterministic inner train/validation rows inside the training split only
- neural feature gate: uses deterministic inner train/validation rows inside the training split only
- segment diagnostics: computed after prediction and excluded from fitting, tuning, and model selection

## CartoBoost vs External Baselines

For each runnable learned-model split, this table compares the single primary `cartoboost` row with the lowest-RMSE external baseline that finished under the same task, split, data sample, target transformation, and global benchmark settings.

| task | split | CartoBoost RMSE | CartoBoost WAPE | best external baseline | external RMSE | external WAPE | RMSE delta | R2 delta | result |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| duration | random | 0.321631 | 0.037386 | hist_gradient_boosting | 0.337427 | 0.039421 | -0.015796 | 0.021386 | cartoboost_lower_rmse |
| duration | spatial_holdout | 0.317218 | 0.037839 | hist_gradient_boosting | 0.330881 | 0.039590 | -0.013663 | 0.021329 | cartoboost_lower_rmse |
| fare | random | 0.174649 | 0.040675 | ridge | 0.169843 | 0.038907 | 0.004807 | -0.006014 | external_lower_or_tied_rmse |
| fare | spatial_holdout | 0.159090 | 0.039392 | ridge | 0.158739 | 0.039154 | 0.000350 | -0.000692 | external_lower_or_tied_rmse |
| pickup_demand | random | 0.598105 | 0.179037 | hist_gradient_boosting | 0.631339 | 0.191566 | -0.033234 | 0.009805 | cartoboost_lower_rmse |

### What Each Comparison Row Models

| task/split | prediction unit | target being modeled | validation question | modeling signal |
| --- | --- | --- | --- | --- |
| duration/random | one completed taxi trip | log trip duration in seconds | Can the model explain ordinary held-out trips drawn from the same month-wide trip distribution? | Base CartoBoost uses trip distance, passenger count, hour/weekday periodicity, pickup/dropoff zones, and route geometry. |
| duration/spatial_holdout | one completed taxi trip from held-out pickup zones | log trip duration in seconds | Does the trip-duration structure transfer when pickup zones are held out? | The gain comes from spatial splitters and route geometry rather than memorizing the exact validation rows. |
| fare/random | one completed taxi trip | log total fare amount | Can the model recover fare structure for ordinary held-out trips? | Distance, pickup/dropoff zones, hour/weekday effects, and cartometric route features align with how fares vary. |
| fare/spatial_holdout | one completed taxi trip from held-out pickup zones | log total fare amount | Does fare modeling generalize to zones not present in the training pickup set? | Route and zone geometry carry transferable fare signal beyond target-mean zone encodings. |
| pickup_demand/random | pickup zone x hour x weekday bucket | log pickup trip count | Can the model explain recurring zone-time demand for observed zones? | The node2vec row adds topology from observed pickup-zone relationships before modeling hour, weekday, and zone effects. |

### Interpretation Notes

- Fare and duration are primarily geotemporal row tasks. The base CartoBoost candidate uses native periodic hour/day splitters, diagonal and radial spatial splitters, and sparse-set taxi-zone membership. Those primitives let the model express pickup/dropoff geometry directly instead of asking an axis-only tabular baseline to approximate it through many rectangular cuts.
- Pickup demand is a zone-time graph problem. Graph rows are kept as diagnostics for topology-sensitive behavior, but the public comparison summary keeps `cartoboost` as the single product row.
- Graph and neural rows are not expected to improve every target. When the base geotemporal splitters already explain the signal, they match the base candidate and mainly add training cost. Their value is in workloads where ID residuals or source-target topology carry signal that ordinary dense columns do not expose.
- The pickup-demand cold-zone spatial holdout intentionally skips learned models. That split removes all zone demand history, so a quality comparison would collapse to priors rather than test model structure.

### Pickup-Zone Segment Diagnostics

These diagnostics are computed after prediction on each holdout split. They summarize pickup-zone error distribution and are not used for training, model selection, or tuning.

| task | split | model | pickup zones | zone rows min-max | zone RMSE p50 | zone RMSE p90 | worst zone RMSE |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| duration | random | cartoboost | 126 | 1-302 | 0.322226 | 0.692416 | 2.272663 |
| duration | random | hist_gradient_boosting | 126 | 1-302 | 0.342045 | 0.698395 | 2.226535 |
| duration | spatial_holdout | cartoboost | 39 | 1-1525 | 0.358977 | 0.700164 | 1.641809 |
| duration | spatial_holdout | hist_gradient_boosting | 39 | 1-1525 | 0.367757 | 0.718512 | 1.740194 |
| fare | random | cartoboost | 126 | 1-302 | 0.161276 | 0.420249 | 1.238979 |
| fare | random | ridge | 126 | 1-302 | 0.169932 | 0.453957 | 1.461132 |
| fare | spatial_holdout | cartoboost | 39 | 1-1525 | 0.176255 | 0.481167 | 0.812295 |
| fare | spatial_holdout | ridge | 39 | 1-1525 | 0.206107 | 0.488287 | 0.811945 |
| pickup_demand | random | cartoboost | 239 | 1-47 | 0.505235 | 0.833421 | 1.955022 |
| pickup_demand | random | hist_gradient_boosting | 239 | 1-47 | 0.555335 | 0.899397 | 2.060286 |

## Trip duration

Predict log trip duration from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.321631 | 0.244797 | 0.787495 | 0.037386 | 50.407302 | 0.011504 | 521570.96 | n_estimators=24 |
| lightgbm | skipped |  |  |  |  |  |  |  | lightgbm is not installed |
| xgboost | ok | 0.340162 | 0.260461 | 0.762302 | 0.039779 | 0.040749 | 0.000441 | 13592589.89 | n_estimators=24 |
| catboost | skipped |  |  |  |  |  |  |  | catboost is not installed |
| hist_gradient_boosting | ok | 0.337427 | 0.258119 | 0.766108 | 0.039421 | 0.084703 | 0.000923 | 6497606.28 | n_estimators=24 |
| random_forest | ok | 0.346062 | 0.266391 | 0.753984 | 0.040684 | 0.156632 | 0.017473 | 343386.94 | n_estimators=24 |
| extra_trees | ok | 0.360183 | 0.279085 | 0.733498 | 0.042623 | 0.046253 | 0.015019 | 399491.77 | n_estimators=24 |
| ridge | ok | 0.361002 | 0.277595 | 0.732285 | 0.042396 | 0.002097 | 0.000104 | 57924543.05 |  |
| mean | ok | 0.697752 | 0.557510 | -0.000130 | 0.085145 | 0.000018 | 0.000007 | 842109170.33 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.317218 | 0.244041 | 0.757616 | 0.037839 | 51.410661 | 0.012235 | 474782.23 | n_estimators=24 |
| lightgbm | skipped |  |  |  |  |  |  |  | lightgbm is not installed |
| xgboost | ok | 0.334581 | 0.258639 | 0.730355 | 0.040102 | 0.044892 | 0.000479 | 12130514.01 | n_estimators=24 |
| catboost | skipped |  |  |  |  |  |  |  | catboost is not installed |
| hist_gradient_boosting | ok | 0.330881 | 0.255337 | 0.736287 | 0.039590 | 0.241046 | 0.003092 | 1878947.16 | n_estimators=24 |
| random_forest | ok | 0.339897 | 0.264278 | 0.721719 | 0.040977 | 0.196771 | 0.012415 | 467898.57 | n_estimators=24 |
| extra_trees | ok | 0.353708 | 0.273389 | 0.698645 | 0.042389 | 0.048145 | 0.016433 | 353488.85 | n_estimators=24 |
| ridge | ok | 0.354533 | 0.274377 | 0.697238 | 0.042542 | 0.001590 | 0.000086 | 67875617.71 |  |
| mean | ok | 0.657256 | 0.519352 | -0.040539 | 0.080526 | 0.000017 | 0.000007 | 850139365.28 |  |

## Fare amount

Predict log total amount from zone, trip, passenger, and time features.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.174649 | 0.128754 | 0.889225 | 0.040675 | 51.119769 | 0.013276 | 451929.19 | n_estimators=24 |
| lightgbm | skipped |  |  |  |  |  |  |  | lightgbm is not installed |
| xgboost | ok | 0.182710 | 0.136996 | 0.878764 | 0.043279 | 0.048212 | 0.000572 | 10489510.67 | n_estimators=24 |
| catboost | skipped |  |  |  |  |  |  |  | catboost is not installed |
| hist_gradient_boosting | ok | 0.180789 | 0.134599 | 0.881300 | 0.042522 | 0.447112 | 0.016097 | 372748.95 | n_estimators=24 |
| random_forest | ok | 0.177191 | 0.129255 | 0.885978 | 0.040834 | 0.283480 | 0.014788 | 405736.68 | n_estimators=24 |
| extra_trees | ok | 0.181030 | 0.133740 | 0.880984 | 0.042251 | 0.057126 | 0.016179 | 370854.93 | n_estimators=24 |
| ridge | ok | 0.169843 | 0.123155 | 0.895239 | 0.038907 | 0.002054 | 0.000114 | 52535255.72 |  |
| mean | ok | 0.524746 | 0.399845 | -0.000010 | 0.126318 | 0.000029 | 0.000008 | 712926667.20 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.159090 | 0.120233 | 0.842719 | 0.039392 | 46.510610 | 0.012291 | 472635.09 | n_estimators=24 |
| lightgbm | skipped |  |  |  |  |  |  |  | lightgbm is not installed |
| xgboost | ok | 0.167078 | 0.127587 | 0.826527 | 0.041801 | 0.043308 | 0.000433 | 13413131.99 | n_estimators=24 |
| catboost | skipped |  |  |  |  |  |  |  | catboost is not installed |
| hist_gradient_boosting | ok | 0.163738 | 0.124610 | 0.833393 | 0.040826 | 0.146589 | 0.001209 | 4806785.25 | n_estimators=24 |
| random_forest | ok | 0.166072 | 0.124522 | 0.828609 | 0.040797 | 0.173391 | 0.015417 | 376797.96 | n_estimators=24 |
| extra_trees | ok | 0.170498 | 0.129226 | 0.819354 | 0.042338 | 0.055385 | 0.016583 | 350306.42 | n_estimators=24 |
| ridge | ok | 0.158739 | 0.119507 | 0.843411 | 0.039154 | 0.001711 | 0.000088 | 66388551.71 |  |
| mean | ok | 0.425543 | 0.344144 | -0.125328 | 0.112751 | 0.000016 | 0.000007 | 865985145.08 |  |

## Pickup-zone demand

Predict log pickup trip count for a pickup zone, hour, and weekday bucket.

### random

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | ok | 0.598105 | 0.481325 | 0.914156 | 0.179037 | 10.946152 | 0.008383 | 588089.13 | n_estimators=24 |
| lightgbm | skipped |  |  |  |  |  |  |  | lightgbm is not installed |
| xgboost | ok | 0.647344 | 0.521361 | 0.899440 | 0.193929 | 0.035647 | 0.000465 | 10597409.64 | n_estimators=24 |
| catboost | skipped |  |  |  |  |  |  |  | catboost is not installed |
| hist_gradient_boosting | ok | 0.631339 | 0.515010 | 0.904351 | 0.191566 | 0.223907 | 0.003299 | 1494562.13 | n_estimators=24 |
| random_forest | ok | 0.645452 | 0.486077 | 0.900027 | 0.180804 | 0.064140 | 0.017189 | 286816.21 | n_estimators=24 |
| extra_trees | ok | 0.697081 | 0.534807 | 0.883394 | 0.198930 | 0.032791 | 0.016629 | 296469.29 | n_estimators=24 |
| ridge | ok | 0.800820 | 0.606266 | 0.846105 | 0.225510 | 0.001162 | 0.000082 | 60152065.62 |  |
| mean | ok | 2.041944 | 1.752775 | -0.000560 | 0.651973 | 0.000011 | 0.000006 | 778462822.40 |  |

### spatial_holdout

| model | status | RMSE | MAE | R2 | WAPE | train sec | predict sec | predict rows/sec | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cartoboost | skipped |  |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| lightgbm | skipped |  |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| xgboost | skipped |  |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| catboost | skipped |  |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| hist_gradient_boosting | skipped |  |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| random_forest | skipped |  |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| extra_trees | skipped |  |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| ridge | skipped |  |  |  |  |  |  |  | learned models are skipped for pickup_demand cold-zone spatial holdout; the split removes all zone demand history, so predictions collapse to priors |
| mean | ok | 2.088607 | 1.807484 | -0.002958 | 0.659779 | 0.000009 | 0.000005 | 1027804139.01 |  |

