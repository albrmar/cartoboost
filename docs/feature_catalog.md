# Feature Catalog

This page is the high-level catalog for CartoBoost features. Detailed API
contracts live in the linked topic pages and in the Python API reference.

## Regression Estimator

`cartoboost.CartoBoostRegressor` is the main sklearn-style estimator for
tabular, temporal, spatial, and sparse-set regression.

| Feature | Public surface | Notes |
| --- | --- | --- |
| L2 regression | `loss="l2"` or `"squared_error"` | Default squared-error objective. |
| L1 regression | `loss="l1"`, `"mae"`, or `"absolute_error"` | Constant leaves only. |
| Huber regression | `loss="huber"`, `huber_delta=...` | Constant leaves only. |
| Log-L2 regression | `loss="log_l2"`, `log_offset=1.0` | Positive-offset log target objective. |
| Quantile regression | `loss="quantile"` or `"pinball"`, `quantile_alpha=...` | Pinball-loss objective for conditional quantiles. |
| Constant leaves | `leaf_predictor="constant"` | Default leaf model. |
| Linear leaves | `leaf_predictor="linear"`, `linear_leaf_features=[...]` | Ridge-regularized residual leaves for local linear trends. |
| Sample weights | `fit(..., sample_weight=...)` | Supported for training and objectives. |
| Monotonic constraints | `monotonic_constraints=[-1, 0, 1, ...]` | Axis-style constant-leaf, non-fuzzy fits. |
| Additive values | `predict_additive_values(...)` | Per-tree additive contributions whose row sum matches prediction. |
| sklearn compatibility | `get_params`, `set_params`, `clone`, `Pipeline`, `GridSearchCV` | Standard estimator ergonomics. |
| Artifacts | `save`, `load`, `save_weights`, `load_weights` | Versioned JSON model artifacts and supported weights artifacts. |
| ONNX export subset | `save_weights(path, format="onnx")` | Requires `cartoboost[onnx]`; dense axis-tree subset only. |

See [Python Estimator](user-guide/python-estimator.md),
[Parameters](user-guide/parameters.md), [Objectives](objectives.md),
[Constraints](constraints.md), and [Model Artifacts](model_artifact.md).

## Splitters And Feature Semantics

CartoBoost splitters tell the model which geometry or data shape is meaningful.

| Splitter | Public names | Use case |
| --- | --- | --- |
| Automatic | `auto` or `splitters=None` | Exact axis search for small/constrained fits, histogram axis for larger dense L2 fits. |
| Axis | `axis` | Standard one-feature thresholds. |
| Histogram axis | `axis_histogram`, `axis_hist`, `histogram`, `axis_histogram:<bins>` | Faster dense numeric threshold search. |
| Diagonal 2D | `diagonal_2d`, `diagonal2d` | Oblique boundaries for projected coordinates or route geometry. |
| Gaussian/radial 2D | `gaussian_2d`, `gaussian2d`, `radial` | Local hotspots, depots, corridors, or neighborhood effects. |
| Periodic | `periodic_time`, `periodic_24`, `periodic:<period>` | Wraparound hour, weekday, or seasonal phase. |
| Sparse set | `sparse_set`, `sparse` | List-valued taxi-zone, route, grid, H3, or S2 memberships. |
| Fuzzy routing | `fuzzy=True`, `fuzzy_bandwidth=...`, `fuzzy_kernel=...` | Fractional routing near split boundaries. |

`FeatureSchema` records dense numeric, periodic, sparse-set, H3 sparse-set, and
S2 sparse-set roles so saved artifacts and prediction inputs can be validated.

See [Feature Schema](feature_schema.md), [Sparse Features](sparse_features.md),
and [Spatial Modeling](spatial_modeling.md).

## Data Inputs And Optional Extras

| Capability | Public surface | Extra |
| --- | --- | --- |
| NumPy-style dense arrays | `fit(X, y)`, `predict(X)` | Core package. |
| pandas/dataframe-style inputs | Dataframe columns in estimator and forecasting helpers | Core package. |
| DuckDB relation inputs | Dense relation/query-result support | `cartoboost[duckdb]`. |
| Polars inputs | Dataframe support where documented | `cartoboost[polars]`. |
| H3 encoding | `latlng_to_h3_id`, `encode_h3_cells`, `build_h3_sparse_sets`, `h3_parent_id`, `normalize_h3_id` | `cartoboost[h3]`. |
| S2 encoding | `latlng_to_s2_id`, `encode_s2_cells`, `build_s2_sparse_sets`, `s2_parent_id`, `normalize_s2_id` | `cartoboost[s2]`. |
| Geographic sparse helpers | `build_geo_sparse_sets`, `build_zip_sparse_sets`, `coerce_geo_to_feature_id`, `coerce_zip_to_feature_id` | Core package. |
| SHAP explanations | `make_shap_explainer`, `explain_shap` | `cartoboost[explain]`. |
| Optuna workflows | Tuning examples/workflows | `cartoboost[optuna]`. |

See [Installation](installation.md), [Sparse Features](sparse_features.md), and
[SHAP Support](shap.md).

## General Utilities

These utilities are Rust-backed and independent of both `CartoBoostRegressor`
and `cartoboost.forecasting`.

| Feature | Public surface | Notes |
| --- | --- | --- |
| Single-series utility forecasts | `cartoboost.naive_forecast`, `seasonal_naive_forecast`, `theta_forecast`, `optimized_theta_forecast`, `ets_forecast`, `arima_forecast`, `auto_arima_forecast` | Rust-backed forecasts for plain numeric sequences without constructing a `ForecastFrame`. |
| Local-level Kalman filter | `cartoboost.local_level_kalman_filter`, `local_level_kalman_forecast` | Filters and forecasts a numeric series with a one-state local-level Kalman model. |
| Local-linear Kalman filter | `cartoboost.kalman_filter` | Filters a numeric series, returns per-step estimates, final level/trend state, and optional future mean forecast. |
| Local-linear Kalman forecast | `cartoboost.local_linear_trend_kalman_forecast` | Forecasts a numeric sequence with the Rust local-linear trend Kalman model. |
| Intermittent demand | `cartoboost.croston_forecast`, `sba_forecast`, `tsb_forecast`, `intermittent_demand_forecast` | Rust-backed Croston, SBA, and TSB utilities for non-negative intermittent demand sequences. |
| Ordinary kriging | `cartoboost.ordinary_kriging_predict` | Interpolates target `(x, y)` coordinates from observed `(x, y, value)` triples and returns means plus kriging weights. |

## Evaluation And Metrics

| Feature | Public surface | Notes |
| --- | --- | --- |
| Out-of-time split | `out_of_time_split` | Sorts by time and reserves a future validation block. |
| Temporal blocked CV | `temporal_blocked_cv` | Time-ordered folds with optional gap. |
| Spatial blocked CV | `spatial_blocked_cv` | Coordinate-blocked validation folds. |
| Grouped blocked CV | `grouped_blocked_cv` | Keeps groups together across folds. |
| Pinball loss | `pinball_loss` | Quantile scoring. |
| Interval metrics | `interval_coverage`, `mean_interval_width`, `calibrated_intervals` | Prediction-interval diagnostics. |
| Residual diagnostics | `residual_morans_i`, `jitter_volatility`, `conformal_residual_quantile` | Spatial autocorrelation, stability, and conformal helpers. |

See [Evaluation Protocol](evaluation_protocol.md).

See [General Utilities](general_utilities.md) for toy examples covering
Kalman filters, ordinary kriging, intermittent-demand methods, and single-series
forecast helpers.

## Forecasting

Forecasting is Rust-native. Python classes validate data and delegate model
training, prediction, rolling-origin backtesting, metrics, and artifact behavior
to `cartoboost._native`.

| Feature | Public surface | Notes |
| --- | --- | --- |
| Forecast frame validation | `ForecastFrame.from_pandas` | Validates timestamps, finite targets, regular frequency, duplicate timestamps, panel ordering, and covariate roles. |
| Stable forecast tables | `ForecastResult`, `PredictionInterval` | Deterministic `series_id`, `timestamp`, `horizon`, `model`, `mean`, and interval columns. |
| Naive baseline | `NaiveForecaster` | Repeats last value per series. |
| Seasonal naive | `SeasonalNaiveForecaster(season_length)` | Repeats last seasonal cycle. |
| Theta | `ThetaForecaster` | Rust local theta method with optional seasonality and residual intervals. |
| Optimized theta | `OptimizedThetaForecaster` | Deterministic theta/alpha grid selection. |
| ETS | `ETSForecaster` | Rust additive ETS with optional additive seasonality. |
| ARIMA | `AutoARIMAForecaster` | Bounded non-seasonal AutoARIMA over ARIMA(p,d,q) candidates. |
| Kalman | `KalmanForecaster` | Independent local-linear state-space forecaster with level, trend, and observation variances. |
| Kriging | `KrigingForecaster` | Independent ordinary-kriging panel forecaster requiring explicit coordinates keyed by series id. |
| CartoBoost lag forecasting | `CartoBoostLagForecaster`, `LagFeatureConfig`, `RollingFeatureConfig`, `CalendarFeatureConfig` | Recursive supervised global forecaster with leakage-safe lag, rolling, calendar, static, and known-future features. |
| Fixed-weight ensembles | `WeightedEnsembleForecaster` | Native ensemble over aligned component forecasts. |
| Rolling-origin evaluation | `RollingOriginSplitter`, `ExpandingWindowSplitter`, `SlidingWindowSplitter`, `RollingOriginBacktester` | Leakage-safe folds with native fast path for native forecasters. |
| Forecast metrics | `ForecastMetricSet` | MAE, RMSE, MAPE, sMAPE, MASE, WAPE, bias, pinball, interval coverage, and interval width. |
| Forecast registry/config | `ForecastRegistry`, `ForecastModelSpec`, `ForecastingConfig` | Strict named construction and TOML-style configuration. |
| Forecast artifacts | `ForecastArtifact`, `ForecastArtifactManifest` | JSON manifest plus forecast table persistence. |
| Forecast CLI | `cartoboost forecast ...` | Fit/predict/backtest/compare command scaffold for CSV workflows. |

See [Forecasting](forecasting.md), [Forecasting API](forecasting_api.md),
[Forecasting Models](forecasting_models.md), [Forecasting Backtesting](forecasting_backtesting.md),
[Forecasting Lag Features](forecasting_lag_features.md), [Forecasting Artifacts](forecasting_artifacts.md),
and [Forecasting CLI](forecasting_cli.md).

## Neural Embedding Models And Features

| Feature | Public surface | Notes |
| --- | --- | --- |
| Standalone supervised ID model | `NeuralEmbeddingStandaloneRegressor` | Direct regressor over learned ID embeddings plus optional dense features. |
| Hybrid neural features | `NeuralEmbeddingRegressor` | Learns ID vectors and appends them to a tabular model. |
| Neural feature blocks | `NeuralEmbeddingFeatures` | Feature-generation helper. |
| Fallback behavior | `ArtifactFallback` and fallback arguments | Handles unseen or rare IDs through configured fallback vectors/chains. |
| Benchmark helper | `benchmark_neural_vs_cartoboost` | Quick held-out comparison between structured and neural-enhanced models. |
| Artifacts | `save`, `load` on standalone models | Persist learned embedding model state. |

See [Neural Embedding Models And Features](neural-features.md).

## Graph Models And Features

Graph support is available both as direct standalone modeling and as optional
feature generation for another estimator.

| Feature | Public surface | Notes |
| --- | --- | --- |
| Node2Vec encoder | `Node2VecEncoder`, `Node2VecFeatureEncoder`, `Node2VecConfig` | Directed/weighted random-walk embeddings with p/q transition bias. |
| GraphSAGE encoder | `GraphSageEncoder`, `GraphSageFeatureEncoder`, `GraphSageConfig` | Homogeneous graph embeddings with node attributes. |
| HeteroGraphSAGE encoder | `HeteroGraphSageEncoder`, `HeteroGraphSageFeatureEncoder`, `HeteroGraphSageConfig` | Typed-edge graph embeddings. |
| HinSAGE encoder | `HinSageEncoder`, `HinSageFeatureEncoder`, `HinSageConfig` | Typed-node and typed-relation graph surface with schema validation. |
| Feature transformer | `GraphFeatureTransformer`, `GraphFeatureBundle` | Produces dense graph columns and optional sparse sets for another model. |
| Graph schemas | `GraphSchema`, `EdgeType`, `DirectionalityConfig`, `DirectedMetaPath`, `TemporalEdge` | Validates directed heterogeneous graph contracts. |
| Graph builders | `HomogeneousGraph`, `HeterogeneousGraph`, `SourceTargetPairNodes`, `materialize_source_target_pair_nodes` | Normalizes graph topology and preserves source-target pair identity. |
| Walk generators | `MetaPathWalkGenerator`, `TemporalWalkGenerator`, `SignedEdgeSampler` | Constrained directed, temporal, and signed walk utilities. |
| Standalone graph regressors | `Node2VecStandaloneRegressor`, `GraphSageStandaloneRegressor`, `HeteroGraphSageStandaloneRegressor`, `HinSageStandaloneRegressor` | Direct graph regression without a boosted wrapper. |
| Standalone link predictors | `Node2VecLinkPredictor`, `GraphSageLinkPredictor`, `HeteroGraphSageLinkPredictor`, `HinSageLinkPredictor` | Link scoring plus reports. |
| Link metrics | `binary_auc`, `binary_average_precision`, `top_k_metrics`, `mean_reciprocal_rank`, `link_prediction_report` | Ranking and binary link-prediction diagnostics. |
| Directional features | `DirectionalFeature`, `DirectionalityConfig` | Preserves `source -> target` semantics and reverse-flow contrasts. |

See [Graph Models And Features](graph-features.md).

## Command Line Interfaces

| Command group | Public surface | Notes |
| --- | --- | --- |
| Regression CLI | `cartoboost train`, `predict`, `eval`, `inspect` | Dense numeric CSV workflows. |
| Forecasting CLI | `cartoboost forecast fit`, `predict`, `backtest`, `compare` | Forecasting command scaffold and strict CSV validation. |

Use Python for sparse-set, graph-derived, neural embedding, and custom
forecasting workflows that need richer in-memory objects.

See [CLI Reference](reference/cli.md), [CLI User Guide](user-guide/cli.md), and
[Forecasting CLI](forecasting_cli.md).

## Quality And Benchmark Reporting

| Feature | Public surface | Notes |
| --- | --- | --- |
| NYC taxi benchmarks | `docs/benchmarks/nyc-taxi.md` and assets | Duration, fare, and pickup-demand reports with split descriptions. |
| Model-suite benchmarks | `docs/benchmarks/model-suite.md` | Synthetic dense, repeated-ID, and graph-signal workloads. |
| Acceptance benchmarks | `docs/benchmarks/taxi-zone.md` | Taxi-zone behavior checks before broader claims. |
| Validation scripts | `scripts/` and test suite | Commands should report exact settings, metrics, and data provenance. |

Benchmark claims should name the dataset, target, split, feature set, metric,
model settings, and whether data is synthetic, generated acceptance data, or
real benchmark data.
