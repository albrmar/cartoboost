# Model Types

CartoBoost exposes several model families. Use this page to choose the right
entry point before tuning parameters.

Most modeling behavior is implemented in Rust and surfaced through thin Python
wrappers. The Python classes handle validation, dataframe ergonomics, sklearn
compatibility where appropriate, and artifact helpers.

## Quick Chooser

| Need | Start with | Why |
| --- | --- | --- |
| General tabular, spatial, temporal, or sparse-set regression | `cartoboost.CartoBoostRegressor` | Main sklearn-style estimator with CartoBoost splitters, objectives, sample weights, artifacts, and SHAP helpers. |
| Demand, fare, or duration forecasting from one series | `NaiveForecaster`, `SeasonalNaiveForecaster`, `ThetaForecaster`, `ETSForecaster`, `AutoARIMAForecaster`, or `KalmanForecaster` | Local Rust-native forecasting models for a single regular series. |
| Pickup-zone, dropoff-zone, or lane-level panel forecasting | `CartoBoostLagForecaster` | Global supervised forecaster that learns across series using lag, rolling, calendar, and known-future features. |
| Spatial interpolation across known series coordinates | `KrigingForecaster` | Ordinary-kriging panel forecaster over explicit `(x, y)` coordinates keyed by series id. |
| Combining several fitted native forecasters | `WeightedEnsembleForecaster` | Fixed-weight ensemble over native component models. |
| Direct supervised ID embedding regression | `NeuralEmbeddingStandaloneRegressor` | Standalone neural artifact for stable IDs such as pickup zones, dropoff zones, pairs, or zone-hour buckets. |
| Graph relationship modeling | Graph standalone regressors or link predictors | Direct graph models for node pairs, typed edges, and directed source-target semantics. |
| Graph or neural columns for a separate tabular model | `GraphFeatureTransformer`, `NeuralEmbeddingFeatures`, or `NeuralEmbeddingRegressor` | Feature-generation workflows when embeddings should become dense model columns. |

## Detailed Forecasting Pages

Each major forecasting model has its own example page:

| Model page | Covers |
| --- | --- |
| [Naive And Seasonal Naive](forecasting-models/naive-seasonal.md) | Last-value and seasonal baselines for pickup-zone demand. |
| [Theta](forecasting-models/theta.md) | Manual theta and optimized theta examples. |
| [ETS](forecasting-models/ets.md) | Additive level, trend, and seasonality. |
| [ARIMA And AutoARIMA](forecasting-models/arima.md) | Fixed-order ARIMA and bounded non-seasonal AutoARIMA. |
| [Kalman](forecasting-models/kalman.md) | Local-linear-trend state-space forecasting. |
| [Kriging](forecasting-models/kriging.md) | Coordinate-aware panel forecasting. |
| [CartoBoost Lag](forecasting-models/cartoboost-lag.md) | Global supervised lag forecasting across many series. |
| [Weighted Ensembles](forecasting-models/ensembles.md) | Fixed-weight combinations of native forecasting models. |

## Tabular Regression

`CartoBoostRegressor` is the main estimator for ordinary regression tasks:

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=20,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24"],
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Use it for taxi-trip fare, duration, demand, or residual modeling when each row
has dense numeric features such as trip distance, projected pickup/dropoff
coordinates, pickup hour, day of week, and route-level aggregates.

Important model variants are controlled by parameters:

| Type | Parameters | Use when |
| --- | --- | --- |
| Axis or histogram boosted trees | `splitters=None`, `["auto"]`, `["axis"]`, or `["axis_histogram:<bins>"]` | You need a strong dense tabular baseline. |
| Spatial trees | `splitters=["axis", "diagonal_2d", "gaussian_2d"]` | Coordinates, projected x/y values, or route geometry shape residuals. |
| Temporal trees | `splitters=["axis", "periodic:24"]` or another `periodic:<period>` | Hour `23` and hour `0`, weekdays, or seasonal phases should be adjacent. |
| Sparse route-membership trees | `splitters=["axis", "sparse_set"]` plus `sparse_sets=` | Rows belong to one or more taxi zones, H3/S2 cells, service areas, or route memberships. |
| Fuzzy spatial-temporal trees | `fuzzy=True`, `fuzzy_bandwidth=...`, `fuzzy_kernel=...` | Nearby coordinates or times should blend smoothly across learned boundaries. |
| Robust objectives | `loss="mae"`, `loss="huber"`, or `loss="log_l2"` | Outliers or log-scale targets dominate squared-error training. |
| Quantile models | `loss="quantile"`, `quantile_alpha=...` | You need conditional lower, median, or upper forecasts for service-level planning. |
| Linear residual leaves | `leaf_predictor="linear"`, `linear_leaf_features=[...]` | A region or time bucket still has an approximately linear residual trend. |
| Monotonic models | `monotonic_constraints=[...]` | Domain rules require predictions to move in one direction with a feature. |

See [Python Estimator](python-estimator.md), [Parameters](parameters.md),
[Feature Schema](../feature_schema.md), and [Sparse Features](../sparse_features.md).

## Forecasting Models

Forecasting models live under `cartoboost.forecasting`. They accept plain
numeric sequences, dictionaries of equal-length panel series, native
`ForecastFrame` objects, or dataframe inputs where documented. For pandas
workflows, prefer `ForecastFrame.from_pandas` because it validates timestamps,
frequency, duplicate rows, target values, panel ids, and covariate roles.

```python
from cartoboost.forecasting import ForecastFrame

frame = ForecastFrame.from_pandas(
    hourly_zone_demand,
    timestamp_col="pickup_hour",
    target_col="pickup_count",
    series_id_col="PULocationID",
    freq="h",
    known_future_covariates=["hour", "day_of_week"],
)
```

### Local Baselines

Local models fit each series pattern directly and are useful as baselines,
fallbacks, or components in ensembles.

| Model | Import | Key controls | Use when |
| --- | --- | --- | --- |
| Naive | `from cartoboost.forecasting import NaiveForecaster` | `prediction_interval_levels` | The last observed value is the benchmark to beat. |
| Seasonal naive | `SeasonalNaiveForecaster(season_length)` | `season_length`, `prediction_interval_levels` | Demand repeats by hour, day, week, or another fixed cycle. |
| Theta | `ThetaForecaster` | `theta`, `alpha`, optional `season_length`, `seasonality` | You need a lightweight trend extrapolator. |
| Optimized theta | `OptimizedThetaForecaster` | `theta_grid`, `alpha_grid`, optional seasonality | You want deterministic grid selection instead of choosing theta manually. |
| ETS | `ETSForecaster` | additive `trend`, additive `seasonal`, `seasonal_periods`, `alpha`, `beta`, `gamma` | Level, trend, and additive seasonality are enough for the series. |
| ARIMA | `from cartoboost.forecasting.local import ArimaForecaster` | `p`, `d`, `q` | You know the bounded non-seasonal ARIMA order. |
| AutoARIMA | `AutoARIMAForecaster` | `max_p`, `max_d`, `max_q`; `seasonal=False` | You want bounded non-seasonal ARIMA candidate search. |
| Kalman | `KalmanForecaster` | level, trend, and observation variances | The series is noisy but has a local level and local trend. |

```python
from cartoboost.forecasting import SeasonalNaiveForecaster, KalmanForecaster

seasonal = SeasonalNaiveForecaster(season_length=24).fit(zone_hourly_counts)
baseline = seasonal.predict(12)

kalman = KalmanForecaster(
    level_process_variance=0.05,
    trend_process_variance=0.005,
    observation_variance=1.0,
).fit(zone_hourly_counts)
forecast = kalman.predict(12)
```

Current Rust bindings intentionally reject unsupported modes such as damped ETS,
multiplicative ETS, seasonal AutoARIMA, and Python fallback algorithms.

### Global Forecasting

Use `CartoBoostLagForecaster` when many related series should share one model:
pickup-zone demand, dropoff-zone demand, airport lanes, borough-to-borough
flows, or route-level fare/duration time series.

```python
from cartoboost.forecasting import CartoBoostLagForecaster

model = CartoBoostLagForecaster(
    time_col="pickup_hour",
    target_col="pickup_count",
    panel_cols=["PULocationID"],
    frequency="h",
    lags=[1, 2, 24, 168],
    rolling_windows=[24, 168],
    calendar_features=True,
    recursive=True,
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=20,
    splitters=["axis", "periodic:24"],
)
model.fit(hourly_zone_demand)
pred = model.predict(24)
```

The native wrapper supports direct lag lists and rolling mean windows. It also
accepts `LagFeatureConfig`, `RollingFeatureConfig`, and
`CalendarFeatureConfig` for supported native options. Keep lag and rolling
features strictly historical so backtests measure real forecast behavior.

### Spatial Panel Forecasting

`KrigingForecaster` is for panels where each series has a stable coordinate,
such as pickup-zone centroids or route midpoint geometry:

```python
from cartoboost.forecasting import KrigingForecaster

coordinates = {
    "132": (-73.7781, 40.6413),
    "161": (-73.9776, 40.7580),
    "236": (-73.9577, 40.7808),
}

model = KrigingForecaster(coordinates=coordinates, range=2.0, nugget=1e-6)
model.fit({"132": series_132, "161": series_161, "236": series_236})
forecast = model.predict(6)
```

Use it when nearby zones should borrow strength spatially. Do not use it as a
replacement for leakage-safe temporal validation; coordinates explain spatial
dependence, not future observations.

### Ensembles

`WeightedEnsembleForecaster` combines native forecasting wrappers with explicit
weights:

```python
from cartoboost.forecasting import (
    SeasonalNaiveForecaster,
    ThetaForecaster,
    WeightedEnsembleForecaster,
)

models = {
    "seasonal": SeasonalNaiveForecaster(season_length=24),
    "theta": ThetaForecaster(),
}
ensemble = WeightedEnsembleForecaster(models=models, weights={"seasonal": 0.6, "theta": 0.4})
ensemble.fit(zone_hourly_counts)
forecast = ensemble.predict(24)
```

Weights must match model names exactly. Prediction intervals for weighted
ensembles are not supported yet.

See [Forecasting](../forecasting.md), [Forecasting API](../forecasting_api.md),
[Forecasting Models](../forecasting_models.md), and
[Forecasting Backtesting](../forecasting_backtesting.md).

## Neural Embedding Models

Neural embedding support has two distinct modes.

`NeuralEmbeddingStandaloneRegressor` is a direct model:

```python
import numpy as np
from cartoboost.neural import NeuralEmbeddingStandaloneRegressor

pickup_zone_ids = np.asarray([132, 161, 132, 236], dtype=np.uint64)
dense = np.asarray([[1.0, 6.0], [2.5, 8.0], [1.2, 6.0], [3.1, 17.0]])
log_fare = np.asarray([2.7, 3.1, 2.8, 3.4])

model = NeuralEmbeddingStandaloneRegressor(
    dim=8,
    n_estimators=50,
    max_depth=3,
    min_samples_leaf=2,
    random_state=7,
)
model.fit(pickup_zone_ids, log_fare, dense=dense)
pred = model.predict(pickup_zone_ids, dense=dense)
```

Use the standalone model when the learned ID embedding is the artifact to train,
score, save, and serve. It works best when train and prediction populations
share stable IDs. Under cold-zone or cold-ID holdouts, report fallback behavior
explicitly because unseen IDs cannot recover learned ID-specific effects.

Use `NeuralEmbeddingFeatures` or `NeuralEmbeddingRegressor` when embeddings
should become dense columns for another model. That workflow is useful for
pickup zones, dropoff zones, pickup-dropoff pairs, zone-hour buckets, or trip
clusters that have repeated signal.

See [Neural Embedding Models And Features](../neural-features.md).

## Graph Models

Graph support is useful when the prediction depends on relationships rather
than only row features. Taxi examples include `PULocationID -> DOLocationID`,
pickup-zone to hour, dropoff-zone to hour, and pickup-dropoff pair nodes.

Standalone graph regressors:

| Model | Use when |
| --- | --- |
| `Node2VecStandaloneRegressor` | Directed or weighted graph structure is useful and node attributes are not required. |
| `GraphSageStandaloneRegressor` | A homogeneous graph has node attributes such as airport flag, borough, or recent volume. |
| `HeteroGraphSageStandaloneRegressor` | Edges have relation IDs, but you do not need strict node-type schema validation. |
| `HinSageStandaloneRegressor` | Nodes and relations are typed and source-target type constraints matter. |

Standalone link predictors:

| Model | Use when |
| --- | --- |
| `Node2VecLinkPredictor` | You need candidate source-target scores from graph topology. |
| `GraphSageLinkPredictor` | Link scores should use node attributes on a homogeneous graph. |
| `HeteroGraphSageLinkPredictor` | Link scores depend on typed edge relations. |
| `HinSageLinkPredictor` | Link scores require typed nodes and typed relation triples. |

```python
import numpy as np
from cartoboost.graph import Node2VecStandaloneRegressor

edges = [(132, 161), (161, 236), (132, 236)]
pickup = np.asarray([132, 161, 132], dtype=np.uint64)
dropoff = np.asarray([161, 236, 236], dtype=np.uint64)
log_duration = np.asarray([3.8, 4.1, 4.4])

model = Node2VecStandaloneRegressor(
    dim=8,
    walk_length=8,
    walks_per_node=4,
    window_size=3,
    epochs=2,
    n_estimators=20,
    max_depth=2,
    min_samples_leaf=1,
)
model.fit(
    node_count=300,
    edges=edges,
    row_nodes=pickup,
    row_targets=dropoff,
    y=log_duration,
)
pred = model.predict(pickup, row_targets=dropoff)
```

If you want graph embeddings as feature columns instead of a standalone graph
artifact, use the graph feature-generation path with `GraphFeatureTransformer`
and the encoder family that matches the graph.

See [Graph Models And Features](../graph-features.md).

## General Forecasting Utilities

Some Rust-backed utilities are plain functions rather than model classes. Use
them for quick sequence forecasts or low-level building blocks:

| Utility | Use when |
| --- | --- |
| `cartoboost.naive_forecast`, `seasonal_naive_forecast`, `theta_forecast`, `optimized_theta_forecast`, `ets_forecast`, `arima_forecast`, `auto_arima_forecast` | You have one numeric sequence and do not need a fitted estimator object. |
| `cartoboost.local_level_kalman_filter`, `local_level_kalman_forecast` | You need local-level filtering or forecasting. |
| `cartoboost.kalman_filter`, `local_linear_trend_kalman_forecast` | You need local-linear trend state estimates or forecasts. |
| `cartoboost.croston_forecast`, `sba_forecast`, `tsb_forecast`, `intermittent_demand_forecast` | Taxi-zone or service-area demand is non-negative and intermittent. |
| `cartoboost.ordinary_kriging_predict` | You need one-off spatial interpolation over observed coordinate values. |

Use model classes when you need `fit`, `predict`, metadata, artifacts,
backtesting, or ensemble composition. Use utility functions for direct numeric
calculations.

See [General Utilities](../general_utilities.md).

## Validation Defaults

Whichever model family you choose, validate it against a serious baseline under
the same split:

| Model family | Minimum comparison |
| --- | --- |
| Tabular regression | Mean baseline plus LightGBM or XGBoost on the same features and split. |
| Forecasting | Naive or seasonal naive plus rolling-origin backtests. |
| Neural embeddings | A non-neural `CartoBoostRegressor` under random, temporal, and cold-ID splits where relevant. |
| Graph models | A tabular route model and a graph-free ID or zone baseline. |

For NYC taxi work, report the target, split, row count, features, RMSE, MAE,
R2 when applicable, train time, prediction time, and exact command or notebook
entry point used to produce the numbers.
