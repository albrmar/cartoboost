# Choose A Model

Use this page as the user-guide router. CartoBoost has several first-class
model surfaces, and the right entry point depends on the scientific structure
in the data: row-level place/time effects, regular time series, shared panels,
direct graph structure, or learned ID embeddings.

Most behavior is implemented in Rust and surfaced through thin Python wrappers.
Python handles dataframe ergonomics, validation, sklearn compatibility where
appropriate, and artifact helpers.

## Start With The Scientific Unit

| Question | Use | Primary guide |
| --- | --- | --- |
| Does each row describe one taxi trip, route observation, zone-hour aggregate, or residual to regress? | `cartoboost.CartoBoostRegressor` | [Python API Reference](../reference/python-api.md) |
| Are you choosing how place, time, sparse memberships, losses, fuzzy routing, or local residual trends enter that row-level model? | `CartoBoostRegressor` parameters | [Parameters](parameters.md) |
| Is the target one regular pickup-zone or lane series with its own history? | Local forecasters such as `SeasonalNaiveForecaster`, `ThetaForecaster`, `ETSForecaster`, `AutoARIMAForecaster`, or `KalmanForecaster` | [Forecasting Model Guides](forecasting-models/index.md) |
| Are many related pickup zones, dropoff zones, or route panels forecast from shared lag features? | `CartoBoostLagForecaster` | [CartoBoost Lag](forecasting-models/cartoboost-lag.md) |
| Should nearby coordinates borrow signal for a forecast panel? | `KrigingForecaster` | [Kriging](forecasting-models/kriging.md) |
| Do you need a fixed combination of fitted native forecasters? | `WeightedEnsembleForecaster` | [Weighted Ensembles](forecasting-models/ensembles.md) |
| Are stable pickup zones, dropoff zones, or pairs themselves the learned artifact? | `NeuralEmbeddingStandaloneRegressor` | [Neural Features](../neural-features.md) |
| Is the relationship network the object being modeled? | Graph standalone regressors or link predictors | [Graph Features](../graph-features.md) |
| Do graph or neural embeddings only need to become columns for another estimator? | `GraphFeatureTransformer`, `NeuralEmbeddingFeatures`, or `NeuralEmbeddingRegressor` | [Graph Features](../graph-features.md), [Neural Features](../neural-features.md) |
| Do you need one-off Rust-backed forecast or spatial utilities? | Functions such as `theta_forecast`, `kalman_filter`, or `ordinary_kriging_predict` | [General Utilities](../general_utilities.md) |

## When CartoBoostRegressor Fits

`CartoBoostRegressor` is the main sklearn-style estimator for row-level taxi
regression. It is a good scientific choice when the target is plausibly shaped
by structured place/time effects rather than only by generic dense covariates.
Examples include fare, duration, demand, or residual models where pickup and
dropoff zones, route memberships, hour-of-day, local neighborhoods, or fuzzy
service boundaries should be part of the model rather than hidden in many
preprocessing columns.

Prefer it for experiments where you want to ask questions such as:

- Do pickup/dropoff effects persist after controlling for trip distance, hour,
  and day features?
- Are sparse zones, routes, H3/S2 cells, or service areas informative even when
  many memberships are rare?
- Does a smooth transition near a learned spatial boundary reduce localized
  residual artifacts?
- Does an outlier-resistant or quantile objective match the scientific target
  more closely than mean regression?
- Can the fitted artifact preserve the schema, splitters, loss, fuzzy settings,
  sparse-set requirements, and additive values needed for later interpretation?

Do not treat this as a broad claim about CartoBoost versus LightGBM, XGBoost,
or a simpler baseline. Use those models as serious comparisons under the same
train/test split and feature set. Select CartoBoost only when the structured
controls satisfy the specific holdout or diagnostic that matters for the study.

## Tabular And Spatial Regression

Start with dense numeric columns for the measured quantities: trip distance,
projected pickup/dropoff coordinates, pickup hour, day of week, route-level
aggregates, fare history, or duration history. Add sparse-set features when a
row belongs to pickup zones, dropoff zones, H3/S2 cells, service areas, route
memberships, or overlapping operational regions.

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

Choose controls from the structure you want to test:

| Scientific need | Parameter family |
| --- | --- |
| Dense tabular baseline | `splitters=None`, `["auto"]`, `["axis"]`, or `["axis_histogram:<bins>"]` |
| Spatial boundaries in coordinates | `["axis", "diagonal_2d", "gaussian_2d"]` |
| Wraparound time effects | `["axis", "periodic:24"]` or another `periodic:<period>` |
| Sparse pickup/dropoff zones, routes, cells, or areas | `["axis", "sparse_set"]` plus `sparse_sets=` |
| Smooth changes near boundaries | `fuzzy=True`, `fuzzy_bandwidth=...`, `fuzzy_kernel=...` |
| Outlier-resistant regression | `loss="mae"`, `loss="huber"`, or `loss="log_l2"` |
| Conditional intervals or asymmetric service targets | `loss="quantile"`, `quantile_alpha=...` |
| Local residual trend inside learned regions | `leaf_predictor="linear"`, `linear_leaf_features=[...]` |
| Domain monotonicity | `monotonic_constraints=[...]` |

See [Python API Reference](../reference/python-api.md), [Parameters](parameters.md),
[Feature Schema](../feature_schema.md), [Sparse Features](../sparse_features.md),
and [Temporal-Spatial Modeling](../spatial_modeling.md).

## Forecasting

Forecasting has two documentation layers:

| Layer | Covers | Start here |
| --- | --- | --- |
| Forecasting wrapper | `ForecastFrame`, dataframe conversion, rolling-origin backtesting, forecast metrics, artifacts, CLI workflows, and leakage checks | [Forecasting](../forecasting.md) |
| Forecasting model guides | Model-specific examples and tuning notes for native forecasting classes | [Forecasting Model Guides](forecasting-models/index.md) |

Use `ForecastFrame.from_pandas` for production taxi demand or fare-duration
workflows because it validates timestamps, frequency, duplicate rows, target
values, panel ids, and covariate roles:

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

Choose the model guide by series structure:

| Series structure | Model guide |
| --- | --- |
| Last value or last season is the benchmark | [Naive And Seasonal Naive](forecasting-models/naive-seasonal.md) |
| Lightweight trend extrapolation | [Theta](forecasting-models/theta.md) |
| Additive level, trend, or seasonality | [ETS](forecasting-models/ets.md) |
| Autocorrelation and differencing | [ARIMA And AutoARIMA](forecasting-models/arima.md) |
| Noisy local level and trend | [Kalman](forecasting-models/kalman.md) |
| Interpretable trend, changepoints, seasonality, events, and regressors | [Piecewise Linear Seasonal](forecasting-models/piecewise-linear-seasonal) |
| Coordinate-aware panel interpolation | [Kriging](forecasting-models/kriging.md) |
| Shared supervised lag model across many series | [CartoBoost Lag](forecasting-models/cartoboost-lag.md) |
| Guarded default selector over native candidates | [AutoForecaster](forecasting-models/auto-forecaster.md) |
| Fixed-weight combinations of native forecasters | [Weighted Ensembles](forecasting-models/ensembles.md) |

Current Rust bindings intentionally reject unsupported modes such as damped ETS,
multiplicative ETS, seasonal AutoARIMA, and Python fallback algorithms.

## Graph And Neural Models

Graph and neural standalone models are direct APIs, not just feature builders
for `CartoBoostRegressor`.

Use `NeuralEmbeddingStandaloneRegressor` when the learned ID embedding is the
artifact to train, score, save, and serve. This works best when train and
prediction populations share stable IDs such as pickup zones, dropoff zones,
pickup-dropoff pairs, zone-hour buckets, or trip clusters. Under cold-zone or
cold-ID holdouts, report fallback behavior explicitly because unseen IDs cannot
recover learned ID-specific effects.

Use graph standalone regressors when relationships matter:

| Model | Use when |
| --- | --- |
| `Node2VecStandaloneRegressor` | Directed or weighted topology is useful and node attributes are not required. |
| `GraphSageStandaloneRegressor` | A homogeneous graph has node attributes such as airport flag, borough, or recent volume. |
| `HeteroGraphSageStandaloneRegressor` | Edges have relation IDs, but strict node-type schema validation is not required. |
| `HinSageStandaloneRegressor` | Nodes and relations are typed and source-target type constraints matter. |

Use graph or neural feature generators only when embeddings should become dense
columns for another model.

See [Graph Features](../graph-features.md) and [Neural Features](../neural-features.md).

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

## Recommended Reading Order

1. Read [Getting Started](../getting-started.md) for installation, the first
   model fit, and local validation commands.
2. Use this chooser to pick the model family.
3. For row-level regression, read the [Python API Reference](../reference/python-api.md), then
   [Parameters](parameters.md), then [Temporal-Spatial Modeling](../spatial_modeling.md)
   and the relevant feature pages.
4. For time-series work, read the [Forecasting](../forecasting.md) page
   when you need `ForecastFrame`, backtesting, forecast artifacts, or the CLI.
   Read [Forecasting Model Guides](forecasting-models/index.md) when you need examples for
   a specific model class.
5. For graph or neural work, start with the standalone model sections before
   using feature-generation helpers.
