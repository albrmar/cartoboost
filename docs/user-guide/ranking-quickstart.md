# Ranking Quickstart

Use `CartoBoostRanker` when labels are comparable only inside a query group:
candidate dropoff zones for one pickup, route alternatives for one customer, or
ranked service actions inside one planning context. The Rust backend trains
pairwise logistic or LambdaRank objectives and the Python wrapper handles group
validation and artifact loading.

## Fit A Grouped Ranker

Rows for each query must be contiguous. Pass `groups` as positive group sizes
that sum to the row count, or as one query id per row when the values do not
form a valid size vector. If the query id is a column in `X`, use `group_col`
and CartoBoost will remove it from the feature matrix and from dense
`feature_schema` entries before fitting. Prediction accepts either the full
frame with that column still present or the already-dropped feature matrix.

```python
from cartoboost import CartoBoostRanker

ranker = CartoBoostRanker(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=4,
    min_samples_leaf=10,
    objective="lambdarank",
    splitters=["axis", "diagonal_2d", "gaussian_2d"],
)
ranker.fit(X_train, relevance_train, groups=query_sizes_train)

scores = ranker.predict(X_validation)
```

Use `objective="pairwise_logit"` for unweighted pairwise logistic gradients and
`objective="lambdarank"` when NDCG-weighted gradients match the evaluation
claim.

## Evaluate

```python
from cartoboost import mean_average_precision, mean_reciprocal_rank, ndcg_at_k

native_metrics = ranker.score_groups(
    X_validation,
    relevance_validation,
    groups=query_sizes_validation,
)

check_metrics = {
    "ndcg@10": ndcg_at_k(relevance_validation, scores, groups=query_sizes_validation, k=10),
    "map": mean_average_precision(relevance_validation, scores, groups=query_sizes_validation),
    "mrr": mean_reciprocal_rank(relevance_validation, scores, groups=query_sizes_validation),
}
```

Compare against a baseline scoring rule inside each query group, such as
historic route conversion, distance-only score, or a standard tabular ranker
trained on the same group split.

## Save And Load

```python
ranker.save("route-ranker.json")
restored = CartoBoostRanker.load("route-ranker.json")
restored.predict(X_validation)
```

Ranker artifacts include native objective metadata, `group_col` metadata, and
categorical feature mappings when categorical columns are present.
