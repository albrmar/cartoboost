# Sparse Features

GeoBoost supports route-cell-style sparse columns through the Rust backend.
Each sparse column is list-valued: every row contains zero or more non-negative
integer IDs.

## Python API

```python
route_cells = [[7, 11], [11], [3], []]

model.fit(
    X_dense,
    y,
    sparse_sets={"route_cells": route_cells},
)

pred = model.predict(
    X_dense_test,
    sparse_sets={"route_cells": [[7], [], [3, 9]]},
)
```

Zip code features can be expanded into geographic sparse-columns with explicit
origin/destination roles:

```python
from geoboost import build_zip_sparse_sets

zip_sparse_sets = build_zip_sparse_sets(
    origin_zip=["94103", "94122"],
    destination_zip=["10001", "94103"],
    parent_prefixes=(3, 2),
)

schema = {
    "dense": [{"name": "distance_m", "kind": "numeric"}],
    "sparse_sets": [
        {"name": "ozip_zip5", "kind": "zip_sparse_set"},
        {"name": "ozip_zip_p3", "kind": "zip_sparse_set"},
        {"name": "dzip_zip5", "kind": "zip_sparse_set"},
        {"name": "dzip_zip_p3", "kind": "zip3_sparse_set"},
    ],
}

model.fit(
    X_dense,
    y,
    sparse_sets=zip_sparse_sets,
    feature_schema=schema,
)

# Emit only ZIP3 hierarchy columns
zip3_only_sparse_sets = build_zip_sparse_sets(
    origin_zip=["94103", "94122"],
    destination_zip=["10001", "94103"],
    zip3_only=True,
)

Arbitrary geo-id features like pickup/dropoff zones can be mapped directly into
geographic sparse columns:

```python
from geoboost import build_geo_sparse_sets

geo_sparse_sets = build_geo_sparse_sets(
    {
        "pickup_zone": ["Z1", "Z2", "Z3"],
        "delivery_zone": ["D1", "D2", "D3"],
    },
    namespace="market_a",
)

schema = {
    "dense": [{"name": "distance_m", "kind": "numeric"}],
    "sparse_sets": [
        {"name": "pickup_zone", "kind": "zone_sparse_set"},
        {"name": "delivery_zone", "kind": "region_sparse_set"},
    ],
}
```
```

Validation rules:

- Each sparse column must have the same row count as `X` and `y` during fit.
- Each sparse prediction column must have the same row count as `X`.
- IDs must be non-negative integers.
- Duplicate IDs in a row are sorted and deduplicated by the Rust dataset layer.
- A model that learned sparse-list splits requires `sparse_sets=` for prediction.

## Rust Semantics

The Rust dataset stores dense values and sparse-set columns separately. Sparse
split candidates use `SparseListContainsAny` against the sparse row, not a dense
one-hot expansion.

Routing is:

```text
left  if row IDs contain any split ID
right otherwise
```

Empty rows and unseen IDs route as no match. Duplicate row IDs do not change the
route.

## CLI Scope

The CLI v1 dense CSV workflow does not accept mixed sparse rows. Use Python with
`backend="rust"` for sparse route-cell training and prediction.

## Limitations

- Candidate search currently considers one sparse ID per candidate.
- Sparse sets accept non-negative integer IDs in the model interface; helper utilities
  can map abstract geo labels (for example zone IDs) to stable numeric IDs.
- Sparse support is regression-only and native-backend-only.
