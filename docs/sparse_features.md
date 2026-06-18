# Sparse Features

CartoBoost supports list-valued sparse columns. Each row can contain zero or
more non-negative integer IDs for route cells, zones,
encoded H3 cells, grid cells, corridors, or other memberships.

This is useful when a temporal-spatial row belongs to several places at once,
such as all route cells crossed by a trip. A generic tabular model usually
needs a wide one-hot or hashing step for this data; CartoBoost can consume the
lists directly.

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
from cartoboost import build_zip_sparse_sets

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
```

## Abstract Geo IDs

Arbitrary geo-id features like pickup/dropoff zones can be mapped directly into
sparse columns:

```python
from cartoboost import build_geo_sparse_sets

geo_sparse_sets = build_geo_sparse_sets(
    {
        "pickup_zone": ["Z1", "Z2", "Z3"],
        "delivery_zone": ["D1", "D2", "D3"],
    },
    namespace="market_a",
)
```

State, county, region, market, and zone features use the same path:

```python
state_geo_sparse_sets = build_geo_sparse_sets(
    {
        "state": ["CA", "NY", "CA", "TX"],
        "region": ["NORTH", "NORTHEAST", "WEST", "SOUTH"],
    },
    namespace="policy",
)
```

Pass these through `sparse_sets=` and declare each column as a sparse-set kind:

```python
schema = {
    "dense": [{"name": "distance_m", "kind": "numeric"}],
    "sparse_sets": [
        {"name": "pickup_zone", "kind": "zone_sparse_set"},
        {"name": "delivery_zone", "kind": "region_sparse_set"},
        {"name": "state", "kind": "geo_sparse_set"},
        {"name": "region", "kind": "zone_sparse_set"},
    ],
}
```

`build_geo_sparse_sets` is deterministic: the `(namespace, column_name, value)`
triple is hashed to a stable non-negative feature ID, so repeated labels map to
the same ID.

Validation rules:

- Each sparse column must have the same row count as `X` and `y` during fit.
- Each sparse prediction column must have the same row count as `X`.
- IDs must be non-negative integers.
- Duplicate IDs in a row are sorted and deduplicated before training.
- A model that learned sparse-list splits requires `sparse_sets=` for prediction.

## H3 Sparse Helpers

`FeatureSchema` accepts sparse entries with `kind="h3_sparse_set"` plus H3
metadata when the caller has already encoded H3 cells:

```python
schema = FeatureSchema(
    dense=[("distance_m", "numeric")],
    sparse_sets=[
        {
            "name": "route_h3",
            "kind": "h3_sparse_set",
            "resolution": 9,
            "parent_resolutions": [5, 7],
        },
    ],
)
```

Saved schema metadata keeps the H3 resolution fields for callers and fitted
estimator metadata.

`cartoboost.h3.normalize_h3_id` accepts non-negative integer IDs plus decimal or
hexadecimal strings. CartoBoost does not compute H3 cells from latitude and
longitude; compute cells upstream, then pass the encoded IDs through
`sparse_sets=`.

## Routing Semantics

The dataset stores dense values and sparse-set columns separately. Sparse split
candidates check membership against the sparse row, not a dense one-hot
expansion.

Routing is:

```text
left  if row IDs contain any split ID
right otherwise
```

Empty rows and unseen IDs route as no match. Duplicate row IDs do not change the
route.

## CLI Scope

The CLI dense CSV workflow does not accept mixed sparse rows. Use the Python
estimator for sparse route-cell training and prediction.

## Limitations

- Candidate search currently considers one sparse ID per candidate.
- Sparse sets accept non-negative integer IDs in the model interface; helper utilities
  can map abstract geo labels (for example zone IDs) to stable numeric IDs.
- Sparse support is regression-only.
