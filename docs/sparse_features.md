# Sparse Features

GeoBoost supports list-valued sparse columns through the Rust backend. Each row
can contain zero or more non-negative integer IDs for route cells, zones,
encoded H3 cells, grid cells, corridors, or other memberships.

This is useful when a temporal-spatial row belongs to several places at once,
such as all route cells crossed by a trip. A generic tabular model usually
needs a wide one-hot or hashing step for this data; GeoBoost can consume the
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

Validation rules:

- Each sparse column must have the same row count as `X` and `y` during fit.
- Each sparse prediction column must have the same row count as `X`.
- IDs must be non-negative integers.
- Duplicate IDs in a row are sorted and deduplicated by the Rust dataset layer.
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

The Rust payload still uses the existing `"SparseSet"` feature kind, while the
Python schema metadata keeps the H3 resolution fields for callers and saved
estimator metadata.

`geoboost.h3.normalize_h3_id` accepts non-negative integer IDs plus decimal or
hexadecimal strings. GeoBoost does not compute H3 cells from latitude and
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
- Sparse IDs are integer route-cell-style identifiers, not arbitrary strings.
- Sparse support is regression-only and native-backend-only.
