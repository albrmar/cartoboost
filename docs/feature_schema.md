# Feature Schema

Feature schemas make training contracts explicit for dense numeric, dense
periodic, and sparse-set columns.

## Compact Python Format

```python
schema = {
    "dense": [
        {"name": "distance_m", "kind": "numeric"},
        {"name": "hour_of_day", "kind": "periodic", "period": 24},
    ],
    "sparse_sets": [
        {"name": "route_cells", "kind": "sparse_set"},
        {"name": "ozip_zip_p3", "kind": "zip_sparse_set"},
        {"name": "dzip_zip5", "kind": "h3_sparse_set"},
    ],
}
```

Pass the schema to native training:

```python
model.fit(
    X_dense,
    y,
    sparse_sets={"route_cells": route_cells},
    feature_schema=schema,
)
```

## Rust Artifact Format

The Python wrapper converts supported schema dictionaries into the Rust schema
payload:

```json
{
  "names": ["distance_m", "hour_of_day", "route_cells"],
  "kinds": ["Numeric", {"Periodic": {"period": 24}}, "SparseSet"]
}
```

## Validation Rules

- Schema length must equal dense feature count plus sparse-set column count.
- `kind` must be numeric, periodic, or sparse-set.
- Geographic sparse identifiers may be declared with `zip_sparse_set`, `zip3_sparse_set`,
  `h3_sparse_set`, or equivalent aliases (`ZipSparseSet`, `H3SparseSet`, `GeoSparseSet`)
  and are sent to
  the trainer as sparse-set features.
- Periodic entries require a positive period.
- Sparse-set entries correspond to sparse columns supplied through
  `sparse_sets=`.

## Training Behavior

When a schema is present:

- Periodic splitters use declared periods and do not rely on observed values
  covering a full cycle.
- Sparse splitters prefer schema-declared sparse-set columns.
- Numeric dense columns remain eligible for numeric/spatial split candidates.

## Limitations

The current schema does not yet express named latitude/longitude pairs or richer
spatial roles. Diagonal and Gaussian splitters still work from dense numeric
feature pairs according to the current Rust candidate search.
