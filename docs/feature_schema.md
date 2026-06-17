# Feature Schema

Feature schemas tell CartoBoost which columns are ordinary numeric features,
which dense columns wrap around like time, and which columns are list-valued
sparse memberships. They are especially useful for temporal-spatial models where
column roles matter.

## Compact Python Format

```python
schema = {
    "dense": [
        {"name": "pickup_x", "kind": "numeric"},
        {"name": "pickup_y", "kind": "numeric"},
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
  "names": ["pickup_x", "pickup_y", "distance_m", "hour_of_day", "route_cells"],
  "kinds": ["Numeric", "Numeric", "Numeric", {"Periodic": {"period": 24}}, "SparseSet"]
}
```

## Validation Rules

- Schema length must equal dense feature count plus sparse-set column count.
- `kind` must be numeric, periodic, or sparse-set.
- Geographic sparse identifiers can be declared with
  `zip_sparse_set`, `zip3_sparse_set`, `zone_sparse_set`, `region_sparse_set`,
  `h3_sparse_set`, or equivalent aliases (`ZipSparseSet`, `ZoneSparseSet`,
  `RegionSparseSet`, `H3SparseSet`, `GeoSparseSet`, `GeoAbstractSparseSet`).
  This is suitable for state, zone, county, market, region, and similar ID fields.
  All listed alias kinds resolve to the underlying sparse-set feature type.
- Periodic entries require a positive period.
- Sparse-set entries correspond to sparse columns supplied through
  `sparse_sets=`.

## Training Behavior

When a schema is present:

- Periodic splitters use declared periods and do not rely on observed values
  covering a full cycle.
- Sparse splitters prefer schema-declared sparse-set columns.
- Numeric dense columns remain eligible for numeric/spatial split candidates.

For example, declaring `hour_of_day` with `period=24` lets `periodic:24` treat
late-night and early-morning rows as neighboring values. Declaring
`route_cells` as sparse-set tells CartoBoost to use list membership instead of
expecting a scalar numeric feature.

## Limitations

The current schema does not express named latitude/longitude pairs or richer
spatial roles. Diagonal and Gaussian splitters still work from dense numeric
feature pairs according to the current candidate search.
