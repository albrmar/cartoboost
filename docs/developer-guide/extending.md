# Extending GeoBoost

Use this page as a map for common changes. Keep docs and tests in the same
change when public behavior moves.

## Add A Splitter

1. Implement candidate generation and routing in `crates/geoboost-core/src/splitters`.
2. Wire the splitter kind into core configuration and serialization.
3. Expose and validate the name in `python/geoboost/regressor.py`.
4. Add PyO3 conversion if the binding surface needs new data.
5. Add CLI config parsing only if the dense CSV interface can support the
   splitter safely.
6. Add Rust routing tests, Python validation tests, and save/load identity
   tests.
7. Update [Parameters](../user-guide/parameters.md) and
   [v1 API Contract](../v1_api.md).

## Add A Loss

1. Add the loss config and residual/leaf behavior in `crates/geoboost-core/src/loss`.
2. Define how sample weights affect the objective.
3. Decide which leaf predictors are valid.
4. Add Python validation and native binding arguments.
5. Add artifact round-trip tests and validation fixtures if the behavior is
   public.

## Add Artifact Fields

Artifact version `1` allows backward-compatible optional metadata. Required
field changes or incompatible semantics need a new artifact version and explicit
loader failures for unknown versions.

Tests should cover:

- Prediction identity after save/load.
- Missing optional fields from older artifacts.
- Clear failure for unsupported versions.
- Sparse-list prediction safety when a model requires sparse sets.

## Add Python API Surface

Public estimator changes should preserve sklearn compatibility:

- `__init__` stores parameters without side effects.
- `get_params` and `set_params` include every public constructor parameter.
- Validation errors use `ValueError` for invalid user input.
- Native-only behavior fails clearly when the native extension is unavailable.

## Add CLI Behavior

The CLI is intentionally scoped to dense numeric CSV. New options should have:

- Explicit option validation.
- JSON and CSV output behavior when command output changes.
- Integration tests for invalid inputs and success paths.
- Documentation in [CLI](../user-guide/cli.md) and [CLI Reference](../reference/cli.md).
