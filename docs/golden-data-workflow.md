# Golden Data Workflow

Golden files make GeoBoost behavior explicit. Update them only when the intended
scoring contract changes or when a fixture is deliberately added.

## Updating Goldens

1. Change the source fixture under `tests/fixtures/`.
2. Recompute the expected output from the documented formula in
   [Fixture Contract](fixture-contract.md).
3. Update the matching file under `tests/goldens/`.
4. Review the diff by feature id and score, not only by file-level replacement.

## Review Checklist

- Every golden result references an existing source fixture id.
- Ranking is deterministic when scores tie.
- Rounding mode and precision are documented in the golden `config`.
- New scenarios include a short description in `docs/fixture-contract.md`.

## Test Boundaries

The scaffold contains two test categories:

- `tests/test_fixture_contract.py` validates committed fixture and golden shape.
- `tests/integration/test_weighted_overlay_contract.py` documents expected
  implementation behavior and skips until the Python API exists.
