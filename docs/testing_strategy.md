# Testing Strategy

GeoBoost validation is intended to prove scoped behavior, not production
superiority.

## Local Source Of Truth

```sh
just validate
```

The validation path covers:

- Rust formatting, clippy, and workspace tests.
- Python ruff formatting/checks and pytest.
- Native extension installation before native validation.
- Full validation artifact generation.
- v1 validation report generation.
- Criterion benchmark compile.

## Unit And Integration Tests

Rust tests cover:

- Loss and gain calculations.
- Split routing for axis, spatial, periodic, fuzzy, and sparse-list paths.
- Fractional fuzzy child weights.
- Linear leaf fitting.
- Save/load prediction identity.
- Mixed dense+sparse dataset behavior.

Python tests cover:

- Estimator parameter validation.
- Dense fit/predict/save/load.
- Native sparse-list train/predict/save/load.
- Feature schema conversion.
- Sample weights.
- sklearn compatibility.
- CLI and validation script integration.

## Validation Artifacts

`scripts/run_full_validation.py` writes artifacts under `target/validation/` and
regenerates deterministic proof images and metric reports. Fixtures include
axis threshold, diagonal boundary, Gaussian/radial, periodic wraparound, fuzzy
boundary, linear leaves, sparse sets, learning-rate shrinkage, and lane-level
combined behavior.

`scripts/run_v1_validation.py` is the v1 release-candidate report entry point.
It should report model config, baseline result, GeoBoost result, acceptance
gate, what each fixture proves, and what it does not prove.

## Property And Fuzz Testing

Property tests should prefer deterministic seeds and strict tolerances:

- Save/load prediction identity: `atol <= 1e-12`.
- Python/Rust parity: `atol <= 1e-12`.
- Fuzzy branch weights sum to `1` within `1e-12`.
- Periodic routing is invariant under period shifts where applicable.
- Duplicate sparse IDs do not change predictions.

Fuzz harnesses live under `fuzz/` and are expected to compile for v1. Longer
fuzz campaigns are post-v1 hardening unless explicitly scheduled.

## Benchmarks

Criterion benchmarks must compile in CI. Benchmark numbers are diagnostic and
must not be presented as broad performance claims without a documented,
reproducible comparison setup.
