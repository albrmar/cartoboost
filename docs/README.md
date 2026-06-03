# GeoBoost Documentation

This documentation captures the v1 API contract, implementation status, fixture
contracts, testing strategy, and release checklist for the clean-room
GeoBoost-inspired repo.

## Contents

- [Fixture Contract](fixture-contract.md) describes the committed test data and
  golden outputs under `tests/`.
- [v1 API](v1_api.md) describes the supported Python, Rust-backed, and CLI
  public surfaces.
- [Sparse Features](sparse_features.md) documents list-valued route-cell-style
  sparse columns.
- [Feature Schema](feature_schema.md) documents numeric, periodic, and
  sparse-set schema declarations.
- [Model Artifact](model_artifact.md) documents native JSON artifact contents
  and compatibility scope.
- [Testing Strategy](testing_strategy.md) describes unit, integration,
  validation, fuzz, and benchmark expectations.
- [Limitations](limitations.md) states the explicit alpha/v1-candidate limits.
- [v1 Release Checklist](v1_release_checklist.md) tracks release-candidate gates.
- [Repository Plan](repo_plan.md) records the target product, architecture,
  milestone plan, testing philosophy, and definition of done.
- [Integration Contract](integration-contract.md) records the expected Python API
  shape that future implementation work should satisfy.
- [Golden Data Workflow](golden-data-workflow.md) explains how to update fixture
  expectations without weakening the test suite.
- [Implementation Status](implementation_status.md) states what is implemented
  and what remains planned.
- [Segmentation Proofs](assets/) contains generated PNGs showing learned
  spatial segmentation boundaries on deterministic synthetic datasets.

## Current Scope

The repository contains a regression implementation with spatial, temporal,
fuzzy, sparse, schema-aware, and linear-leaf support, plus documented future
hardening items. It is a clean-room implementation and does not claim
equivalence to Lyft's proprietary GeoBoost.
