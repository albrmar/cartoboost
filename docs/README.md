# GeoBoost Documentation

This documentation captures the implementation plan, current status, fixture
contracts, and integration-test contracts for the clean-room GeoBoost-inspired
repo.

## Contents

- [Fixture Contract](fixture-contract.md) describes the committed test data and
  golden outputs under `tests/`.
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
fuzzy, sparse, and linear-leaf support, plus documented future hardening items.
