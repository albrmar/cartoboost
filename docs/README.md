# GeoBoost Documentation

This documentation scaffold captures the fixture and integration-test contract
for the first GeoBoost implementation pass.

## Contents

- [Fixture Contract](fixture-contract.md) describes the committed test data and
  golden outputs under `tests/`.
- [Integration Contract](integration-contract.md) records the expected Python API
  shape that future implementation work should satisfy.
- [Golden Data Workflow](golden-data-workflow.md) explains how to update fixture
  expectations without weakening the test suite.
- [Implementation Status](implementation_status.md) states what is implemented
  and what remains planned.

## Current Scope

The repository contains a Milestone 1 regression implementation plus tested core
primitives for later spatial, temporal, fuzzy, sparse, and linear-leaf work.
