# Fair Benchmarking Program

## Purpose

This page defines the evidence bar for public CartoBoost benchmark claims. The
short version: a benchmark writeup must show the data, command, model roster,
feature access, split, metrics, timing, artifacts, plots, interpretation, and
limitations.

Protocol-only pages are not enough. Readers need the numbers and the pictures.

## Required Benchmark Writeup

Every committed benchmark refresh must include:

- exact command;
- data source and extraction period;
- sample size, task rows, and split rows;
- target definition and target transform;
- model roster and model settings;
- baseline feature-access policy;
- primary and secondary metric tables;
- timing or throughput breakdown;
- artifact paths for JSON, Markdown, and images;
- embedded plots or screenshots where plots exist;
- winner, tie, or failure interpretation;
- limitations and claim boundary.

## Fairness Rules

A public quality claim needs more than a single score:

- same train/test rows for every model;
- comparable feature access;
- no test-set peeking;
- complete required baselines;
- equal model-selection budget when HPO is used;
- repeated seeds, folds, or uncertainty intervals for broad claims;
- hard failures when required real data is missing.

If a required baseline fails, the benchmark is incomplete unless the failure is
itself the result being audited.

## Track Contracts

| Track | Contract location | Required evidence |
| --- | --- | --- |
| Tabular supervised | `benchmarks/tracks/tabular/` | Public tabular datasets, repeated folds, GBDT and deep-tabular baselines. |
| Spatial and repeated-ID | `benchmarks/tracks/spatial/` | NYC taxi data, spatial/cold-route splits, subgroup slices, serious GBDT baseline. |
| Graph structured | `benchmarks/tracks/graph/` | Public graph split/evaluator, graph and tabularized baselines. |
| Forecasting | `benchmarks/tracks/forecasting/` | Rolling-origin splits, horizon metrics, statistical and global baselines. |

## Claim Boundary

Use cautious language:

- "On this maintained January 2024 TLC artifact..."
- "On this synthetic diagnostic fixture..."
- "This row ties the best seasonal baseline..."
- "This run does not include equal-budget HPO or confidence intervals..."

Avoid broad claims unless the benchmark actually has the data breadth,
repeatability, and statistical evidence to support them.
