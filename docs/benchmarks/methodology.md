# Benchmark Methodology

CartoBoost benchmark claims must be tied to real commands, fixed comparable
settings, and recorded artifacts. A benchmark page should let a reader see what
was run, which data were used, which split was evaluated, which models were
compared, and which metric supports the interpretation.

## Required Fields

Each benchmark report should name:

- command and working directory;
- data source and whether the data are real, synthetic, or generated acceptance
  data;
- sample size and task definitions;
- train/test split or CV fold construction;
- model roster and comparable estimator settings;
- metric table with timing fields;
- artifact paths for JSON, JSONL, markdown, and plots;
- limitations that affect interpretation.

## v0.2 Modeling Gates

For the v0.2 spatial boosting release, the maintained benchmark set should
include:

- binary spatial classification versus dummy and tabular baselines;
- grouped ranking versus baseline scoring;
- native categorical versus one-hot preprocessing;
- random CV versus buffered spatial CV leakage comparison;
- regression benchmark showing no more than 5 percent slowdown on the existing
  regressor workload.

The deterministic smoke harness for these release gates is:

```sh
PYTHONPATH=python uv run --group dev python scripts/run_v02_modeling_benchmarks.py \
  --output-dir target/v02-benchmarks \
  --seed 42 \
  --sample-size 240 \
  --n-estimators 24
```

To turn the regression fit-speed guard from a current-code repeatability check
into a before/after slowdown check, pass a prior artifact from the same harness:

```sh
PYTHONPATH=python uv run --group dev python scripts/run_v02_modeling_benchmarks.py \
  --output-dir target/v02-benchmarks \
  --regression-baseline-json target/v02-baseline/v02_modeling_benchmark.json
```

It writes:

- `target/v02-benchmarks/v02_modeling_benchmark.json`
- `target/v02-benchmarks/v02_modeling_benchmark.jsonl`
- `target/v02-benchmarks/v02_modeling_benchmark.md`

The output is synthetic taxi-shaped smoke evidence. It proves that the release
gates execute and fail loudly when they should; it does not replace real NYC
TLC benchmark artifacts for public quality claims. When no
`--regression-baseline-json` is supplied, the regression guard records
`evidence_kind=current_code_repeatability` and should not be interpreted as a
historical slowdown comparison.

Classification reports should include logloss, ROC-AUC or PR-AUC, Brier score,
ECE, fit time, prediction time, and save/load probability drift. Ranking
reports should include NDCG, MAP, MRR, fit time, prediction time, and save/load
score drift. Categorical reports should state the number of categories, chosen
encoding strategy, unknown-category rate, and whether the saved model
round-tripped predictions within tolerance. Unsupported export checks should
assert loud `NotImplementedError` failures for categorical regressor export and
classifier/ranker portable-weight or ONNX export.

## Interpretation Rules

Do not use stale artifacts after changing benchmark-affecting code. If feature
generation, fitting, prediction, metric computation, or split construction
changes, rerun the affected benchmark before updating public claims.

Do not frame benchmark pages around process labels such as cleanup or
provenance. Lead with the current-code result, then show command, data, split,
model roster, metrics, timing, artifact paths, and limits.
