# Neural Embedding Strategy Benchmark

## Research Question

Can residual embeddings improve repeated-ID prediction while reducing leakage
and avoiding false cold-start claims?

## Motivation

Earlier neural embedding experiments behaved like an in-sample residual lookup
table. That inflated repeated-ID validation and did not translate cleanly to
spatial holdouts. The revised strategy adds out-of-fold training, shrinkage,
multi-key IDs, hierarchical fallback, and graph-aware neighbor fallback.

## Dataset Context

The strategy is evaluated on two families of benchmarks:

- NYC taxi tasks: duration, fare, and pickup demand with pickup/dropoff zone
  context and spatial holdouts.
- Synthetic neural-ID tasks: controlled repeated-ID and group-holdout splits.

## Target

The target depends on the benchmark:

- NYC duration and fare use transformed continuous trip targets.
- NYC pickup demand uses transformed zone-time demand.
- Synthetic neural-ID uses a continuous target with ID-specific residual
  signal.

## Feature Strategy

The current neural path supports:

- Out-of-fold residual embeddings.
- Support-aware shrinkage for rare IDs.
- Multi-key embeddings such as pickup zone, dropoff zone, and lane.
- Hierarchical fallback IDs.
- Neighbor-based fallback for graph or spatial adjacency.

## Results: Before vs After

| Benchmark | Split | Old neural R2 | New neural R2 | Change |
| --- | --- | ---: | ---: | ---: |
| NYC duration | random | `0.7118` | `0.7272` | `+0.0153` |
| NYC duration | spatial holdout | `0.5862` | `0.6815` | `+0.0953` |
| NYC fare | random | `0.8391` | `0.8492` | `+0.0101` |
| NYC fare | spatial holdout | `0.7929` | `0.8104` | `+0.0175` |
| NYC pickup demand | random | `0.8733` | `0.8814` | `+0.0081` |
| Synthetic neural ID | random | `0.9353` | `0.9217` | `-0.0136` |
| Synthetic neural ID | group holdout | `0.7790` | `0.7834` | `+0.0044` |

## Current Standing

| Benchmark | Split | Best relevant result | Neural result | Read |
| --- | --- | ---: | ---: | --- |
| NYC duration | random | CartoBoost `0.7337` | `0.7272` | Improved, but plain CartoBoost remains better. |
| NYC duration | spatial holdout | CartoBoost `0.6919` | `0.6815` | Large neural improvement, still behind plain CartoBoost. |
| NYC fare | random | Hetero graph `0.8514` | `0.8492` | Neural is competitive; graph is slightly best. |
| NYC fare | spatial holdout | Hetero graph `0.8140` | `0.8104` | Neural is close to plain CartoBoost; graph is best. |
| NYC pickup demand | random | Neural `0.8814` | `0.8814` | Neural is best on the valid demand split. |
| Synthetic neural ID | random | Neural `0.9217` | `0.9217` | Neural still wins where IDs repeat. |
| Synthetic neural ID | group holdout | LightGBM `0.8875` | `0.7834` | Neural still loses on true cold IDs. |

## Interpretation

The revised strategy reduces leakage risk. The synthetic random score falls
slightly because out-of-fold embeddings remove some in-sample residual
memorization. That is a better quality signal, not a regression in the
benchmark design.

Use neural embeddings when IDs repeat, residual structure remains after base
features, and there is a meaningful fallback hierarchy or graph. Do not use
them as the default when the production problem is true cold-ID prediction.

## Decision

Plain CartoBoost remains the safest default. Neural embeddings are an optional
strategy for repeated-ID residual signal and zone-demand style tasks. Report
neural results with the split protocol because repeated-ID gains are not
cold-start generalization.
