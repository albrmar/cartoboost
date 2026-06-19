# Benchmarks

Use these pages to answer one question: can this result guide model choice?
Real taxi results can, if the split, features, baselines, metrics, and command
are clear. Synthetic results are useful for debugging, but they are not evidence
for real taxi deployment.

## Reports

| Page | Purpose |
| --- | --- |
| [NYC Taxi Benchmarks](nyc-taxi.md) | Real TLC fare, duration, and demand tasks. |
| [Forecasting Tool Benchmark](forecasting.md) | Rolling-origin taxi lane demand and public forecasting datasets. |
| [Model Benchmark Suite](model-suite.md) | Synthetic checks for dense, repeated-ID, and graph behavior. |
| [Taxi Zone Acceptance](taxi-zone.md) | Deterministic acceptance checks for lane membership, route geometry, and cyclic hour behavior. |
| [Neural Embedding Strategy](neural-embedding-strategy.md) | Rules for evaluating leakage-aware residual embeddings. |
| [Neural Embedding Benchmark](neural-embedding-benchmark-latest.md) | Synthetic repeated-ID and cold-ID checks. |

## Claim Rules

A CartoBoost claim requires the CartoBoost row to satisfy the primary metric
threshold under the same split, comparable feature access, comparable tuning
budget, and complete baseline set. A missing required baseline makes the
benchmark incomplete.

Random splits can show interpolation quality. Public claims about taxi
deployment need the split that matches deployment risk: out-of-time for future
demand, spatial holdout for new pickup zones, grouped holdout for new routes or
lanes, and cold-ID splits when IDs will be unseen.

Synthetic fixtures can prove a feature family is wired correctly or catch a
regression. They cannot prove broad accuracy.
