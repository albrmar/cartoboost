# Neural Embedding Strategy

## What It Covers

Neural residual embeddings are a specialist option for repeated pickup zones,
dropoff zones, lanes, or route IDs. They are not the default CartoBoost claim
path. Present them as cold-start generalization only when the cold-ID split
shows it.

Use these rules when a benchmark includes neural embedding rows.

## Required Safeguards

Use:

- out-of-fold residual targets for embedding training;
- train-only encodings and train-only fallback statistics;
- support-aware shrinkage for rare IDs;
- explicit cold-ID or cold-route splits when deployment may see unseen IDs;
- repeated-ID splits when deployment will reuse known zones or routes;
- separate reporting for random, temporal, spatial, grouped, and cold-ID
  regimes.

If any safeguard is missing, the neural result is exploratory.

## Allowed Claims

Allowed:

- "Embeddings pass repeated-ID validation under this split."
- "Embeddings do not pass cold-zone validation."
- "The fallback hierarchy reduces leakage risk compared with in-sample residual
  lookup."

Not allowed:

- "Neural embeddings are a universal upgrade."
- "Random repeated-ID gains imply cold-ID performance."
- "A neural row should be compared against weaker baseline feature access."

## Metrics

Report the same primary metric as the parent benchmark, usually RMSE or MAE.
Also report:

- cold-ID gap;
- rare-ID gap;
- support decile slices;
- fit and prediction cost;
- repeatability across folds or seeds when available.

## Model Choice

Choose neural residual embeddings only when the production problem has recurring
IDs with enough training support and the repeated-ID split passes without
damaging the deployment split that matters. Choose plain CartoBoost or a
simpler baseline when IDs are mostly cold.
