# Neural Embedding Strategy Assessment

This note summarizes the current neural embedding strategy after adding
out-of-fold training, support-aware shrinkage, multi-key IDs, hierarchical
fallback, and graph-aware neighbor fallback.

## What Changed

The neural path is no longer a single in-sample residual lookup table. The
current implementation supports:

- `oof_folds`: builds final-model embedding columns out of fold so the final
  booster sees realistic embedding noise.
- `support_prior_strength`: shrinks rare IDs toward prior vectors while letting
  frequent IDs carry stronger residual signal.
- 2D `ids`: appends one embedding block per key, such as pickup zone, dropoff
  zone, and pickup-dropoff pair.
- `fallback_ids`: tries row-level hierarchical fallback IDs before the global
  fallback vector.
- `neighbor_ids`: averages known neighbor embeddings for graph-aware fallback on
  unseen spatial IDs.

The NYC benchmark neural row now uses multi-key zone context, five OOF folds,
support-aware shrinkage, same service-zone and same-borough fallback
representatives, and adjacent-zone fallback. The synthetic model suite neural
row uses OOF embeddings and support-aware shrinkage.

## Before And After

The old neural path was optimistic on repeated-ID synthetic splits and weak on
NYC spatial holdouts. The new path is less inflated on repeated IDs and much
stronger on real spatial tasks.

| Benchmark | Split | Old neural R2 | New neural R2 | Change |
| --- | --- | ---: | ---: | ---: |
| NYC duration | random | `0.7118` | `0.7272` | `+0.0153` |
| NYC duration | spatial holdout | `0.5862` | `0.6815` | `+0.0953` |
| NYC fare | random | `0.8391` | `0.8492` | `+0.0101` |
| NYC fare | spatial holdout | `0.7929` | `0.8104` | `+0.0175` |
| NYC pickup demand | random | `0.8733` | `0.8814` | `+0.0081` |
| Synthetic neural ID | random | `0.9353` | `0.9217` | `-0.0136` |
| Synthetic neural ID | group holdout | `0.7790` | `0.7834` | `+0.0044` |

The synthetic random decline is expected. Out-of-fold embeddings remove some
in-sample residual leakage, so the random repeated-ID score is less optimistic
than before. That is a better benchmark signal.

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

Pickup-demand spatial holdout remains intentionally skipped for learned models:
the split removes all zone demand history, so learned models collapse to priors
and predicted-vs-actual plots are misleading.

## Interpretation

The new strategy fixes the main neural failure mode in the NYC benchmark. It is
now a real repeated-ID and spatial-context augmentation rather than mostly an
in-sample residual memorizer.

Use neural embeddings when:

- IDs repeat between training and inference.
- ID residuals remain after structured features and target-mean context.
- There are meaningful parent or neighbor fallback IDs.
- The task benefits from multiple keys, such as pickup zone plus dropoff zone or
  pickup zone plus hour bucket.

Do not use neural embeddings as the default when:

- The deployment target is a true cold-ID or cold-zone problem.
- There is no hierarchy, graph, or parent context for fallback.
- Plain CartoBoost already captures the signal through structured features.
- The model claim depends on spatial generalization but only random validation
  improved.

## Decision

Plain CartoBoost remains the safest default. Neural embeddings are now a strong
optional strategy for repeated-ID residual signal and zone-demand style tasks.
Graph augmentation remains the best current option for the strongest NYC fare
spatial result. LightGBM and XGBoost remain required external baselines for
claims outside CartoBoost-specific structure.

Report neural results with the split protocol. Repeated-ID gains are not
cold-start generalization.
