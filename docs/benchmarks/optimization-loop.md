# Optimization Loop

This page tracks the waterfall speed loop for the maintained NYC taxi
benchmark. Each loop follows the same order:

1. Profile the maintained benchmark path.
2. Bucket elapsed time by subsystem.
3. Change one implementation detail.
4. Run unit tests.
5. Rebuild the release native extension.
6. Re-run the profiled benchmark slice.
7. Keep the change only if speed improves without quality degradation.

## Buckets

The current fit profile buckets are emitted with `GEOBOOST_PROFILE_FIT=1`.
For the maintained `duration` task, the dominant buckets are:

| bucket | role | status |
| --- | --- | --- |
| `tree_fit` | Total tree construction time. | Parent bucket. |
| `materialize` | Row partitioning plus child histogram preparation. | Primary target. |
| `hist_score` | Histogram candidate scoring. | Secondary target. |
| `hist_accumulate` | Histogram accumulation from active rows. | Secondary target. |
| `pred_update` | Per-tree prediction update. | Low priority. |

## Loops

| loop | bucket | change | outcome |
| ---: | --- | --- | --- |
| 1 | `materialize_child_hist` | Accumulate the smaller-child histogram during partitioning instead of scanning the smaller child afterward. | Rejected. Child-hist time dropped, but partition time increased and total fit time regressed. Change was reverted. |
| 2 | `materialize_child_hist` | Skip child-histogram preparation when both children cannot legally split at the next level. | Kept as an exact-preserving guard. Neutral for the maintained preset because `min_samples_leaf=1` means the condition rarely fires on the hot path. |
| 3 | `materialize_partition` | Parallelize partitioning for large histogram nodes. | Rejected. Rayon partition overhead dominated the 25k-row workload and made fit time much worse. Change was reverted. |
| 4 | `materialize_partition` | Compare histogram bins as `u16` in the hot partition path instead of converting each bin to `usize`. | Kept. Exact-preserving; speed was mixed in the repeated benchmark, so it is not counted as a full speed-gate win. |
| 5 | `hist_score` | Hoist parent-loss computation out of the split-bin loop. | Kept. Exact-preserving; reduces repeated scoring work without changing selected splits. |
| 6 | `materialize_partition` | Read selected-feature bins from the feature-local prebinned vector during partitioning. | Kept. Exact-preserving; reduces hot partition lookup work, with the largest profiler effect in `materialize_partition`. |
| 7 | `materialize_child_hist` | Replace iterator collection in histogram subtraction with a manual push loop. | Rejected. The final profile did not improve the child-hist bucket; change was reverted. |
| 8 | `hist_score` | Use direct threshold indexing where prebinning guarantees `bin_count == thresholds.len() + 1`. | Kept. Exact-preserving; removes per-split optional threshold lookup. |
| 9 | `hist_score` | Use known-valid histogram feature entries directly in the all-features scoring path. | Kept. Exact-preserving; removes repeated option checks. |
| 10 | `materialize` | Use a direct stored histogram feature lookup when materializing a histogram candidate. | Kept. Exact-preserving; removes a redundant compatibility check on the candidate handoff. |
| 11 | `hist_accumulate` | Add a six-feature specialization to histogram row accumulation for raw row-task benchmarks. | Kept. Exact-preserving; the maintained target-mean preset uses four/eight features, but this covers the raw row-task path. |
| 12 | `hist_score` | Avoid constructing full right-child `CandidateStats` for every split in the all-features histogram scorer. | Kept. Exact-preserving; right-child stats are materialized only for the winning candidate. |
| 13 | `hist_score` | Apply the same right-child scalar scoring path to fallback fixed-width histogram scoring. | Kept. Exact-preserving; reduces per-bin temporary work outside the prebinned path. |
| 14 | `hist_score` | Apply the right-child scalar scoring path to single-feature prebinned scoring. | Kept. Exact-preserving; reduces per-bin temporary work in the generic prebinned path. |
| 15 | `materialize_partition` | Use feature-local bins in the non-all-features histogram materialization fallback. | Kept. Exact-preserving; removes row-major offset arithmetic in that fallback. |
| 16 | `materialize_partition` | Remove the now-unused row-major histogram-bin helper. | Kept. Cleanup from Loop 15; no behavior change. |
| 17 | `tree_fit` | Iterate over configured splitters by reference instead of cloning the splitter vector for every node. | Kept. Exact-preserving; removes per-node allocation on the split search path. |
| 18 | `leaf` | Reuse cached node stats for unclamped L2/log-L2 constant leaf training loss. | Kept. Exact-preserving; avoids a leaf-row pass when the leaf value is the cached weighted mean. |
| 19 | `tree_fit` | Prefer carried node stats over recomputing branch weight sums from histograms. | Kept. Exact-preserving; removes redundant histogram summing when parent materialization already carried node stats. |
| 20 | `hist_score` | Use the known first histogram feature directly for all-features common-total scoring. | Kept. Exact-preserving; removes optional helper overhead in the all-features scoring path. |

## Current Result

The refreshed repeated NYC benchmark keeps the quality utility: GeoBoost has
lower RMSE and no-worse R2 than the XGBoost histogram baseline on every
maintained task/split. Median prediction throughput is also faster on every
task/split. Training remains slower than XGBoost, so the strict all-gates result
is still a miss.

## Parameter Search Result

A 10-candidate parameter sweep was run under `target/nyc_taxi_optimization_loop/`.
Every candidate improved at least one speed metric but degraded RMSE or R2 on at
least one task/split under a strict no-degradation gate. The speed loop should
therefore continue with implementation changes rather than reducing
estimators, depth, histogram bins, or leaf constraints in the maintained preset.
