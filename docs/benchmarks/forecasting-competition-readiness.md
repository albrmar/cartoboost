# Forecasting Competition Readiness

CartoBoost has maintained forecasting evidence for real taxi demand, synthetic
taxi-shaped panels, M4 samples, M5 samples, and M6 proxy assets. That is useful
evidence, but it is not enough to claim that CartoBoost would drop into any
future forecasting competition and work without surprises.

This page defines the next benchmark layer: competition-style datasets that are
outside the current maintained M4/M5/M6 evidence. These datasets are adapter
targets, not quality claims. A dataset only becomes CartoBoost evidence after the repository
contains a fixed loader, split policy, model roster, command, artifact, metric
table, timing, and limitations.

The machine-readable catalog is committed at
`docs/assets/nyc_taxi_benchmarks/forecasting_competition_catalog.json`.

## What Would Prove Drop-In Readiness

A future competition-style run should satisfy all of these conditions:

| Requirement | Pass condition |
| --- | --- |
| Fresh task | The dataset is not one of the maintained M4/M5/M6 artifacts used during current quality work. |
| Fixed split | The split is time-ordered, documented, and generated without test leakage. |
| Fixed roster | CartoBoost auto, CartoBoost lag, seasonal or statistical baselines, and at least one serious external tabular baseline run on the same rows. |
| Native behavior | Forecasting logic remains Rust-backed; Python only loads data and orchestrates benchmarks. |
| Metric fit | The primary metric matches the competition target, such as RMSLE for store sales or sMAPE for web traffic. |
| Artifact proof | The committed artifact includes dataset hash, source files, seed, command, timing, peak memory, and full metric tables. |
| No overfit path | No dataset-specific hand routing, hyperopt, leaderboard probing, or manual test-set feedback. |

The standard to claim “competition-ready” is not that CartoBoost wins every
dataset. The standard is that `cartoboost_auto_forecast` behaves like a guarded
production model: it beats or ties `cartoboost_lag` when validation supports a
candidate, falls back when it does not, and stays competitive with strong
external baselines without manual tuning.

## Candidate Datasets

| Dataset | Domain | Why It Matters | Primary Read |
| --- | --- | --- | --- |
| Tourism forecasting competition | Tourism demand | M4-like mixed frequency behavior outside the maintained M4 artifact. | sMAPE/MASE or OWA-style proxy by frequency group. |
| NN5 cash withdrawals | ATM demand | Operational daily demand with weekly seasonality and spikes. | sMAPE, MASE, RMSE, WAPE. |
| Kaggle Store Sales | Grocery retail | Modern retail panel with stores, families, promotions, holidays, and daily sales. | RMSLE first, then RMSE/WAPE. |
| Kaggle Web Traffic | Wikipedia traffic | Large high-cardinality panel with spikes, missing values, and intermittent traffic. | sMAPE first, then scale-aware diagnostics. |
| GEFCom2012 load | Energy load | Hourly zonal demand with hierarchy and strong multi-seasonality. | RMSE/WAPE plus total-vs-zone coherence. |
| KDD Cup 2018 air quality | Environmental sensors | Hourly sensor panel with missingness and exogenous context. | RMSE/MAE/WAPE with train-only imputation. |

## Source Notes

- The Monash Time Series Forecasting Repository is the best umbrella source for
  broad forecasting stress tests because it collects many real-world and
  competition datasets in a common format.
- Kaggle Store Sales tests a different retail structure than M5: stores,
  product families, promotions, holidays, and RMSLE-style scoring.
- Kaggle Web Traffic tests scale and high-cardinality page panels rather than
  product/store hierarchy.
- GEFCom2012 load tests hourly energy demand and hierarchy; it is closer to a
  real operational forecasting deployment than a generic univariate sample.

## Adapter Order

Build adapters in this order:

1. Tourism and NN5 through the Monash archive. These are the fastest way to add
   new competition-style coverage without Kaggle authentication.
2. Store Sales. This tests modern retail covariates and RMSLE behavior outside
   M5.
3. GEFCom2012 load. This tests hourly hierarchy and multi-seasonality.
4. Web Traffic. This should be split into a fixed sample first, then a large
   scalable slice after memory behavior is known.
5. KDD Cup 2018 air quality. This should come after the missing-data policy is
   explicit.

## Reporting Template

Every new competition-style artifact should add one compact table:

| Field | Required value |
| --- | --- |
| Dataset | Public dataset name and source URL |
| Split | Exact train, validation, and holdout windows |
| Roster | CartoBoost rows plus external baselines |
| Primary metric | Metric that matches the competition scoring rule |
| Secondary metrics | RMSE, MAE, WAPE, sMAPE, MASE, or task-specific diagnostics |
| Runtime | Train time, prediction time, total wall time, and peak RSS |
| Result | Winner, CartoBoost rank, auto-vs-lag result, and limits |

Do not merge a new quality claim if `cartoboost_auto_forecast` is only better
because of a metric mismatch. For example, a WAPE value near `1.0` on signed
returns is a scale diagnostic, not a reason to prefer a model.

## Current Status

No dataset on this page is currently claimed as CartoBoost benchmark evidence.
The page is a commitment to the next proof layer: fixed competition-style
adapters and artifacts that are separate from the M4/M5/M6 work already in the
forecasting benchmark report.
