# Taxi Zone Acceptance Assets

This directory contains generated evidence for the taxi-zone acceptance
benchmark. The maintained narrative is
[Taxi Zone Acceptance](../../benchmarks/taxi-zone.md).

The fixture isolates pickup/dropoff lane membership, route midpoint geometry,
hour-of-day wraparound, and combined lane/spatial/temporal structure. Use the
plots and metrics to inspect whether those feature families behave as expected.
Refresh them only from
`scripts/run_lane_level_acceptance_metrics.py`.
