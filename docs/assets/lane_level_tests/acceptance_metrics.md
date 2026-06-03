# GeoBoost lane-level acceptance metrics

These deterministic fixtures adapt the upstream regional lane CPM idea to this repo's
current API. The matrix columns are observable route features: origin/destination
coordinates, lane ID, hour of day, route midpoint, and route distance.

| phase | model | metric | value |
| --- | --- | --- | ---: |
| sparse_lane_membership | axis_lane_id | train_rmse | 5.19615242e+01 |
| sparse_lane_membership | sparse_lane_id | train_rmse | 0.00000000e+00 |
| route_midpoint_geometry | axis_midpoint | train_rmse | 6.94022094e+01 |
| route_midpoint_geometry | gaussian_midpoint | train_rmse | 0.00000000e+00 |
| wraparound_lane_hour | axis_hour | train_rmse | 3.15772547e+01 |
| wraparound_lane_hour | periodic_hour | train_rmse | 1.98951966e-13 |
| regional_lane_boosting | axis_only | holdout_rmse | 1.26819804e+01 |
| regional_lane_boosting | lane_spatial_temporal | holdout_rmse | 4.85635574e+00 |

## Inspection Metrics

### sparse_lane_membership
- `hot_lane_prediction`: 3.20000000e+02
- `cold_neighbor_lane_prediction`: 8.00000000e+01
- `hot_lane_margin`: 2.40000000e+02
- `hot_lane_id`: 7.00000000e+00
- `lane_count`: 1.60000000e+01
- `sparse_lane_exact`: PASS (0.00000000e+00 < 1.00000000e-12)
- `sparse_beats_axis_lane_id`: PASS (0.00000000e+00 < 5.00000000e-02)
- `hot_lane_margin_gt_200`: PASS (2.40000000e+02 > 2.00000000e+02)

### route_midpoint_geometry
- `center_lane_prediction`: 2.60000000e+02
- `outer_lane_prediction`: 9.00000000e+01
- `center_outer_margin`: 1.70000000e+02
- `axis_to_gaussian_rmse_ratio`: 0.00000000e+00
- `gaussian_route_rmse_lt_axis_half`: PASS (0.00000000e+00 < 5.00000000e-01)
- `center_outer_margin_gt_100`: PASS (1.70000000e+02 > 1.00000000e+02)

### wraparound_lane_hour
- `periodic_23_vs_1_gap`: 0.00000000e+00
- `axis_23_vs_1_gap`: 1.04047619e+02
- `periodic_peak_to_midday_margin`: 1.15000000e+02
- `periodic_hour_exact`: PASS (1.98951966e-13 < 1.00000000e-12)
- `periodic_edge_gap_lt_1e_12`: PASS (0.00000000e+00 < 1.00000000e-12)
- `axis_edge_gap_gt_50`: PASS (1.04047619e+02 > 5.00000000e+01)

### regional_lane_boosting
- `holdout_rmse_ratio`: 3.82933548e-01
- `hot_lane_midnight_prediction`: 2.93406476e+02
- `cold_lane_midday_prediction`: 1.43021033e+02
- `hot_cold_operating_contrast`: 1.50385443e+02
- `uses_hidden_simulator_metadata_in_training`: 0.00000000e+00
- `full_beats_axis_holdout`: PASS (3.82933548e-01 < 6.50000000e-01)
- `hot_cold_contrast_gt_120`: PASS (1.50385443e+02 > 1.20000000e+02)

