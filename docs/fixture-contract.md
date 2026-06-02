# Fixture Contract

GeoBoost fixtures use small, deterministic GeoJSON files so expected behavior can
be reviewed without external services or large geospatial datasets.

## Source Fixtures

`tests/fixtures/neighborhood_points.geojson` contains three synthetic point features on a
toy coordinate plane:

- `site-alpha`
- `site-bravo`
- `site-charlie`

Each point has normalized numeric properties:

- `base_score`
- `traffic_index`
- `population_index`

`tests/fixtures/delivery_zones.geojson` contains two synthetic polygon features:

- `zone-a`
- `zone-b`

Each zone has:

- `priority`, a positive multiplier
- `service_level`, currently `same_day` or `standard`

## Golden Output

`tests/goldens/neighborhood_boosts.json` defines the canonical output for a
weighted overlay scenario. The expected scoring formula is:

```text
boost_score = zone.priority * (
  base_score * 0.5 +
  traffic_index * 0.3 +
  population_index * 0.2
)
```

Scores are rounded to six decimal places using half-even rounding. Results are
ranked by descending `boost_score`, with `rank` starting at `1`.

## Fixture Rules

- Keep fixtures small enough to inspect in code review.
- Use stable feature ids; golden files refer to ids, not array positions.
- Keep coordinates in `[x, y]` order on a synthetic toy plane.
- Do not use real-world coordinates, downloaded datasets, customer data, or
  copied external datasets in committed fixtures.
