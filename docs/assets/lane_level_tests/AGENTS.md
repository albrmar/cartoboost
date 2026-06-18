# Taxi Zone Assets Agent Guide

## Dev environment tips
- This folder contains taxi-zone acceptance metric artifacts.
- Keep images, JSON metrics, and markdown summaries consistent with their generating scripts.
- The directory name is legacy; public prose should refer to taxi-zone acceptance, pickup/dropoff behavior, and trip cartometry rather than lane-level or freight-style language.

## Testing instructions
- Update generating scripts and tests before refreshing artifacts.
- Verify acceptance metrics when route cartometry or thresholds change.
- After text changes, verify no old freight/truck/carrier/lane wording leaks into public docs unless it is an intentionally generated historical artifact.

## PR instructions
- Explain artifact changes and generation commands.
- Call out threshold or route cartometry contract changes.
