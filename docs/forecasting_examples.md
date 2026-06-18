# Forecasting Examples

The forecasting examples are deterministic and use the committed
`examples/forecasting/forecast_cli_input.csv` taxi pickup-demand panel. They do
not download data or call private services.

Run the single-series style theta workflow:

```bash
uv run --group dev python examples/forecasting/single_series_theta.py
```

Run a panel forecast across pickup zones:

```bash
uv run --group dev python examples/forecasting/panel_forecasting.py
```

Run a final-window backtest:

```bash
uv run --group dev python examples/forecasting/rolling_origin_backtest.py
```

Print point forecasts with 80 percent residual intervals:

```bash
uv run --group dev python examples/forecasting/probabilistic_intervals.py
```

Generated artifacts in these examples are written to temporary directories. For
persisted local runs, use paths under `target/forecasting/`.
