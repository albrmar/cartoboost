from __future__ import annotations

import pandas as pd

from cartoboost.forecasting import ForecastResult, PredictionInterval


def main() -> None:
    future_dates = pd.to_datetime(["2026-01-15", "2026-01-16", "2026-01-15", "2026-01-16"])
    pickup_zones = ["142", "142", "236", "236"]
    mean_forecast = [211.0, 219.0, 178.0, 185.0]
    interval = PredictionInterval(
        level=0.8,
        lower=[201.0, 208.0, 168.0, 174.0],
        upper=[221.0, 230.0, 188.0, 196.0],
    )
    result = ForecastResult.from_predictions(
        timestamps=future_dates,
        predictions=mean_forecast,
        series_id=pickup_zones,
        intervals=[interval],
        prediction_col="forecast",
        series_id_col="PULocationID",
    )

    print(result.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
