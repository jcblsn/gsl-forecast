import logging
from typing import Optional, Self

import pandas as pd
from dateutil.relativedelta import relativedelta

from ..base import Forecaster


class DriftForecaster(Forecaster):
    """Project the average slope over the last `window` months forward."""

    def __init__(self, window: int = 24, time_col: str = "month", target_col: str = "avg_elevation"):
        super().__init__(name=f"drift_{window}m")
        self.window = window
        self.time_col = time_col
        self.target_col = target_col

    def fit(self, data: pd.DataFrame) -> Self:
        data = data.sort_values(self.time_col)
        if len(data) < 2:
            raise ValueError(f"DriftForecaster requires at least 2 observations, got {len(data)}")
        data = data.tail(self.window + 1)
        vals = data[self.target_col].values
        if len(vals) < self.window + 1:
            logging.warning(
                f"DriftForecaster: requested window={self.window} but only {len(vals) - 1} "
                f"intervals available; slope estimated from available data"
            )
        self.last_value = vals[-1]
        self.slope = (vals[-1] - vals[0]) / (len(vals) - 1)
        self.last_date = data[self.time_col].max()
        self.is_fitted = True
        return self

    def predict(self, h: int, start_date=None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        origin = start_date or self.last_date
        dates = [origin + relativedelta(months=i) for i in range(1, h + 1)]
        preds = [self.last_value + i * self.slope for i in range(1, h + 1)]
        return pd.DataFrame({
            self.time_col: dates,
            "target": self.target_col,
            "pred": preds,
            "model_name": self.name,
        })

    def get_metrics(self):
        return {"window": self.window}
