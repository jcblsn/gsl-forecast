from datetime import date
from typing import Literal, Self

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ..base import Forecaster


class MovingAverageForecaster(Forecaster):
    def __init__(
        self,
        time_col: str = "month",
        target_col: str = "avg_elevation",
        window: int = 3,
        method: Literal["simple", "weighted"] = "simple",
    ):
        super().__init__(name=f"ma_{method}_{window}")
        self.time_col = time_col
        self.target_col = target_col
        self.window = window
        self.method = method
        self.last_date = None
        self.ma_value = None

    def fit(self, data: pd.DataFrame) -> Self:
        if self.time_col not in data.columns or self.target_col not in data.columns:
            raise ValueError(f"Data must contain '{self.time_col}' and '{self.target_col}' columns")

        if len(data) < self.window:
            raise ValueError(f"Data must contain at least {self.window} observations")

        data = data.sort_values(self.time_col)
        self.last_date = data[self.time_col].max()

        values = data[self.target_col].values

        if self.method == "simple":
            self.ma_value = np.mean(values[-self.window :])
        elif self.method == "weighted":
            # linear weights with more weight on recent observations
            weights = np.linspace(1, 2, self.window)
            weights = weights / np.sum(weights)  # normalized
            self.ma_value = np.sum(values[-self.window :] * weights)

        self.is_fitted = True
        return self

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        origin = start_date or self.last_date
        dates = [origin + relativedelta(months=i) for i in range(1, h + 1)]

        # repeat the MA value for all future points
        predictions = [self.ma_value] * h

        return pd.DataFrame(
            {
                self.time_col: dates,
                "target": self.target_col,
                "pred": predictions,
                "model_name": self.name,
            }
        )

    def get_metrics(self) -> dict[str, float]:
        return {"method": self.method, "window": self.window}
