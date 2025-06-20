from datetime import date
from typing import Dict, Optional, Self

import pandas as pd
from dateutil.relativedelta import relativedelta

from ..base import Forecaster


class NaiveForecaster(Forecaster):
    def __init__(
        self,
        time_col: str = "month",
        target_col: str = "avg_elevation",
        method: str = "last",
        seasonal_period: int = 12,
    ):
        super().__init__(name=f"naive_{method}")
        self.time_col = time_col
        self.target_col = target_col
        self.method = method
        self.seasonal_period = seasonal_period
        self.last_obs = None

    def fit(self, data: pd.DataFrame) -> Self:
        if self.time_col not in data.columns or self.target_col not in data.columns:
            raise ValueError(
                f"Data must contain '{self.time_col}' and '{self.target_col}' columns"
            )

        data = data.sort_values(self.time_col)
        self.last_date = data[self.time_col].max()

        if self.method == "last":
            self.last_obs = data[self.target_col].iloc[-1]
        elif self.method == "seasonal":
            values = data[self.target_col].values
            if len(values) >= self.seasonal_period:
                self.last_obs = values[-self.seasonal_period :]
            else:
                raise ValueError(
                    f"Not enough data for seasonal forecasting with period {self.seasonal_period}"
                )

        self.is_fitted = True
        return self

    def predict(self, h: int, start_date: Optional[date] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        origin = start_date or self.last_date
        dates = [origin + relativedelta(months=i) for i in range(1, h + 1)]

        if self.method == "last":
            predictions = [self.last_obs] * h
        else:  # seasonal method
            predictions = [
                self.last_obs[(i - 1) % self.seasonal_period] for i in range(1, h + 1)
            ]

        return pd.DataFrame(
            {
                self.time_col: dates,
                "target": self.target_col,
                "pred": predictions,
                "model_name": self.name,
            }
        )

    def get_metrics(self) -> Dict[str, float]:
        return {"method": self.method, "seasonal_period": self.seasonal_period}
