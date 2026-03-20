from typing import Optional, Self

import pandas as pd
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ..base import Forecaster


class HoltWintersForecaster(Forecaster):
    """Holt-Winters exponential smoothing via statsmodels.

    trend: 'add' | None
    seasonal: 'add' | None
    damped_trend: bool
    """

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: int = 12,
        damped_trend: bool = False,
        time_col: str = "month",
        target_col: str = "avg_elevation",
    ):
        parts = []
        if trend:
            parts.append("damped" if damped_trend else trend)
        parts.append(f"s{seasonal_periods}" if seasonal else "noseas")
        super().__init__(name=f"ets_{'_'.join(parts)}")

        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.time_col = time_col
        self.target_col = target_col
        self._result = None

    def fit(self, data: pd.DataFrame) -> Self:
        data = data.sort_values(self.time_col)
        self.last_date = data[self.time_col].max()
        series = data[self.target_col].values.astype(float)

        model = ExponentialSmoothing(
            series,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods if self.seasonal else None,
            damped_trend=self.damped_trend,
            initialization_method="estimated",
        )
        self._result = model.fit(optimized=True, remove_bias=True)
        self.is_fitted = True
        return self

    def predict(self, h: int, start_date=None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        origin = start_date or self.last_date
        dates = [origin + relativedelta(months=i) for i in range(1, h + 1)]
        preds = self._result.forecast(h).tolist()
        return pd.DataFrame({
            self.time_col: dates,
            "target": self.target_col,
            "pred": preds,
            "model_name": self.name,
        })

    def get_metrics(self):
        return {
            "trend": self.trend or "none",
            "seasonal": self.seasonal or "none",
            "damped_trend": self.damped_trend,
            "seasonal_periods": self.seasonal_periods,
        }
