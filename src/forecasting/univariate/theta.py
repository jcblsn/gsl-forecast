from typing import Self

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from ..base import Forecaster


class ThetaForecaster(Forecaster):
    """Theta method (Assimakopoulos & Nikolopoulos, 2000).

    Forecast = average of:
      - theta_0: linear OLS trend extrapolated h steps
      - theta_2: SES forecast (captures level + recent movement)

    This combination is equivalent to SES + half the long-run linear slope,
    which naturally dampens the trend contribution relative to pure drift.
    """

    def __init__(self, time_col: str = "month", target_col: str = "avg_elevation"):
        super().__init__(name="theta")
        self.time_col = time_col
        self.target_col = target_col

    def fit(self, data: pd.DataFrame) -> Self:
        data = data.sort_values(self.time_col)
        vals = data[self.target_col].values.astype(float)
        n = len(vals)
        self.last_date = data[self.time_col].max()

        # OLS linear trend: vals ~ a + b*t, t=1..n
        t = np.arange(1, n + 1)
        b, a = np.polyfit(t, vals, 1)
        self._trend_intercept = a
        self._trend_slope = b
        self._n = n

        # SES fit
        ses = SimpleExpSmoothing(vals, initialization_method="estimated").fit(optimized=True)
        self._ses = ses

        self.is_fitted = True
        return self

    def predict(self, h: int, start_date=None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        origin = start_date or self.last_date
        dates = [origin + relativedelta(months=i) for i in range(1, h + 1)]

        ses_preds = self._ses.forecast(h)
        trend_preds = [
            self._trend_intercept + self._trend_slope * (self._n + i)
            for i in range(1, h + 1)
        ]
        preds = [(s + t) / 2 for s, t in zip(ses_preds, trend_preds)]

        return pd.DataFrame({
            self.time_col: dates,
            "target": self.target_col,
            "pred": preds,
            "model_name": self.name,
        })

    def get_metrics(self):
        return {"alpha": float(self._ses.params["smoothing_level"]) if self.is_fitted else None}
