from datetime import date
from typing import Self

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from ..base import Forecaster


class ThetaForecaster(Forecaster):
    """Standard theta method (Assimakopoulos & Nikolopoulos 2000; Hyndman & Billah 2003).

    Equivalent to SES with drift equal to half the OLS trend slope:
        y_hat(h) = SES(h) + (b / 2) * (h - 1 + 1/alpha)
    The SES component anchors the forecast at the current level; the drift term
    carries half the long-run slope. Averaging SES with the raw OLS line (a common
    mis-reading of the method) would instead anchor the forecast at the regression
    line, which can sit far from the current level on a non-stationary series.
    """

    def __init__(self, time_col: str = "month", target_col: str = "avg_elevation"):
        super().__init__(name="theta")
        self.time_col = time_col
        self.target_col = target_col

    def fit(self, data: pd.DataFrame) -> Self:
        data = data.sort_values(self.time_col)
        vals = data[self.target_col].values.astype(float)
        self.last_date = data[self.time_col].max()

        t = np.arange(1, len(vals) + 1)
        self._trend_slope, _ = np.polyfit(t, vals, 1)

        self._ses = SimpleExpSmoothing(vals, initialization_method="estimated").fit(optimized=True)
        self._alpha = float(self._ses.params["smoothing_level"])

        self.is_fitted = True
        return self

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        origin = start_date or self.last_date
        dates = [origin + relativedelta(months=i) for i in range(1, h + 1)]

        ses_preds = np.asarray(self._ses.forecast(h))
        steps = np.arange(1, h + 1)
        alpha = max(self._alpha, 1e-6)
        drift = (self._trend_slope / 2.0) * (steps - 1 + 1.0 / alpha)
        preds = ses_preds + drift

        return pd.DataFrame(
            {
                self.time_col: dates,
                "target": self.target_col,
                "pred": preds.tolist(),
                "model_name": self.name,
            }
        )

    def get_metrics(self):
        if not self.is_fitted:
            return {"alpha": None, "slope": None}
        return {"alpha": self._alpha, "slope": float(self._trend_slope)}
