"""Direct multi-horizon regression of the lake's change on month-end snowpack.

This is the NRCS outlook generalised to every month and every lead: for a cutoff in calendar
month m and lead h, the change in elevation over h months is regressed on the current
elevation and the current basin snow water equivalent and water-year precipitation, using
every past year's cutoff in the same calendar month. Only values known at the cutoff enter.
"""

from datetime import date
from typing import Self

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ..base import Forecaster
from .regression import TARGET_COL, TIME_COL, design, require_columns, ridge_fit

DEFAULT_FEATURES = ["swe_eom_gsl", "prec_wy_eom_gsl"]


class SweRegressionForecaster(Forecaster):
    def __init__(
        self,
        features: list[str] | None = None,
        min_obs: int = 10,
        alpha: float = 1e-3,
        name: str = "swe_regression",
    ):
        super().__init__(name=name)
        self.features = list(features or DEFAULT_FEATURES)
        self.min_obs = min_obs
        self.alpha = alpha
        self._data: pd.DataFrame | None = None

    def fit(self, data: pd.DataFrame) -> Self:
        require_columns(data, [TIME_COL, TARGET_COL, *self.features])
        self._data = data.sort_values(TIME_COL).reset_index(drop=True)
        self.last_date = self._data[TIME_COL].iloc[-1]
        self.is_fitted = True
        return self

    def _delta(self, h: int) -> float:
        df = self._data
        n = len(df)
        last = df.iloc[-1]
        month = last[TIME_COL].month
        idx = np.array(
            [
                i
                for i in range(n - h)
                if df[TIME_COL].iloc[i].month == month
                and df[TIME_COL].iloc[i + h] == df[TIME_COL].iloc[i] + relativedelta(months=h)
            ]
        )
        if len(idx) == 0:
            return 0.0
        y = df[TARGET_COL].to_numpy()
        dy = y[idx + h] - y[idx]
        rows = df.iloc[idx]
        ok = rows[self.features].notna().all(axis=1).to_numpy()
        have_now = bool(last[self.features].notna().all())
        if ok.sum() >= self.min_obs and have_now:
            cols = [TARGET_COL, *self.features]
            beta = ridge_fit(design(rows[ok], cols), dy[ok], self.alpha)
            x = np.concatenate([[1.0], last[cols].to_numpy(dtype=float)])
            return float(x @ beta)
        beta = ridge_fit(design(rows, [TARGET_COL]), dy, self.alpha)
        return float(np.array([1.0, last[TARGET_COL]]) @ beta)

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        origin = start_date or self.last_date
        y0 = float(self._data[TARGET_COL].iloc[-1])
        return pd.DataFrame(
            {
                TIME_COL: [origin + relativedelta(months=i) for i in range(1, h + 1)],
                "target": TARGET_COL,
                "pred": [y0 + self._delta(i) for i in range(1, h + 1)],
                "model_name": self.name,
            }
        )

    def get_metrics(self) -> dict[str, object]:
        return {"features": ",".join(self.features), "min_obs": self.min_obs, "alpha": self.alpha}
