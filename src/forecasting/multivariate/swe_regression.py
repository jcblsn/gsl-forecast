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

    def _horizon_fit(self, h: int) -> dict[str, object]:
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
            rows = df.iloc[[-1]]
            return {"beta": np.array([0.0, 0.0]), "columns": [TARGET_COL], "rows": rows}
        y = df[TARGET_COL].to_numpy()
        dy = y[idx + h] - y[idx]
        rows = df.iloc[idx]
        ok = rows[self.features].notna().all(axis=1).to_numpy()
        have_now = bool(last[self.features].notna().all())
        if ok.sum() >= self.min_obs and have_now:
            cols = [TARGET_COL, *self.features]
            beta = ridge_fit(design(rows[ok], cols), dy[ok], self.alpha)
            return {"beta": beta, "columns": cols, "rows": rows[ok]}
        beta = ridge_fit(design(rows, [TARGET_COL]), dy, self.alpha)
        return {"beta": beta, "columns": [TARGET_COL], "rows": rows}

    def _delta(self, h: int) -> float:
        fitted = self._horizon_fit(h)
        columns = fitted["columns"]
        x = np.concatenate([[1.0], self._data.iloc[-1][columns].to_numpy(dtype=float)])
        return float(x @ fitted["beta"])

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

    def contributions(self, h: int) -> pd.DataFrame:
        """Return centered additive terms for each forecast date."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before explanation")
        last = self._data.iloc[-1]
        output = []
        for lead in range(1, h + 1):
            fitted = self._horizon_fit(lead)
            beta = fitted["beta"]
            columns = fitted["columns"]
            rows = fitted["rows"]
            means = rows[columns].mean()
            reference = float(means[TARGET_COL] + beta[0] + means.to_numpy() @ beta[1:])
            terms = [
                {
                    "input": "reference_path",
                    "value": None,
                    "reference": None,
                    "contribution_ft": reference,
                }
            ]
            for i, column in enumerate(columns, start=1):
                coefficient = float(beta[i]) + (1.0 if column == TARGET_COL else 0.0)
                terms.append(
                    {
                        "input": column,
                        "value": float(last[column]),
                        "reference": float(means[column]),
                        "contribution_ft": coefficient * float(last[column] - means[column]),
                    }
                )
            month = self.last_date + relativedelta(months=lead)
            for term in terms:
                output.append({"month": month, "h": lead, **term})
        return pd.DataFrame(output)
