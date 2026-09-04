"""Direct multi-horizon regression of south-arm elevation change.

For each cutoff month and lead, the fit uses past rows from the same calendar month. Default
predictors are current elevation, month-end snow water equivalent, and water-year
precipitation. Only values available at the cutoff enter the fit.
"""

from datetime import date
from typing import Self

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ..base import Forecaster
from .regression import (
    MIN_OBS,
    TARGET_COL,
    TIME_COL,
    design,
    log_fallback,
    require_columns,
    ridge_fit,
    select_features,
)

DEFAULT_FEATURES = ["swe_eom_gsl", "prec_wy_eom_gsl"]


class SweRegressionForecaster(Forecaster):
    def __init__(
        self,
        features: list[str] | None = None,
        min_obs: int = MIN_OBS,
        alpha: float | None = None,
        name: str = "swe_regression",
    ):
        super().__init__(name=name)
        self.features = list(features or DEFAULT_FEATURES)
        self.min_obs = min_obs
        self.alpha = alpha
        self._data: pd.DataFrame | None = None

    def feature_columns(self) -> list[str]:
        return list(self.features)

    def fit(self, data: pd.DataFrame) -> Self:
        require_columns(data, [TIME_COL, TARGET_COL, *self.features])
        self._data = data.sort_values(TIME_COL).reset_index(drop=True)
        self.last_date = self._data[TIME_COL].iloc[-1]
        self.is_fitted = True
        return self

    def _horizon_fit(self, h: int) -> dict[str, object]:
        """The fit for one lead: the coefficients, the columns, and the training rows."""
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
            return {"beta": np.array([0.0, 0.0]), "columns": [TARGET_COL], "rows": df.iloc[[-1]]}
        y = df[TARGET_COL].to_numpy()
        dy = y[idx + h] - y[idx]
        rows = df.iloc[idx]
        features, dropped = select_features(rows, self.features, last, self.min_obs, df)
        ok = rows[features].notna().all(axis=1).to_numpy()
        if features and int(ok.sum()) < self.min_obs:
            dropped |= dict.fromkeys(features, "too few rows carry all the kept features")
            features, ok = [], np.ones(len(rows), dtype=bool)
        log_fallback(self.name, h, dropped)
        cols = [TARGET_COL, *features]
        beta = ridge_fit(design(rows[ok], cols), dy[ok], self.alpha)
        return {"beta": beta, "columns": cols, "rows": rows[ok]}

    def _delta(self, h: int) -> float:
        fitted = self._horizon_fit(h)
        columns = fitted["columns"]
        x = np.concatenate([[1.0], self._data.iloc[-1][columns].to_numpy(dtype=float)])
        return float(x @ fitted["beta"])

    def contributions(self, h: int) -> pd.DataFrame:
        """Decompose the point forecast around the training-feature means.

        Each term is a coefficient times the input's distance from its mean. The terms sum
        to the fitted prediction but are not causal effects.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before explanation")
        last = self._data.iloc[-1]
        output = []
        for lead in range(1, h + 1):
            fitted = self._horizon_fit(lead)
            beta, columns, rows = fitted["beta"], fitted["columns"], fitted["rows"]
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
            output.extend({"month": month, "h": lead, **term} for term in terms)
        return pd.DataFrame(output)

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
        return {
            "features": ",".join(self.features),
            "min_obs": self.min_obs,
            "alpha": self.alpha if self.alpha is not None else "gcv",
        }
