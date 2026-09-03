"""Inflow-driven elevation recursion: snowpack -> inflow -> elevation change.

Stage one predicts each future month's tributary inflow (kaf) from the snowpack known at
the cutoff, by calendar month and lead. Stage two is an empirical monthly recursion:
the change in elevation from one month to the next as a function of that month's inflow and
either the starting elevation or, with `level_term="area"`, lake area from the USGS
hypsometry. It does not conserve storage or close all inflows and outflows. The elevation is
rolled forward one month at a time.
"""

from datetime import date
from typing import Self

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from .. import hypsometry
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

INFLOW_COL = "inflow_kaf_total"
DEFAULT_SNOW = ["swe_eom_gsl", "prec_wy_eom_gsl"]


class InflowChainForecaster(Forecaster):
    def __init__(
        self,
        snow_features: list[str] | None = None,
        min_obs: int = MIN_OBS,
        alpha: float | None = None,
        level_term: str = "level",
        name: str = "inflow_chain",
    ):
        super().__init__(name=name)
        self.snow_features = list(snow_features or DEFAULT_SNOW)
        self.min_obs = min_obs
        self.alpha = alpha
        self.level_term = level_term
        self._data: pd.DataFrame | None = None
        self._step: dict[int, np.ndarray] = {}

    def feature_columns(self) -> list[str]:
        return [INFLOW_COL, *self.snow_features]

    def fit(self, data: pd.DataFrame) -> Self:
        require_columns(data, [TIME_COL, TARGET_COL, INFLOW_COL, *self.snow_features])
        df = data.sort_values(TIME_COL).reset_index(drop=True)
        self._data = df
        self.last_date = df[TIME_COL].iloc[-1]
        y = df[TARGET_COL].to_numpy(dtype=float)
        months = df[TIME_COL].dt.month.to_numpy()
        consecutive = np.array(
            [
                df[TIME_COL].iloc[i + 1] == df[TIME_COL].iloc[i] + relativedelta(months=1)
                for i in range(len(df) - 1)
            ]
        )
        inflow_next = df[INFLOW_COL].to_numpy(dtype=float)[1:]
        state = self._state(y[:-1])
        dy = y[1:] - y[:-1]
        self._step = {}
        for m in range(1, 13):
            sel = consecutive & (months[1:] == m) & ~np.isnan(inflow_next)
            if sel.sum() >= self.min_obs:
                X = np.column_stack([np.ones(sel.sum()), inflow_next[sel], state[sel]])
                self._step[m] = ridge_fit(X, dy[sel], self.alpha)
            else:
                fallback = consecutive & (months[1:] == m)
                self._step[m] = np.array([dy[fallback].mean() if fallback.any() else 0.0, 0, 0])
        self._mean_inflow = df.groupby(df[TIME_COL].dt.month)[INFLOW_COL].mean().to_dict()
        self.is_fitted = True
        return self

    def _state(self, level):
        return hypsometry.area_km2(level) if self.level_term == "area" else level

    def inflow_forecast(self, h: int) -> float:
        """Stage one: the tributary inflow (kaf) predicted for lead h from the snowpack now."""
        return self._inflow(h)

    def _inflow(self, h: int) -> float:
        df = self._data
        last = df.iloc[-1]
        month = last[TIME_COL].month
        target_month = (month + h - 1) % 12 + 1
        idx = [
            i
            for i in range(len(df) - h)
            if df[TIME_COL].iloc[i].month == month
            and df[TIME_COL].iloc[i + h] == df[TIME_COL].iloc[i] + relativedelta(months=h)
        ]
        rows = df.iloc[idx]
        target = df[INFLOW_COL].to_numpy(dtype=float)[[i + h for i in idx]]
        features, dropped = select_features(rows, self.snow_features, last, self.min_obs, df)
        ok = rows[features].notna().all(axis=1).to_numpy() & ~np.isnan(target)
        log_fallback(self.name, h, dropped)
        if not features or int(ok.sum()) < self.min_obs:
            return float(self._mean_inflow.get(target_month, np.nan))
        beta = ridge_fit(design(rows[ok], features), target[ok], self.alpha)
        x = np.concatenate([[1.0], last[features].to_numpy(dtype=float)])
        return max(float(x @ beta), 0.0)

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        origin = start_date or self.last_date
        level = float(self._data[TARGET_COL].iloc[-1])
        preds = []
        for i in range(1, h + 1):
            month = (origin + relativedelta(months=i)).month
            inflow = self._inflow(i)
            b = self._step[month]
            if np.isnan(inflow):
                level += b[0]
            else:
                level += float(b[0] + b[1] * inflow + b[2] * self._state(level))
            preds.append(level)
        return pd.DataFrame(
            {
                TIME_COL: [origin + relativedelta(months=i) for i in range(1, h + 1)],
                "target": TARGET_COL,
                "pred": preds,
                "model_name": self.name,
            }
        )

    def get_metrics(self) -> dict[str, object]:
        return {
            "snow_features": ",".join(self.snow_features),
            "min_obs": self.min_obs,
            "alpha": self.alpha if self.alpha is not None else "gcv",
            "level_term": self.level_term,
        }
