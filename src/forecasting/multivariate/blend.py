"""The official blended model: a per-lead convex mix of snowpack and univariate skill.

Snowpack decides the next 6 to 12 months, and it says nothing about the winter after next.
So the better model changes with the lead. This model puts weight w(h) on `swe_regression`
and 1 - w(h) on `ets_damped_s12`. It fits w(h) by a walk-forward pass inside the training
data, then forces the weights to fall with the lead, so the blend hands over to the
univariate model as the snowpack signal expires.

The inner pass uses the same 15-year cutoff window as the outer harness. A wider window
gives less noisy weights but a biased answer: before about 1995 the SNOTEL record was too
short to fit the snowpack model on, so the inner pass sees it fail for a reason that no
longer applies and gives it half the weight it earns. On 30 years the fitted weight at lead
6 is 0.60, against an outer error of 0.51 ft for the snowpack model and 0.82 ft for the
univariate one.
"""

from datetime import date
from typing import Self

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ..base import Forecaster
from ..univariate.exponential_smoothing import HoltWintersForecaster
from .regression import TARGET_COL, TIME_COL
from .swe_regression import SweRegressionForecaster

WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)
FULL_WEIGHT_LEAD = 6
ZERO_WEIGHT_LEAD = 24

_CACHE: dict[tuple, np.ndarray] = {}
_CACHE_LIMIT = 4000


def component_factories() -> list:
    """The two models the blend mixes. Both are also registered in their own right."""
    return [
        SweRegressionForecaster,
        lambda: HoltWintersForecaster(
            trend="add", seasonal="add", seasonal_periods=12, damped_trend=True
        ),
    ]


def default_weights(horizon: int) -> np.ndarray:
    """The ramp used when the training frame holds too few walk-forward cutoffs to fit on."""
    leads = np.arange(1, horizon + 1)
    span = ZERO_WEIGHT_LEAD - FULL_WEIGHT_LEAD
    return np.clip((ZERO_WEIGHT_LEAD - leads) / span, 0.0, 1.0)


def monotone_decreasing(weights: np.ndarray) -> np.ndarray:
    """Pool adjacent violators so the weight never rises with the lead."""
    stack: list[tuple[float, int]] = []
    for value in weights:
        current, count = float(value), 1
        while stack and stack[-1][0] < current:
            previous, n = stack.pop()
            current = (previous * n + current * count) / (n + count)
            count += n
        stack.append((current, count))
    out: list[float] = []
    for value, count in stack:
        out.extend([value] * count)
    return np.array(out)


def _fingerprint(train: pd.DataFrame) -> tuple:
    y = train[TARGET_COL].to_numpy(dtype=float)
    return (len(y), round(float(y[0]), 6), round(float(y[-1]), 6), round(float(np.nansum(y)), 4))


def _cached_prediction(index: int, factory, train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Walk-forward predictions of one component, memoised on the training slice.

    An outer cross-validation calls `fit` once per cutoff, and consecutive cutoffs share
    almost every inner cutoff. The prediction at an inner cutoff depends only on rows at or
    before it, so the memo is safe and it removes most of the cost.
    """
    key = (index, train[TIME_COL].iloc[-1], horizon, _fingerprint(train))
    hit = _CACHE.get(key)
    if hit is None:
        hit = factory().fit(train).predict(horizon)["pred"].to_numpy(dtype=float)
        if len(_CACHE) > _CACHE_LIMIT:
            _CACHE.clear()
        _CACHE[key] = hit
    return hit


class BlendForecaster(Forecaster):
    def __init__(
        self,
        horizon: int = 24,
        history_years: int = 15,
        max_cutoffs: int | None = None,
        min_cutoffs: int = 24,
        name: str = "blend",
    ):
        super().__init__(name=name)
        self.horizon = horizon
        self.history_years = history_years
        self.max_cutoffs = max_cutoffs
        self.min_cutoffs = min_cutoffs
        self.weights = default_weights(horizon)
        self.n_weight_cutoffs = 0
        self._fitted: list[Forecaster] = []

    def _walk_forward(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        from src.forecasting.cross_validate import valid_cutoffs

        cutoffs = valid_cutoffs(data, self.history_years, self.horizon)
        if self.max_cutoffs is not None:
            cutoffs = cutoffs[-self.max_cutoffs :]
        if len(cutoffs) < self.min_cutoffs:
            self.n_weight_cutoffs = len(cutoffs)
            return None
        observed = data.set_index(TIME_COL)[TARGET_COL]
        actuals, predictions = [], []
        for cutoff in cutoffs:
            train = data[data[TIME_COL] <= cutoff]
            months = [cutoff + relativedelta(months=i) for i in range(1, self.horizon + 1)]
            try:
                rows = [
                    _cached_prediction(i, factory, train, self.horizon)
                    for i, factory in enumerate(component_factories())
                ]
            except Exception:
                continue
            actuals.append(observed.reindex(months).to_numpy(dtype=float))
            predictions.append(np.stack(rows))
        self.n_weight_cutoffs = len(actuals)
        if len(actuals) < self.min_cutoffs:
            return None
        return np.array(actuals), np.stack(predictions)

    def _fit_weights(self, data: pd.DataFrame) -> np.ndarray:
        result = self._walk_forward(data)
        default = default_weights(self.horizon)
        if result is None:
            return default
        actual, pred = result
        snow, univariate = pred[:, 0, :], pred[:, 1, :]
        weights = np.full(self.horizon, np.nan)
        for lead in range(self.horizon):
            usable = (
                np.isfinite(actual[:, lead])
                & np.isfinite(snow[:, lead])
                & np.isfinite(univariate[:, lead])
            )
            if usable.sum() < self.min_cutoffs:
                continue
            a, s, u = actual[usable, lead], snow[usable, lead], univariate[usable, lead]
            mae = [np.abs(w * s + (1 - w) * u - a).mean() for w in WEIGHT_GRID]
            weights[lead] = float(WEIGHT_GRID[int(np.argmin(mae))])
        return monotone_decreasing(np.where(np.isfinite(weights), weights, default))

    def fit(self, data: pd.DataFrame) -> Self:
        df = data.sort_values(TIME_COL).reset_index(drop=True)
        self.weights = self._fit_weights(df)
        self._fitted = [factory().fit(df) for factory in component_factories()]
        self.last_date = df[TIME_COL].iloc[-1]
        self.is_fitted = True
        return self

    def _weights_for(self, h: int) -> np.ndarray:
        if h <= len(self.weights):
            return self.weights[:h]
        pad = np.full(h - len(self.weights), self.weights[-1])
        return np.concatenate([self.weights, pad])

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        origin = start_date or self.last_date
        snow, univariate = (f.predict(h, start_date) for f in self._fitted)
        w = self._weights_for(h)
        return pd.DataFrame(
            {
                TIME_COL: [origin + relativedelta(months=i) for i in range(1, h + 1)],
                "target": TARGET_COL,
                "pred": w * snow["pred"].to_numpy(dtype=float)
                + (1 - w) * univariate["pred"].to_numpy(dtype=float),
                "model_name": self.name,
            }
        )

    def get_metrics(self) -> dict[str, object]:
        w = self.weights
        return {
            "components": "swe_regression,ets_damped_s12",
            "n_weight_cutoffs": self.n_weight_cutoffs,
            "weight_h1": float(w[0]),
            "weight_h6": float(w[min(5, len(w) - 1)]),
            "weight_h12": float(w[min(11, len(w) - 1)]),
            "weight_h24": float(w[-1]),
        }
