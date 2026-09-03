"""The official model. It mixes 2 component models with a fitted weight.

Snowpack controls the lake level for the next 6 to 12 months. It gives no information about
the winter after next. Therefore the best model changes with the forecast lead.

This model applies the weight w to the snowpack model, and 1 - w to `ets_damped_s12`. It
fits w for each lead and for each issue season.

The issue season is necessary. A lead of 6 months from a February issue gives the spring
peak, which the current snowpack controls. The same lead from an August issue gives a month
in the next accumulation season, which the current snowpack does not control.

The model fits the weight curves with a walk-forward pass in the training data. Each curve
must decrease with the lead. The search finds the curve with the lowest total absolute error
under that condition. It does not fit each lead separately and then correct the result.

The inner pass uses the same 15-year window as the harness. A longer window gives less noise
in the weights, but it gives a biased result. Before approximately 1995 the SNOTEL record is
too short to fit the snowpack model. The inner pass then records a failure that cannot occur
now, and gives the snowpack model approximately half of the correct weight.
"""

from datetime import date
from typing import Self

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ..base import Forecaster
from ..cutoffs import valid_cutoffs
from ..univariate.exponential_smoothing import HoltWintersForecaster
from .regression import TARGET_COL, TIME_COL
from .swe_regression import SweRegressionForecaster

SNOW_FEATURES = ["swe_eom_gsl", "prec_wy_eom_gsl", "head_diff_ft"]
SNOW_NAME = "swe_head"
UNIVARIATE_NAME = "ets_damped_s12"
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
FULL_WEIGHT_LEAD = 6
ZERO_WEIGHT_LEAD = 24

SEASON_MONTHS = {
    "accumulation": {11, 12, 1, 2, 3},
    "melt": {4, 5, 6},
    "recession": {7, 8, 9, 10},
}

_CACHE: dict[tuple, np.ndarray] = {}
_CACHE_LIMIT = 4000


def issue_season(cutoff: pd.Timestamp) -> str:
    """The water-year stage of the issue that follows a cutoff."""
    issue_month = pd.Timestamp(cutoff).month % 12 + 1
    return next(name for name, months in SEASON_MONTHS.items() if issue_month in months)


def default_weights(horizon: int) -> np.ndarray:
    """The fixed ramp. The model uses it when the training data gives too few cutoffs."""
    leads = np.arange(1, horizon + 1)
    span = ZERO_WEIGHT_LEAD - FULL_WEIGHT_LEAD
    return np.clip((ZERO_WEIGHT_LEAD - leads) / span, 0.0, 1.0)


def monotone_weight_path(loss: np.ndarray) -> np.ndarray:
    """The weight path with the lowest total loss that does not increase with the lead.

    `loss[i, j]` is the total absolute error at lead i + 1 for the weight `WEIGHT_GRID[j]`. A
    dynamic program finds the minimum over all leads under the condition. The condition is
    thus part of the objective, and not a correction to a free fit. For equal loss, the
    program selects the smaller weight on the snowpack model.
    """
    n_leads, n_weights = loss.shape
    cost = np.full((n_leads, n_weights), np.inf)
    previous = np.zeros((n_leads, n_weights), dtype=int)
    cost[0] = loss[0]
    for i in range(1, n_leads):
        for current in range(n_weights):
            prior = current + int(np.argmin(cost[i - 1, current:]))
            cost[i, current] = loss[i, current] + cost[i - 1, prior]
            previous[i, current] = prior
    selected = np.zeros(n_leads, dtype=int)
    selected[-1] = int(np.argmin(cost[-1]))
    for i in range(n_leads - 1, 0, -1):
        selected[i - 1] = previous[i, selected[i]]
    return WEIGHT_GRID[selected]


def _fingerprint(train: pd.DataFrame) -> tuple:
    y = train[TARGET_COL].to_numpy(dtype=float)
    return (len(y), round(float(y[0]), 6), round(float(y[-1]), 6), round(float(np.nansum(y)), 4))


def _cached_prediction(label: str, factory, train: pd.DataFrame, horizon: int) -> np.ndarray:
    """The prediction of one component at one inner cutoff, kept in a cache.

    Cross-validation calls `fit` one time for each outer cutoff. Two adjacent outer cutoffs
    share almost all of their inner cutoffs. A prediction at an inner cutoff uses only the
    rows on or before that cutoff. Therefore the cached value is always correct.

    The label identifies the component. Two blends with different snowpack components must
    not share a cache entry.
    """
    key = (label, train[TIME_COL].iloc[-1], horizon, _fingerprint(train))
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
        snow_features: list[str] | None = None,
        snow_name: str = SNOW_NAME,
        horizon: int = 24,
        history_years: int = 15,
        max_cutoffs: int | None = None,
        min_rows: int = 20,
        name: str = "blend",
    ):
        super().__init__(name=name)
        self.snow_features = snow_features or list(SNOW_FEATURES)
        self.snow_name = snow_name
        self.horizon = horizon
        self.history_years = history_years
        self.max_cutoffs = max_cutoffs
        self.min_rows = min_rows
        self.weights = {season: default_weights(horizon) for season in SEASON_MONTHS}
        self.fitted_seasons: list[str] = []
        self.n_weight_cutoffs = 0
        self._fitted: list[Forecaster] = []
        self._factories = self.component_factories()

    def component_factories(self) -> list[tuple[str, object]]:
        """The 2 models that this model mixes. The registry also holds each one separately."""
        return [
            (
                self.snow_name,
                lambda: SweRegressionForecaster(features=self.snow_features, name=self.snow_name),
            ),
            (
                UNIVARIATE_NAME,
                lambda: HoltWintersForecaster(
                    trend="add", seasonal="add", seasonal_periods=12, damped_trend=True
                ),
            ),
        ]

    def _walk_forward(self, data: pd.DataFrame) -> tuple | None:
        """Component predictions, actuals and issue seasons at every inner cutoff."""
        cutoffs = valid_cutoffs(data, self.history_years, self.horizon)
        if self.max_cutoffs is not None:
            cutoffs = cutoffs[-self.max_cutoffs :]
        observed = data.set_index(TIME_COL)[TARGET_COL]
        factories = self._factories
        actuals, predictions, seasons = [], [], []
        for cutoff in cutoffs:
            train = data[data[TIME_COL] <= cutoff]
            months = [cutoff + relativedelta(months=i) for i in range(1, self.horizon + 1)]
            rows = [
                _cached_prediction(label, factory, train, self.horizon)
                for label, factory in factories
            ]
            actuals.append(observed.reindex(months).to_numpy(dtype=float))
            predictions.append(np.stack(rows))
            seasons.append(issue_season(cutoff))
        self.n_weight_cutoffs = len(actuals)
        if not actuals:
            return None
        return np.array(actuals), np.stack(predictions), np.array(seasons, dtype=object)

    def _fit_weights(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        result = self._walk_forward(data)
        weights = {season: default_weights(self.horizon) for season in SEASON_MONTHS}
        self.fitted_seasons = []
        if result is None:
            return weights
        actual, pred, seasons = result
        snow, univariate = pred[:, 0, :], pred[:, 1, :]
        for season in SEASON_MONTHS:
            in_season = seasons == season
            loss = np.zeros((self.horizon, len(WEIGHT_GRID)))
            enough = True
            for lead in range(self.horizon):
                usable = (
                    in_season
                    & np.isfinite(actual[:, lead])
                    & np.isfinite(snow[:, lead])
                    & np.isfinite(univariate[:, lead])
                )
                if usable.sum() < self.min_rows:
                    enough = False
                    break
                a = actual[usable, lead][:, None]
                s = snow[usable, lead][:, None]
                u = univariate[usable, lead][:, None]
                loss[lead] = np.abs(a - (u + WEIGHT_GRID[None, :] * (s - u))).sum(axis=0)
            if enough:
                weights[season] = monotone_weight_path(loss)
                self.fitted_seasons.append(season)
        return weights

    def fit(self, data: pd.DataFrame) -> Self:
        df = data.sort_values(TIME_COL).reset_index(drop=True)
        self.weights = self._fit_weights(df)
        self._fitted = [factory().fit(df) for _, factory in self._factories]
        self.last_date = df[TIME_COL].iloc[-1]
        self.is_fitted = True
        return self

    def weights_for(self, h: int, season: str | None = None) -> np.ndarray:
        """The weight curve for one issue season, set to a length of `h` leads."""
        curve = self.weights[season or issue_season(self.last_date)]
        if h <= len(curve):
            return curve[:h]
        return np.concatenate([curve, np.full(h - len(curve), curve[-1])])

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        origin = start_date or self.last_date
        snow, univariate = (f.predict(h, start_date) for f in self._fitted)
        w = self.weights_for(h, issue_season(origin))
        return pd.DataFrame(
            {
                TIME_COL: [origin + relativedelta(months=i) for i in range(1, h + 1)],
                "target": TARGET_COL,
                "pred": w * snow["pred"].to_numpy(dtype=float)
                + (1 - w) * univariate["pred"].to_numpy(dtype=float),
                "model_name": self.name,
            }
        )

    def contributions(self, h: int) -> pd.DataFrame:
        """The terms that add to the blended point forecast.

        The model multiplies each snowpack term by the weight for its lead. The univariate
        part has no terms for the inputs, so it goes into the reference path. The reference
        path is thus the part of the forecast that no named input changes.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before explanation")
        snow, univariate = self._fitted
        leads = range(1, h + 1)
        weight = pd.Series(self.weights_for(h), index=leads)
        share = pd.Series(univariate.predict(h)["pred"].to_numpy(dtype=float), index=leads)
        out = snow.contributions(h)
        out["snow_weight"] = out["h"].map(weight)
        out["contribution_ft"] = out["contribution_ft"] * out["snow_weight"]
        reference = out["input"] == "reference_path"
        out.loc[reference, "contribution_ft"] += (
            1.0 - out.loc[reference, "snow_weight"]
        ) * out.loc[reference, "h"].map(share)
        return out

    def get_metrics(self) -> dict[str, object]:
        season = issue_season(self.last_date) if self.is_fitted else None
        curve = self.weights[season] if season else default_weights(self.horizon)
        return {
            "components": f"{self.snow_name},{UNIVARIATE_NAME}",
            "n_weight_cutoffs": self.n_weight_cutoffs,
            "fitted_seasons": ",".join(self.fitted_seasons) or "none",
            "issue_season": season or "unfitted",
            **{f"weight_h{h}": float(curve[h - 1]) for h in (1, 6, 12, 24) if h <= len(curve)},
        }
