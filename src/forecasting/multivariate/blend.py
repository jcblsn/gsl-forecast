"""The prototype headline model. It mixes component models with fitted weights.

Snowpack controls the lake level for the next 6 to 12 months. It gives no information about
the winter after next. Therefore the best model changes with the forecast lead.

The model puts a weight on each component. The weights sum to 1 at every lead. The last
component is the anchor, `ets_damped_s12`, which uses the lake record alone. The weight on
everything except the anchor is the covariate share, and that share must not increase with
the lead. Inside the share the mix is free, so a component that is strong at short leads and
a component that is strong at long leads can trade places as the lead grows.

The model fits the weights for each lead and for each issue season.

The issue season is necessary. A lead of 6 months from a February issue gives the spring
peak, which the current snowpack controls. The same lead from an August issue gives a month
in the next accumulation season, which the current snowpack does not control.

The model fits the weight curves with a walk-forward pass in the training data. The search
finds the curve with the lowest total absolute error under the condition above. It does not
fit each lead separately and then correct the result.

The inner pass uses the same 15-year window as the harness. A longer window gives less noise
in the weights, but it gives a biased result. Before approximately 1995 the SNOTEL record is
too short to fit the snowpack model. The inner pass then records a failure that cannot occur
now, and gives the snowpack model approximately half of the correct weight.
"""

from datetime import date
from itertools import product
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
PAIR_STEP = 0.01
SIMPLEX_STEP = 0.05
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


def simplex_grid(k: int, step: float) -> np.ndarray:
    """Every weight vector of length k on the simplex, in steps of `step`.

    The rows are sorted by the covariate share, which is 1 minus the last weight. The
    dynamic program needs that order, because its condition is on the share.
    """
    n = int(round(1.0 / step))
    rows = [
        np.array([*counts, n - sum(counts)], dtype=float) / n
        for counts in product(range(n + 1), repeat=k - 1)
        if sum(counts) <= n
    ]
    grid = np.array(rows)
    return grid[np.argsort(1.0 - grid[:, -1], kind="stable")]


def covariate_share(grid: np.ndarray) -> np.ndarray:
    return 1.0 - grid[:, -1]


def default_weights(horizon: int, k: int = 2) -> np.ndarray:
    """The fixed ramp. The model uses it when the training data gives too few cutoffs.

    The share falls from 1 at lead 6 to 0 at lead 24, and the covariate components hold
    equal parts of it.
    """
    leads = np.arange(1, horizon + 1)
    span = ZERO_WEIGHT_LEAD - FULL_WEIGHT_LEAD
    share = np.clip((ZERO_WEIGHT_LEAD - leads) / span, 0.0, 1.0)
    weights = np.zeros((horizon, k))
    weights[:, :-1] = (share / (k - 1))[:, None]
    weights[:, -1] = 1.0 - share
    return weights


def monotone_weight_path(loss: np.ndarray, share: np.ndarray) -> np.ndarray:
    """The grid point at each lead with the lowest total loss, under the share condition.

    `loss[i, j]` is the total absolute error at lead i + 1 for grid point j, and `share[j]`
    is the covariate share of that point. `share` must not increase with the lead. A dynamic
    program finds the minimum over all leads under the condition, so the condition is part
    of the objective and not a correction to a free fit.

    Grid points with an equal share form one group. The program takes the best point of
    every group at or above the current share. For an equal loss it selects the lower share,
    and inside a group the earlier point.
    """
    n_leads, n_grid = loss.shape
    bounds = np.flatnonzero(np.diff(share) > 0) + 1
    groups = list(zip([0, *bounds], [*bounds, n_grid], strict=True))

    cost = np.full((n_leads, n_grid), np.inf)
    previous = np.zeros((n_leads, n_grid), dtype=int)
    cost[0] = loss[0]
    for i in range(1, n_leads):
        best_value, best_index = np.inf, 0
        carried_value = np.empty(n_grid)
        carried_index = np.empty(n_grid, dtype=int)
        for start, end in reversed(groups):
            inside = start + int(np.argmin(cost[i - 1, start:end]))
            if cost[i - 1, inside] <= best_value:
                best_value, best_index = cost[i - 1, inside], inside
            carried_value[start:end] = best_value
            carried_index[start:end] = best_index
        cost[i] = loss[i] + carried_value
        previous[i] = carried_index
    selected = np.zeros(n_leads, dtype=int)
    selected[-1] = int(np.argmin(cost[-1]))
    for i in range(n_leads - 1, 0, -1):
        selected[i - 1] = previous[i, selected[i]]
    return selected


def _fingerprint(train: pd.DataFrame) -> tuple:
    y = train[TARGET_COL].to_numpy(dtype=float)
    return (len(y), round(float(y[0]), 6), round(float(y[-1]), 6), round(float(np.nansum(y)), 4))


def _cached_prediction(label: str, factory, train: pd.DataFrame, horizon: int) -> np.ndarray:
    """The prediction of one component at one inner cutoff, kept in a cache.

    Cross-validation calls `fit` one time for each outer cutoff. Two adjacent outer cutoffs
    share almost all of their inner cutoffs. A prediction at an inner cutoff uses only the
    rows on or before that cutoff. Therefore the cached value is always correct.

    The label identifies the component, and it carries the component's settings, so 2
    components with the same name and different features cannot share an entry.
    """
    key = (label, train[TIME_COL].iloc[-1], horizon, _fingerprint(train))
    hit = _CACHE.get(key)
    if hit is None:
        hit = factory().fit(train).predict(horizon)["pred"].to_numpy(dtype=float)
        if len(_CACHE) > _CACHE_LIMIT:
            _CACHE.clear()
        _CACHE[key] = hit
    return hit


def _settings_label(name: str, factory) -> str:
    """A label that separates 2 components with the same name and different settings."""
    settings = sorted(f"{k}={v}" for k, v in factory().get_metrics().items())
    return f"{name}|{'|'.join(settings)}"


class BlendForecaster(Forecaster):
    def __init__(
        self,
        snow_features: list[str] | None = None,
        snow_name: str = SNOW_NAME,
        components: list[tuple[str, object]] | None = None,
        horizon: int = 24,
        history_years: int = 15,
        max_cutoffs: int | None = None,
        min_rows: int = 20,
        weight_step: float | None = None,
        name: str = "blend",
    ):
        super().__init__(name=name)
        self.snow_features = snow_features or list(SNOW_FEATURES)
        self.snow_name = snow_name
        self.horizon = horizon
        self.history_years = history_years
        self.max_cutoffs = max_cutoffs
        self.min_rows = min_rows
        self._factories = components or self.component_factories()
        self.component_names = [label for label, _ in self._factories]
        self.k = len(self._factories)
        self.weight_step = weight_step or (PAIR_STEP if self.k == 2 else SIMPLEX_STEP)
        self._grid = simplex_grid(self.k, self.weight_step)
        self._share = covariate_share(self._grid)
        self.weights = {season: default_weights(horizon, self.k) for season in SEASON_MONTHS}
        self.fitted_seasons: list[str] = []
        self.n_weight_cutoffs = 0
        self._fitted: list[Forecaster] = []
        self._labels = [_settings_label(label, f) for label, f in self._factories]

    def component_factories(self) -> list[tuple[str, object]]:
        """The 2 models that this model mixes. The registry also holds each one separately.

        The last entry is the anchor. It uses the lake record alone, and it holds the weight
        that the covariate components give up as the lead grows.
        """
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

    def feature_columns(self) -> list[str]:
        """Every column the components read, so the availability test covers the blend."""
        seen: dict[str, None] = {}
        for _, factory in self._factories:
            for column in factory().feature_columns():
                seen[column] = None
        return list(seen)

    def _walk_forward(self, data: pd.DataFrame) -> tuple | None:
        """Component predictions, actuals and issue seasons at every inner cutoff."""
        cutoffs = valid_cutoffs(data, self.history_years, self.horizon)
        if self.max_cutoffs is not None:
            cutoffs = cutoffs[-self.max_cutoffs :]
        observed = data.set_index(TIME_COL)[TARGET_COL]
        actuals, predictions, seasons = [], [], []
        for cutoff in cutoffs:
            train = data[data[TIME_COL] <= cutoff]
            months = [cutoff + relativedelta(months=i) for i in range(1, self.horizon + 1)]
            rows = [
                _cached_prediction(label, factory, train, self.horizon)
                for label, (_, factory) in zip(self._labels, self._factories, strict=True)
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
        weights = {season: default_weights(self.horizon, self.k) for season in SEASON_MONTHS}
        self.fitted_seasons = []
        if result is None:
            return weights
        actual, pred, seasons = result
        for season in SEASON_MONTHS:
            in_season = seasons == season
            loss = np.zeros((self.horizon, len(self._grid)))
            enough = True
            for lead in range(self.horizon):
                usable = in_season & np.isfinite(actual[:, lead])
                for component in range(self.k):
                    usable = usable & np.isfinite(pred[:, component, lead])
                if usable.sum() < self.min_rows:
                    enough = False
                    break
                a = actual[usable, lead][:, None]
                mixed = pred[usable, :, lead] @ self._grid.T
                loss[lead] = np.abs(a - mixed).sum(axis=0)
            if enough:
                weights[season] = self._grid[monotone_weight_path(loss, self._share)]
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
        """The weights for one issue season, set to a length of `h` leads."""
        curve = self.weights[season or issue_season(self.last_date)]
        if h <= len(curve):
            return curve[:h]
        pad = np.repeat(curve[-1][None, :], h - len(curve), axis=0)
        return np.concatenate([curve, pad])

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        origin = start_date or self.last_date
        parts = np.stack(
            [f.predict(h, start_date)["pred"].to_numpy(dtype=float) for f in self._fitted]
        )
        w = self.weights_for(h, issue_season(origin))
        return pd.DataFrame(
            {
                TIME_COL: [origin + relativedelta(months=i) for i in range(1, h + 1)],
                "target": TARGET_COL,
                "pred": (w.T * parts).sum(axis=0),
                "model_name": self.name,
            }
        )

    def contributions(self, h: int) -> pd.DataFrame:
        """The terms that add to the blended point forecast.

        The model multiplies the terms of each component by that component's weight, and it
        adds the terms that name the same input. A component with no `contributions` method
        gives no term for an input, so its whole weighted forecast joins the reference path.
        The anchor is always such a component. The reference path is thus the part of the
        forecast that no named input changes, and the terms still add to the prediction.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before explanation")
        leads = range(1, h + 1)
        weights = self.weights_for(h)
        frames, unexplained = [], pd.Series(0.0, index=leads)
        for i, model in enumerate(self._fitted):
            scale = pd.Series(weights[:, i], index=leads)
            if not hasattr(model, "contributions"):
                path = pd.Series(model.predict(h)["pred"].to_numpy(dtype=float), index=leads)
                unexplained = unexplained + scale * path
                continue
            part = model.contributions(h)
            part["contribution_ft"] = part["contribution_ft"] * part["h"].map(scale)
            frames.append(part)
        out = (
            pd.concat(frames, ignore_index=True)
            .groupby(["month", "h", "input"], as_index=False)
            .agg(
                value=("value", "first"),
                reference=("reference", "first"),
                contribution_ft=("contribution_ft", "sum"),
            )
        )
        reference = out["input"] == "reference_path"
        out.loc[reference, "contribution_ft"] += out.loc[reference, "h"].map(unexplained)
        explained = sum(
            weights[:, i] for i, m in enumerate(self._fitted) if hasattr(m, "contributions")
        )
        out["covariate_weight"] = out["h"].map(pd.Series(explained, index=leads))
        out["snow_weight"] = out["covariate_weight"]
        return out

    def get_metrics(self) -> dict[str, object]:
        season = issue_season(self.last_date) if self.is_fitted else None
        curve = self.weights[season] if season else default_weights(self.horizon, self.k)
        share = 1.0 - curve[:, -1]
        return {
            "components": ",".join(self.component_names),
            "weight_step": self.weight_step,
            "n_weight_cutoffs": self.n_weight_cutoffs,
            "fitted_seasons": ",".join(self.fitted_seasons) or "none",
            "issue_season": season or "unfitted",
            **{
                f"covariate_weight_h{h}": round(float(share[h - 1]), 3)
                for h in (1, 6, 12, 24)
                if h <= len(curve)
            },
        }
