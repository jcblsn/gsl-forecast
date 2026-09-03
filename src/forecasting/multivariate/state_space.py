"""A linear Gaussian state-space water balance, fitted by maximum likelihood.

`inflow_chain` is a deterministic version of this model. It fits the monthly bucket step on
the observed inflow, then runs the step on the predicted inflow, and it rolls the level
forward with no account of the error it carries. Two weaknesses follow, and section 10 of
`docs/model-spec.md` records both: the fitted response to inflow is biased, because the
predictor in the fit is not the predictor at run time; and the recursion accumulates error
that nothing measures.

This model puts the level in a latent state and the gauge in an observation equation:

    level:       m_t = phi * m_(t-1) + a_(month) + b * Q_t + w_t
    observation: y_t = m_t + v_t

`Q_t` is the tributary inflow in kaf for month t. `a_month` absorbs the mean net evaporation
and diversion for that calendar month. `phi` damps the level, in the place of the `c_m` term
in `inflow_chain`. `w_t` is the process error and `v_t` is the measurement error.

Three properties follow from the form, and each one answers a recorded open question.

1. The 24-month path is one recursion, not 24 separate fits. The path is therefore smooth by
   construction.
2. The variance of the state grows with each step, so the interval widens with the lead
   without a rule that makes it do so.
3. One coefficient `b` covers every calendar month. A volume of water raises the lake by the
   same amount in March and in August; the month changes the evaporation, which `a_month`
   holds. `inflow_chain` fits a separate response for each calendar month on about 63 rows.

Stage one, the inflow for a future month, is unchanged: this model holds an
`InflowChainForecaster` and calls its `inflow_forecast`. So a comparison of the 2 models
measures the change in stage two alone.

This is phase 1 of the spike in the plan: Gaussian, maximum likelihood, no new dependency.
Phase 2, which gives the monthly terms a shared prior and needs a sampler, follows only if
this phase passes the criterion in `docs/autoresearch.log`.
"""

import logging
import warnings
from datetime import date
from typing import Self

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

from ..base import Forecaster
from .inflow_chain import INFLOW_COL, InflowChainForecaster
from .regression import TARGET_COL, TIME_COL, require_columns

N_MONTHS = 12


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def _shift_back(values: np.ndarray) -> np.ndarray:
    """Element t holds what was element t + 1. The last element repeats the one before it.

    The last element drives a state after the end of the sample, which nothing observes, so
    its value does not change any result.
    """
    out = np.empty_like(values)
    out[:-1] = values[1:]
    out[-1] = values[-1]
    return out


class WaterBalanceSSM(sm.tsa.statespace.MLEModel):
    """The state-space form. `exog` is the monthly inflow, already free of missing values."""

    def __init__(self, endog: np.ndarray, inflow: np.ndarray, months: np.ndarray):
        super().__init__(endog, k_states=1, k_posdef=1, initialization="approximate_diffuse")
        # The inflow is in kaf, so its coefficient is near 0.001 while a monthly term is near
        # 0.1 and phi_raw is near 6. The optimizer fails a line search on that spread, so the
        # column enters in units of its own standard deviation.
        self.inflow_scale = float(np.std(inflow)) or 1.0
        # statsmodels writes the transition as alpha_(t+1) = c_t + T alpha_t, so the state
        # intercept at t drives the level at t + 1. The driver of the level in month t is
        # therefore month t's own term and month t's own inflow, held at index t - 1. Both
        # arrays move back 1 step. Without the shift the fit puts each monthly term 1 month
        # early, and the forecast peaks and troughs 1 month before the record does.
        self.inflow = _shift_back(np.asarray(inflow, dtype=float)) / self.inflow_scale
        self.month_index = _shift_back(np.asarray(months, dtype=int)) - 1
        self["design", 0, 0] = 1.0
        self["selection", 0, 0] = 1.0
        self.ssm["state_intercept"] = np.zeros((1, self.nobs))

    @property
    def param_names(self) -> list[str]:
        return [
            "phi_raw",
            "inflow_ft_per_sd",
            *[f"month_{m + 1}_ft" for m in range(N_MONTHS)],
            "log_process_sd",
            "log_measurement_sd",
        ]

    @property
    def start_params(self) -> np.ndarray:
        return np.array([6.0, 0.2, *np.zeros(N_MONTHS), np.log(0.1), np.log(0.02)])

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        phi = _sigmoid(params[0])
        b = params[1]
        monthly = params[2 : 2 + N_MONTHS]
        self["transition", 0, 0] = phi
        self.ssm["state_intercept"] = (monthly[self.month_index] + b * self.inflow).reshape(
            1, self.nobs
        )
        self["state_cov", 0, 0] = float(np.exp(2.0 * params[-2]))
        self["obs_cov", 0, 0] = float(np.exp(2.0 * params[-1]))

    def parts(self, params: np.ndarray) -> dict[str, object]:
        """The parameters on their own scale, for the forward recursion and the metrics."""
        return {
            "phi": _sigmoid(params[0]),
            "b": float(params[1]) / self.inflow_scale,
            "monthly": np.asarray(params[2 : 2 + N_MONTHS], dtype=float),
            "process_var": float(np.exp(2.0 * params[-2])),
            "measurement_var": float(np.exp(2.0 * params[-1])),
        }


class StateSpaceForecaster(Forecaster):
    def __init__(
        self,
        snow_features: list[str] | None = None,
        min_obs: int = 10,
        maxiter: int = 200,
        name: str = "state_space",
    ):
        super().__init__(name=name)
        self.min_obs = min_obs
        self.maxiter = maxiter
        self._stage_one = InflowChainForecaster(snow_features=snow_features, min_obs=min_obs)
        self.snow_features = self._stage_one.snow_features
        self._data: pd.DataFrame | None = None
        self._parts: dict[str, object] | None = None
        self._state_mean = 0.0
        self._state_var = 0.0
        self._center = 0.0
        self.converged = False

    def feature_columns(self) -> list[str]:
        return self._stage_one.feature_columns()

    def _monthly_inflow(self, df: pd.DataFrame) -> np.ndarray:
        """The inflow column with the calendar-month mean in place of a missing value.

        The state intercept holds this term, and the filter cannot take a missing intercept.
        The early record has gaps, and the mean for that month is the honest stand-in.
        """
        inflow = df[INFLOW_COL].astype(float)
        by_month = inflow.groupby(df[TIME_COL].dt.month).transform("mean")
        return inflow.fillna(by_month).fillna(inflow.mean()).fillna(0.0).to_numpy()

    def fit(self, data: pd.DataFrame) -> Self:
        require_columns(data, [TIME_COL, TARGET_COL, INFLOW_COL, *self.snow_features])
        df = data.sort_values(TIME_COL).reset_index(drop=True)
        self._data = df
        self.last_date = df[TIME_COL].iloc[-1]
        self._stage_one.fit(df)

        level = df[TARGET_COL].to_numpy(dtype=float)
        self._center = float(np.nanmean(level))
        model = WaterBalanceSSM(
            level - self._center, self._monthly_inflow(df), df[TIME_COL].dt.month.to_numpy()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp=False, maxiter=self.maxiter)
        retvals = getattr(result, "mle_retvals", None) or {}
        self.converged = bool(retvals.get("converged", True))
        if not self.converged:
            logging.debug(f"{self.name}: the optimizer stopped without converging")
        self._parts = model.parts(np.asarray(result.params, dtype=float))
        self._state_mean = float(result.filtered_state[0, -1])
        self._state_var = float(result.filtered_state_cov[0, 0, -1])
        self.is_fitted = True
        return self

    def _roll_forward(self, h: int, origin: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        """The mean and variance of the observation at each lead, from one recursion."""
        p = self._parts
        phi, b, monthly = p["phi"], p["b"], p["monthly"]
        mean, var = self._state_mean, self._state_var
        means, variances = [], []
        for lead in range(1, h + 1):
            month = (origin + relativedelta(months=lead)).month
            inflow = self._stage_one.inflow_forecast(lead)
            if not np.isfinite(inflow):
                inflow = 0.0
            mean = phi * mean + monthly[month - 1] + b * inflow
            var = phi**2 * var + p["process_var"]
            means.append(mean + self._center)
            variances.append(var + p["measurement_var"])
        return np.array(means), np.array(variances)

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        origin = pd.Timestamp(start_date or self.last_date)
        means, _ = self._roll_forward(h, origin)
        return pd.DataFrame(
            {
                TIME_COL: [origin + relativedelta(months=i) for i in range(1, h + 1)],
                "target": TARGET_COL,
                "pred": means,
                "model_name": self.name,
            }
        )

    def predict_quantiles(self, h: int, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
        """The interval the model gives itself, which widens with the lead by construction.

        The harness scores every model with intervals from its walk-forward errors, so this
        frame is a second reading and not the one the comparison uses.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        from scipy.stats import norm

        origin = pd.Timestamp(self.last_date)
        means, variances = self._roll_forward(h, origin)
        sd = np.sqrt(variances)
        out = pd.DataFrame(
            {
                TIME_COL: [origin + relativedelta(months=i) for i in range(1, h + 1)],
                "h": range(1, h + 1),
                "pred": means,
            }
        )
        for q in quantiles:
            out[f"q{int(round(q * 100)):02d}"] = means + norm.ppf(q) * sd
        return out

    def get_metrics(self) -> dict[str, object]:
        if self._parts is None:
            return {"fitted": False}
        p = self._parts
        return {
            "phi": round(p["phi"], 5),
            "inflow_ft_per_kaf": round(p["b"], 6),
            "process_sd_ft": round(float(np.sqrt(p["process_var"])), 4),
            "measurement_sd_ft": round(float(np.sqrt(p["measurement_var"])), 4),
            "converged": self.converged,
            "n_params": 2 + N_MONTHS + 2,
        }


def log_if_degenerate(model: StateSpaceForecaster) -> None:
    """A fit that puts every difference in the measurement term has no water balance left."""
    if model._parts and model._parts["process_var"] < 1e-8:
        logging.warning(f"{model.name}: the process variance collapsed to zero")
