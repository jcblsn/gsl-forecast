"""Experimental structural state-space forecast on south-arm storage.

The hypsometry converts monthly-mean elevation to storage before fitting. A standard local
linear trend with deterministic monthly seasonality evolves that storage, and the forecast is
converted back to elevation. This is a statistical structural time-series model. It does not
claim to close a water balance or use forecast inflow as if it were known.
"""

from datetime import date
from typing import Self

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

from .. import hypsometry
from ..base import Forecaster
from .regression import TARGET_COL, TIME_COL, require_columns

STORAGE_SCALE_KAF = 1_000.0


class StateSpaceForecaster(Forecaster):
    """Local-linear-trend state-space model in hypsometric storage coordinates."""

    def __init__(self, maxiter: int = 300, name: str = "state_space"):
        super().__init__(name=name)
        self.maxiter = maxiter
        self._result = None

    def feature_columns(self) -> list[str]:
        return []

    def fit(self, data: pd.DataFrame) -> Self:
        require_columns(data, [TIME_COL, TARGET_COL])
        frame = data.sort_values(TIME_COL).reset_index(drop=True)
        if len(frame) < 24:
            raise ValueError("state_space needs at least 24 monthly observations")
        elevation = frame[TARGET_COL].to_numpy(dtype=float)
        limits = hypsometry.table()["elev_ft_ngvd29"].agg(["min", "max"])
        if (
            not np.isfinite(elevation).all()
            or (elevation < limits["min"]).any()
            or (elevation > limits["max"]).any()
        ):
            raise ValueError("state_space elevation is outside the hypsometry domain")
        storage = np.asarray(hypsometry.volume_kaf(elevation), dtype=float)
        if not np.isfinite(storage).all():
            raise ValueError("state_space received a non-finite storage observation")
        model = sm.tsa.UnobservedComponents(
            storage / STORAGE_SCALE_KAF,
            level=True,
            trend=True,
            seasonal=12,
            irregular=False,
            stochastic_level=True,
            stochastic_trend=True,
            stochastic_seasonal=False,
            use_exact_diffuse=True,
        )
        result = model.fit(disp=False, maxiter=self.maxiter)
        if not bool(result.mle_retvals.get("converged", False)):
            raise RuntimeError("state_space maximum-likelihood fit did not converge")
        self._result = result
        self.last_date = pd.Timestamp(frame[TIME_COL].iloc[-1])
        self.is_fitted = True
        return self

    def _forecast_storage(self, h: int):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        if h < 1:
            raise ValueError("Forecast horizon must be positive")
        return self._result.get_forecast(h)

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        forecast = self._forecast_storage(h)
        origin = pd.Timestamp(start_date or self.last_date)
        elevation = hypsometry.elevation_ft(
            np.asarray(forecast.predicted_mean, dtype=float) * STORAGE_SCALE_KAF
        )
        return pd.DataFrame(
            {
                TIME_COL: [origin + relativedelta(months=i) for i in range(1, h + 1)],
                "target": TARGET_COL,
                "pred": elevation,
                "model_name": self.name,
            }
        )

    def predict_quantiles(self, h: int, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
        from scipy.stats import norm

        forecast = self._forecast_storage(h)
        origin = pd.Timestamp(self.last_date)
        mean = np.asarray(forecast.predicted_mean, dtype=float)
        sd = np.sqrt(np.asarray(forecast.var_pred_mean, dtype=float))
        out = pd.DataFrame(
            {
                TIME_COL: [origin + relativedelta(months=i) for i in range(1, h + 1)],
                "h": range(1, h + 1),
                "pred": hypsometry.elevation_ft(mean * STORAGE_SCALE_KAF),
            }
        )
        for quantile in quantiles:
            storage = (mean + norm.ppf(quantile) * sd) * STORAGE_SCALE_KAF
            out[f"q{int(round(quantile * 100)):02d}"] = hypsometry.elevation_ft(storage)
        return out

    def get_metrics(self) -> dict[str, object]:
        if self._result is None:
            return {"fitted": False}
        return {
            "state": "south_arm_storage_kaf",
            "structure": "local_linear_trend_with_monthly_seasonality",
            "converged": True,
            "n_params": len(self._result.params),
        }
