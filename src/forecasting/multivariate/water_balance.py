"""Experimental monthly storage balance for the south arm.

The model adds tributary inflow and precipitation, then subtracts salinity-adjusted reference
evaporation. A fitted residual combines unmeasured flux and measurement error. The state is
end-of-month storage; a centered observation operator converts it to monthly mean elevation.
Future inflow comes from a snowpack regression, and future weather uses monthly climatology.
"""

from datetime import date
from typing import Self

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from .. import hypsometry
from ..base import Forecaster
from .inflow_chain import InflowChainForecaster
from .regression import MIN_OBS, TARGET_COL, TIME_COL, require_columns, ridge_fit

EOM_COL = "elevation_eom_ft"
INFLOW_COL = "inflow_kaf_total"
SALT_COL = "salt_mass_mt"
WEATHER_COLS = ["tmax_f_kslc", "tmin_f_kslc", "prcp_in_kslc"]
DEFAULT_SNOW = ["swe_eom_gsl", "prec_wy_eom_gsl"]

# Acres per square kilometre, and the depth-times-area to volume conversion. A depth in feet
# over an area in acres is a volume in acre-feet; dividing by 1000 gives kaf.
ACRES_PER_KM2 = 247.105
# 1 kaf of brine at 1 g/L holds this many million tonnes of salt. Salinity is recovered from
# the carried salt mass so that it follows the volume the model itself is stepping.
KAF_GL_TO_MT = 1.2335e9 * 1e-6 / 1e6
# The gauged terminal sum over the estimated total delivery to the lake. Fitted, with this
# as the starting value and the reported fit in `get_metrics`.
DEFAULT_DELIVERY_RATIO = 0.8246
LATITUDE_DEG = 41.0
# Mid-month day of the year, for the extraterrestrial radiation term.
MID_MONTH_DOY = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])


def extraterrestrial_radiation(month: np.ndarray, latitude: float = LATITUDE_DEG) -> np.ndarray:
    """Ra in mm/day of equivalent evaporation, by calendar month (FAO-56).

    This is astronomy, not weather. It carries the seasonal shape of the energy available
    for evaporation, so the fitted temperature coefficient does not have to.
    """
    doy = MID_MONTH_DOY[np.asarray(month, dtype=int) - 1]
    phi = np.radians(latitude)
    distance = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    declination = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    sunset = np.arccos(np.clip(-np.tan(phi) * np.tan(declination), -1, 1))
    return (
        15.392
        * distance
        * (
            sunset * np.sin(phi) * np.sin(declination)
            + np.cos(phi) * np.cos(declination) * np.sin(sunset)
        )
    )


def hargreaves_depth_ft(
    tmax_f: np.ndarray, tmin_f: np.ndarray, month: np.ndarray, days: np.ndarray
) -> np.ndarray:
    """Reference evaporation depth in feet for a month, from the daily temperature range.

    Hargreaves-Samani needs only temperature, which is the one meteorological variable this
    lake has continuously since 1948. The daily range stands in for humidity and cloud.
    """
    tmax_c = (np.asarray(tmax_f, dtype=float) - 32) * 5 / 9
    tmin_c = (np.asarray(tmin_f, dtype=float) - 32) * 5 / 9
    mean_c = (tmax_c + tmin_c) / 2
    spread = np.sqrt(np.clip(tmax_c - tmin_c, 0.0, None))
    radiation = extraterrestrial_radiation(month)
    mm_per_day = 0.0023 * radiation * (mean_c + 17.8) * spread
    return mm_per_day * np.asarray(days, dtype=float) / 304.8


def salinity_factor(salinity_gl: np.ndarray, coefficient: float) -> np.ndarray:
    """Evaporation suppression by dissolved salt, as a fraction of fresh-water evaporation.

    Water activity falls roughly linearly with concentration over the range this lake
    occupies. The factor is clipped so a fitted coefficient cannot drive it negative.
    """
    return np.clip(1.0 - coefficient * np.asarray(salinity_gl, dtype=float) / 1000.0, 0.05, 1.0)


class WaterBalanceForecaster(Forecaster):
    """A storage balance with an explicit evaporation term and a named residual."""

    def __init__(
        self,
        snow_features: list[str] | None = None,
        min_obs: int = MIN_OBS,
        alpha: float | None = None,
        delivery_ratio: float | None = None,
        salinity: bool = True,
        name: str = "water_balance",
    ):
        super().__init__(name=name)
        self.snow_features = list(snow_features or DEFAULT_SNOW)
        self.min_obs = min_obs
        self.alpha = alpha
        self.delivery_ratio = delivery_ratio
        self.salinity = salinity
        self._data: pd.DataFrame | None = None
        self._inflow_model: InflowChainForecaster | None = None
        self._residual: dict[int, float] = {}
        self._climatology: pd.DataFrame | None = None
        self._coefficients: dict[str, float] = {}

    def feature_columns(self) -> list[str]:
        columns = [EOM_COL, INFLOW_COL, *WEATHER_COLS, *self.snow_features]
        if self.salinity:
            columns.append(SALT_COL)
        return columns

    def _frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """The per-month balance terms, each aligned to the step it belongs to."""
        df = data.sort_values(TIME_COL).reset_index(drop=True).copy()
        df["month_number"] = df[TIME_COL].dt.month
        df["days"] = df[TIME_COL].dt.days_in_month
        elevation = df[EOM_COL].to_numpy(dtype=float)
        inside = np.isfinite(elevation)
        volume = np.full(len(df), np.nan)
        area = np.full(len(df), np.nan)
        volume[inside] = hypsometry.volume_kaf(elevation[inside])
        area[inside] = hypsometry.area_km2(elevation[inside]) * ACRES_PER_KM2
        df["volume_kaf"] = volume
        df["area_acres"] = area
        df["area_prev"] = df["area_acres"].shift(1)
        df["volume_prev"] = df["volume_kaf"].shift(1)
        df["delta_v"] = df["volume_kaf"].diff()
        df["evap_ft"] = hargreaves_depth_ft(
            df["tmax_f_kslc"], df["tmin_f_kslc"], df["month_number"], df["days"]
        )
        df["precip_ft"] = df["prcp_in_kslc"] / 12.0
        if self.salinity and SALT_COL in df:
            df["salinity_prev"] = df[SALT_COL].shift(1) / df["volume_prev"] / KAF_GL_TO_MT
        else:
            df["salinity_prev"] = 0.0
        # A step is only a step between adjacent months. A gap in the record is not one.
        gap = df[TIME_COL].diff().dt.days
        df["consecutive"] = gap.between(28, 31)
        return df

    def _design(self, df: pd.DataFrame, salt_coefficient: float) -> tuple[np.ndarray, np.ndarray]:
        """Rows against dV, with a leading column of ones.

        `ridge_fit` reads column 0 as the unpenalized intercept and standardizes the rest,
        so the leading ones column is part of its contract and not a spare term.
        """
        area = df["area_prev"].to_numpy(dtype=float)
        factor = salinity_factor(df["salinity_prev"].to_numpy(dtype=float), salt_coefficient)
        columns = [
            np.ones(len(df)),
            df[INFLOW_COL].to_numpy(dtype=float),
            df["precip_ft"].to_numpy(dtype=float) * area / 1000.0,
            -df["evap_ft"].to_numpy(dtype=float) * factor * area / 1000.0,
        ]
        # 11 season terms, with January as the reference the intercept carries. Keeping all
        # 12 beside an intercept would make the design singular.
        for m in range(2, 13):
            columns.append((df["month_number"].to_numpy() == m).astype(float) * area / 1000.0)
        return np.column_stack(columns), df["delta_v"].to_numpy(dtype=float)

    # Positions in the fitted coefficient vector.
    INTERCEPT, INFLOW, PRECIP, EVAP, SEASON = 0, 1, 2, 3, 4

    def _usable(self, df: pd.DataFrame) -> np.ndarray:
        needed = ["delta_v", "area_prev", INFLOW_COL, "evap_ft", "precip_ft"]
        ok = np.array(df["consecutive"].to_numpy(dtype=bool), copy=True)
        for column in needed:
            ok &= np.isfinite(df[column].to_numpy(dtype=float))
        if self.salinity:
            ok &= np.isfinite(df["salinity_prev"].to_numpy(dtype=float))
        return ok

    def fit(self, data: pd.DataFrame) -> Self:
        columns = [TIME_COL, TARGET_COL, EOM_COL, INFLOW_COL, *WEATHER_COLS]
        require_columns(data, columns)
        df = self._frame(data)
        self._data = df
        self.last_date = df[TIME_COL].iloc[-1]
        ok = self._usable(df)
        if int(ok.sum()) < self.min_obs:
            raise ValueError(
                f"{self.name} needs {self.min_obs} closed monthly steps, found {int(ok.sum())}"
            )
        rows = df[ok]

        # The salinity coefficient enters the design multiplicatively, so it is chosen on a
        # small grid and the rest of the terms are solved in closed form for each candidate.
        candidates = [0.0] if not self.salinity else np.linspace(0.0, 3.0, 13)
        best = None
        for candidate in candidates:
            X, y = self._design(rows, float(candidate))
            beta = ridge_fit(X, y, self.alpha)
            residual = y - X @ beta
            score = float(np.mean(residual**2))
            if best is None or score < best[0]:
                best = (score, float(candidate), beta, residual)
        _, salt_coefficient, beta, residual = best
        seasonal = np.concatenate([[0.0], beta[self.SEASON :]])
        self._coefficients = {
            "delivery_ratio": 1.0 / beta[self.INFLOW] if beta[self.INFLOW] > 0 else np.nan,
            "intercept_kaf": float(beta[self.INTERCEPT]),
            "precip_scale": float(beta[self.PRECIP]),
            "evap_scale": float(beta[self.EVAP]),
            "salt_coefficient": salt_coefficient,
            "seasonal_ft": seasonal.tolist(),
        }
        self._beta = beta
        # `net_unmeasured_flux`: what the balance did not close, pooled by calendar month and
        # expressed as a depth so it scales with the lake instead of with the year.
        months = rows["month_number"].to_numpy()
        depth = residual * 1000.0 / rows["area_prev"].to_numpy(dtype=float)
        pooled = float(np.mean(depth))
        self._residual = {}
        for m in range(1, 13):
            sel = months == m
            n = int(sel.sum())
            # Shrink a thin month toward the pooled value, the same rule the interval layer
            # uses, so one unusual January cannot set the January residual on its own.
            weight = n / (n + 10.0)
            if n:
                self._residual[m] = weight * float(np.mean(depth[sel])) + (1 - weight) * pooled
            else:
                self._residual[m] = pooled
        self._residual_sd_kaf = float(np.std(residual))
        self._residual_sd_ft = float(np.std(depth))
        self._climatology = (
            rows.groupby("month_number")[["tmax_f_kslc", "tmin_f_kslc", "prcp_in_kslc"]]
            .mean()
            .reindex(range(1, 13))
            .ffill()
            .bfill()
        )
        self._inflow_model = InflowChainForecaster(
            snow_features=self.snow_features, min_obs=self.min_obs, alpha=self.alpha
        ).fit(data)
        self._mean_inflow = (
            df[df[INFLOW_COL].notna()].groupby("month_number")[INFLOW_COL].mean().to_dict()
        )
        self.is_fitted = True
        return self

    def _step(self, volume: float, salt_mass: float, month: int, days: int, inflow: float) -> float:
        """One month of the balance, from end-of-month storage to end-of-month storage."""
        low, high = hypsometry.volume_domain()
        volume = float(np.clip(volume, low, high))
        elevation = float(hypsometry.elevation_ft(volume))
        area = float(hypsometry.area_km2(elevation)) * ACRES_PER_KM2
        weather = self._climatology.loc[month]
        evaporation = float(
            hargreaves_depth_ft(
                np.array([weather["tmax_f_kslc"]]),
                np.array([weather["tmin_f_kslc"]]),
                np.array([month]),
                np.array([days]),
            )[0]
        )
        salinity = salt_mass / volume / KAF_GL_TO_MT if self.salinity and volume > 0 else 0.0
        coefficient = self._coefficients["salt_coefficient"]
        factor = float(salinity_factor(np.array([salinity]), coefficient)[0])
        precipitation = float(weather["prcp_in_kslc"]) / 12.0
        change = (
            self._coefficients["intercept_kaf"]
            + self._beta[self.INFLOW] * inflow
            + self._beta[self.PRECIP] * precipitation * area / 1000.0
            - self._beta[self.EVAP] * evaporation * factor * area / 1000.0
            + self._coefficients["seasonal_ft"][month - 1] * area / 1000.0
            + self._residual[month] * area / 1000.0
        )
        return volume + change

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        df = self._data
        origin = start_date or self.last_date
        low, high = hypsometry.volume_domain()
        volume = float(df["volume_kaf"].iloc[-1])
        if not np.isfinite(volume):
            volume = float(hypsometry.volume_kaf(float(df[TARGET_COL].iloc[-1]), strict=False))
        salt_mass = float(df[SALT_COL].iloc[-1]) if self.salinity and SALT_COL in df else 0.0
        if not np.isfinite(salt_mass):
            salt_mass = 0.0
        previous = volume
        months, values = [], []
        for i in range(1, h + 1):
            target = origin + relativedelta(months=i)
            inflow = self._inflow_model.inflow_forecast(i)
            if not np.isfinite(inflow):
                inflow = float(self._mean_inflow.get(target.month, 0.0))
            stepped = self._step(previous, salt_mass, target.month, target.days_in_month, inflow)
            volume = float(np.clip(stepped, low, high))
            # The published target is a monthly mean. The state is an instant at each month
            # end, so the mean of the 2 bracketing instants is the matching observation.
            mean_volume = float(np.clip((previous + volume) / 2.0, low, high))
            months.append(target)
            values.append(float(hypsometry.elevation_ft(mean_volume)))
            previous = volume
        return pd.DataFrame(
            {
                TIME_COL: months,
                "target": TARGET_COL,
                "pred": values,
                "model_name": self.name,
            }
        )

    def get_metrics(self) -> dict[str, object]:
        if not self.is_fitted:
            return {}
        return {
            "fitted_delivery_ratio": round(self._coefficients["delivery_ratio"], 4),
            "evap_scale": round(self._coefficients["evap_scale"], 4),
            "precip_scale": round(self._coefficients["precip_scale"], 4),
            "salt_coefficient": round(self._coefficients["salt_coefficient"], 4),
            "net_unmeasured_flux_sd_kaf": round(self._residual_sd_kaf, 2),
            "net_unmeasured_flux_sd_ft": round(self._residual_sd_ft, 4),
            "salinity": self.salinity,
            "min_obs": self.min_obs,
        }
