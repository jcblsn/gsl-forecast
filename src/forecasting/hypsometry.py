"""Interpolate USGS south-arm area and volume at 0.1-ft elevation steps.

The source is Root (2023), https://doi.org/10.5066/P9DGG75W. Elevations use ft NGVD 29.
Lookups reject values outside the published table by default because `numpy.interp` would
otherwise clamp them silently.
"""

import os
from functools import cache

import numpy as np
import pandas as pd

TABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "external",
    "gsl_south_arm_hypsometry.csv",
)


@cache
def table() -> pd.DataFrame:
    return pd.read_csv(TABLE).sort_values("elev_ft_ngvd29").reset_index(drop=True)


def elevation_domain() -> tuple[float, float]:
    """Return the table's minimum and maximum elevation in ft NGVD 29."""
    t = table()
    return float(t["elev_ft_ngvd29"].iloc[0]), float(t["elev_ft_ngvd29"].iloc[-1])


def volume_domain() -> tuple[float, float]:
    """The lowest and highest storage the table covers, in kaf."""
    t = table()
    return float(t["volume_kaf"].iloc[0]), float(t["volume_kaf"].iloc[-1])


def _check(values, low: float, high: float, what: str, strict: bool) -> None:
    if not strict:
        return
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size and (finite.min() < low or finite.max() > high):
        raise ValueError(
            f"{what} outside the hypsometry table: "
            f"[{finite.min():.4g}, {finite.max():.4g}] is not within [{low:g}, {high:g}]"
        )


def area_km2(elev_ft, strict: bool = True) -> np.ndarray | float:
    t = table()
    _check(elev_ft, *elevation_domain(), "Elevation", strict)
    return np.interp(elev_ft, t["elev_ft_ngvd29"], t["area_km2"])


def volume_kaf(elev_ft, strict: bool = True) -> np.ndarray | float:
    t = table()
    _check(elev_ft, *elevation_domain(), "Elevation", strict)
    return np.interp(elev_ft, t["elev_ft_ngvd29"], t["volume_kaf"])


def elevation_ft(volume, strict: bool = True) -> np.ndarray | float:
    t = table()
    _check(volume, *volume_domain(), "Storage", strict)
    return np.interp(volume, t["volume_kaf"], t["elev_ft_ngvd29"])
