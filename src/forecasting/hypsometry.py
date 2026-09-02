"""South-arm elevation-area-volume lookups from the USGS 2023 topobathymetric tables
(Root, 2023, https://doi.org/10.5066/P9DGG75W), thinned to 0.1 ft steps. Elevations are
NGVD29 feet to match the Saltair gauge."""

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


def area_km2(elev_ft) -> np.ndarray | float:
    t = table()
    return np.interp(elev_ft, t["elev_ft_ngvd29"], t["area_km2"])


def volume_kaf(elev_ft) -> np.ndarray | float:
    t = table()
    return np.interp(elev_ft, t["elev_ft_ngvd29"], t["volume_kaf"])


def elevation_ft(volume) -> np.ndarray | float:
    t = table()
    return np.interp(volume, t["volume_kaf"], t["elev_ft_ngvd29"])
