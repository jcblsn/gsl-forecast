"""Compare reconstructed salinity with an external HydroShare density compilation.

The check fits a density-to-salinity conversion on paired shallow UGS samples, then compares
monthly south-arm values. Shared observations or methods may remain, so agreement is a
cross-check rather than an independent validation.
"""

import logging

import duckdb
import numpy as np
import pandas as pd

from src.pipeline.brine import KAF_GL_TO_MT, UPPER_BRINE_MAX_FT
from src.pipeline.usgs import get_with_retry

HYDROSHARE = (
    "https://www.hydroshare.org/django_irods/download/438b751c1ff84555a36592345fcfa6b7"
    "/data/contents/GSL_ow_density_data_Jun1966_Feb2024.csv"
)
# The Union Pacific causeway runs near this latitude. South of it is Gilbert Bay, which is
# the arm this project forecasts.
CAUSEWAY_LAT = 41.1


def density_to_salinity(samples: pd.DataFrame) -> tuple[float, float, float]:
    """Fit salinity on density using the UGS samples that report both."""
    paired = samples.dropna(subset=["density_g_cm3", "salinity_gl"])
    paired = paired[paired["density_g_cm3"].between(1.0, 1.30)]
    if len(paired) < 30:
        raise ValueError("Too few paired density and salinity samples to fit the conversion")
    x = paired["density_g_cm3"].to_numpy(dtype=float)
    y = paired["salinity_gl"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    rmse = float(np.std(y - (slope * x + intercept)))
    return float(slope), float(intercept), rmse


def fetch_independent_salinity(slope: float, intercept: float) -> pd.DataFrame:
    """Return monthly south-arm salinity from the external HydroShare compilation."""
    text = get_with_retry(HYDROSHARE, timeout=300).text
    raw = pd.read_csv(pd.io.common.StringIO(text))
    raw.columns = [c.strip() for c in raw.columns]
    raw["sample_dt"] = pd.to_datetime(raw["sample_dt"], errors="coerce")
    south = raw[
        (raw["lat"] < CAUSEWAY_LAT)
        & (raw["sample_depth(ft)"] <= UPPER_BRINE_MAX_FT)
        & raw["density(g/cm3)"].between(1.0, 1.30)
        & raw["sample_dt"].notna()
    ].copy()
    south["salinity_gl"] = slope * south["density(g/cm3)"] + intercept
    south["month"] = south["sample_dt"].values.astype("datetime64[M]")
    return (
        south.groupby("month", as_index=False)
        .agg(independent_gl=("salinity_gl", "median"), n_samples=("salinity_gl", "size"))
        .sort_values("month")
    )


def compare(db_path: str) -> pd.DataFrame:
    """Compare reconstructed salinity with the external record by month."""
    with duckdb.connect(db_path, read_only=True) as conn:
        samples = conn.execute("SELECT * FROM gsl_brine_samples").fetchdf()
        reconstructed = conn.execute("""
            SELECT c.month, c.salt_mass_mt, h.volume_kaf
            FROM monthly_covariates c
            LEFT JOIN monthly_elevation e USING (month)
            LEFT JOIN gsl_hypsometry h
                ON h.elev_ft_ngvd29 = ROUND(e.elevation_eom_ft::DECIMAL(10, 1), 1)
            WHERE c.salt_mass_mt IS NOT NULL
        """).fetchdf()
    slope, intercept, rmse = density_to_salinity(samples)
    logging.info(
        f"density to salinity: {slope:.1f} * density {intercept:+.1f}, rmse {rmse:.2f} g/L"
    )
    reconstructed["month"] = pd.to_datetime(reconstructed["month"])
    reconstructed["reconstructed_gl"] = (
        reconstructed["salt_mass_mt"] / reconstructed["volume_kaf"] / KAF_GL_TO_MT
    )
    independent = fetch_independent_salinity(slope, intercept)
    joined = reconstructed.merge(independent, on="month", how="inner").dropna(
        subset=["reconstructed_gl", "independent_gl"]
    )
    joined["difference_gl"] = joined["reconstructed_gl"] - joined["independent_gl"]
    return joined


def render(db_path: str) -> str:
    joined = compare(db_path)
    if joined.empty:
        return "No overlapping months between the reconstruction and the external record."
    difference = joined["difference_gl"]
    correlation = joined["reconstructed_gl"].corr(joined["independent_gl"])
    return "\n".join(
        [
            "Salinity cross-check against the HydroShare open-water compilation",
            "",
            f"Overlapping months        {len(joined)} "
            f"({joined['month'].min():%Y-%m} to {joined['month'].max():%Y-%m})",
            f"Correlation               {correlation:.3f}",
            f"Mean difference           {difference.mean():+.1f} g/L",
            f"Mean absolute difference  {difference.abs().mean():.1f} g/L",
            f"External series spread    {joined['independent_gl'].std():.1f} g/L",
            "",
            "Mean difference summarizes bias and correlation summarizes co-variation.",
            "Neither statistic validates individual months or establishes independence.",
        ]
    )
