"""South-arm brine chemistry from the Utah Geological Survey.

UGS has sampled Great Salt Lake brine since 1966 and publishes the whole record as one
workbook with a sheet per site. Site AS2 is the open-water Gilbert Bay site that Utah's own
lake summary plots, and it is the only site on the south transect that carries a computed
salinity for the full record.

Sampling is 1 to 4 boat campaigns per year, so this is not a monthly series and it cannot
resolve a seasonal cycle. It resolves what it is used for here: the dissolved salt mass,
which moves over years, not months.

Salt mass is not conserved. Mineral extraction removes salt and the causeway exports it to
the north arm. Estimated from this record it falls from about 1685 Mt in 1993 to about 863
Mt in 2025. A model that holds salt mass constant therefore gets the recent lake wrong.

The estimate multiplies an upper-brine salinity by the whole south-arm volume. When a deep
brine layer is present the true mass is higher, so the level carries a low bias. The
trajectory is the part this project uses, and the level is reported so a reader can see it.
"""

import logging
import os
from datetime import date
from io import BytesIO
from wsgiref.handlers import format_date_time

import duckdb
import numpy as np
import pandas as pd
import requests

from src.forecasting import hypsometry
from src.pipeline.usgs import get_with_retry

UGS_WORKBOOK = "https://geology.utah.gov/docs/xls/GSL_brine_chem_db.xlsx"
SAMPLES_TABLE = "gsl_brine_samples"
# The sample depths that represent the upper brine layer. Below this a deep brine layer can
# sit under the south arm, and its salinity is not the salinity the lake evaporates at.
UPPER_BRINE_MAX_FT = 15.0
# 1 kaf of brine at 1 g/L holds this many million tonnes of salt.
KAF_GL_TO_MT = 1.2335e9 * 1e-6 / 1e6
COLUMNS = {
    "SITE": "site",
    "DATE": "sample_date",
    "DEPTH-FT": "depth_ft",
    "Salinity EOS (g/L)": "salinity_gl",
    "LAB-DEN (g/cm3)": "density_g_cm3",
    "TDS (g/L)": "tds_gl",
    "LK-ELEV (feet)": "lake_elev_ft",
}


def _normalize(name: str) -> str:
    return " ".join(str(name).split())


def read_workbook(content: bytes, sites: list[str]) -> pd.DataFrame:
    """Campaign rows for the named site sheets, one row per sample and depth."""
    frames = []
    for site in sites:
        raw = pd.read_excel(BytesIO(content), sheet_name=site)
        raw.columns = [_normalize(c) for c in raw.columns]
        frame = pd.DataFrame(index=raw.index)
        for source, target in COLUMNS.items():
            frame[target] = raw[source] if source in raw else np.nan
        frame["site"] = site
        frame["sample_date"] = pd.to_datetime(frame["sample_date"], errors="coerce")
        for column in ("depth_ft", "salinity_gl", "density_g_cm3", "tds_gl", "lake_elev_ft"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame.dropna(subset=["sample_date"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def upper_brine_campaigns(samples: pd.DataFrame, site: str) -> pd.DataFrame:
    """One salinity per campaign at one site: the mean over the upper-brine depths."""
    upper = samples[
        (samples["site"] == site)
        & (samples["depth_ft"] <= UPPER_BRINE_MAX_FT)
        & samples["salinity_gl"].notna()
    ]
    return (
        upper.groupby("sample_date", as_index=False)["salinity_gl"]
        .mean()
        .sort_values("sample_date")
        .reset_index(drop=True)
    )


def salt_mass_series(campaigns: pd.DataFrame, elevations: pd.DataFrame) -> pd.DataFrame:
    """Salt mass in million tonnes at each campaign, from salinity and lake volume.

    The elevation comes from the project's own gauge and not from the workbook, so the
    volume is the same volume the balance model steps.
    """
    if campaigns.empty or elevations.empty:
        return pd.DataFrame(columns=["sample_date", "salinity_gl", "salt_mass_mt"])
    gauge = elevations.sort_values("d").reset_index(drop=True)
    index = np.searchsorted(gauge["d"].to_numpy(), campaigns["sample_date"].to_numpy())
    index = np.clip(index, 0, len(gauge) - 1)
    elevation = gauge["elevation"].to_numpy()[index]
    out = campaigns.copy()
    out["lake_elev_ft"] = elevation
    out["volume_kaf"] = hypsometry.volume_kaf(elevation, strict=False)
    out["salt_mass_mt"] = out["salinity_gl"] * out["volume_kaf"] * KAF_GL_TO_MT
    return out


def monthly_salt_mass(salt: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate the campaign salt mass onto months, and say how old each value is.

    Interpolating between 2 campaigns a year would invent a seasonal cycle in a salinity,
    which the sampling cannot support. It does not invent one in a salt mass, because salt
    mass has no seasonal cycle to invent: it moves with extraction and causeway export.
    Salinity is therefore recovered later, from this mass and the volume of the month.

    After the last campaign the value is carried forward, because the forecast runs every
    month and a campaign can be 6 months old at issue time. A carried value is an assumption
    that the mass has not moved, so `salt_mass_age_days` reports how long it has been
    carried and a reader can judge it. Before the first campaign there is nothing to carry
    and the value stays null.
    """
    empty = pd.DataFrame({"month": months, "salt_mass_mt": np.nan, "salt_mass_age_days": np.nan})
    if salt.empty:
        return empty
    known = salt.dropna(subset=["salt_mass_mt"]).sort_values("sample_date")
    if known.empty:
        return empty
    dates = known["sample_date"].astype("int64").to_numpy()
    values = np.interp(
        months.astype("int64"),
        dates,
        known["salt_mass_mt"].to_numpy(),
        left=np.nan,
        right=known["salt_mass_mt"].to_numpy()[-1],
    )
    last = known["sample_date"].max()
    age = (months - last).days.to_numpy().astype(float)
    age[age < 0] = 0.0
    age[np.isnan(values)] = np.nan
    return pd.DataFrame({"month": months, "salt_mass_mt": values, "salt_mass_age_days": age})


def fetch_workbook(cache_path: str | None) -> bytes:
    """The workbook bytes, from the cache when the published file has not changed.

    UGS revises this file about twice a year and it is 4 MB. A conditional request keeps the
    monthly pipeline run from downloading it again for nothing.
    """
    if not cache_path or not os.path.exists(cache_path):
        content = get_with_retry(UGS_WORKBOOK, timeout=300).content
    else:
        stamp = format_date_time(os.path.getmtime(cache_path))
        resp = requests.get(UGS_WORKBOOK, headers={"If-Modified-Since": stamp}, timeout=300)
        if resp.status_code == 304:
            logging.info(f"UGS workbook unchanged; reading {cache_path}")
            with open(cache_path, "rb") as handle:
                return handle.read()
        resp.raise_for_status()
        content = resp.content
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as handle:
            handle.write(content)
    return content


def ingest_brine(conn: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    """Fetch the workbook, land the samples, and write the monthly salt-mass trajectory."""
    sites = [cfg["primary_site"], *cfg.get("cross_check_sites", [])]
    content = fetch_workbook(cfg.get("cache_path"))
    samples = read_workbook(content, sites)
    conn.register("_brine", samples)
    conn.execute(f"""
        CREATE OR REPLACE TABLE {SAMPLES_TABLE} AS
        SELECT site, CAST(sample_date AS DATE) AS sample_date, depth_ft, salinity_gl,
               density_g_cm3, tds_gl, lake_elev_ft
        FROM _brine ORDER BY site, sample_date, depth_ft
    """)
    conn.unregister("_brine")
    logging.info(f"UGS brine: {len(samples)} samples across {sites}")


def build_salt_mass(conn: duckdb.DuckDBPyConnection, cfg: dict, elevation_table: str) -> None:
    """Write `gsl_salt_mass_monthly` from the primary site's upper-brine campaigns."""
    samples = conn.execute(f"SELECT * FROM {SAMPLES_TABLE}").fetchdf()
    samples["sample_date"] = pd.to_datetime(samples["sample_date"])
    elevations = conn.execute(f"SELECT d, elevation FROM {elevation_table}").fetchdf()
    elevations["d"] = pd.to_datetime(elevations["d"])
    campaigns = upper_brine_campaigns(samples, cfg["primary_site"])
    salt = salt_mass_series(campaigns, elevations)
    first = pd.Timestamp(cfg.get("start", "1966-01-01")).to_period("M").to_timestamp()
    last = pd.Timestamp(date.today()).to_period("M").to_timestamp()
    months = pd.date_range(first, last, freq="MS")
    monthly = monthly_salt_mass(salt, months)
    conn.register("_salt", monthly)
    conn.execute("""
        CREATE OR REPLACE TABLE gsl_salt_mass_monthly AS
        SELECT CAST(month AS DATE) AS month, salt_mass_mt, salt_mass_age_days
        FROM _salt ORDER BY month
    """)
    conn.unregister("_salt")
    known = monthly["salt_mass_mt"].notna().sum()
    logging.info(f"Salt mass: {len(campaigns)} campaigns interpolated onto {known} months")


def materialize_hypsometry(conn: duckdb.DuckDBPyConnection) -> None:
    """Put the elevation-area-volume table in the database.

    A salinity is a salt mass over a volume, and the volume comes from this table. Keeping
    a copy in the database lets the monthly rollup do the lookup in SQL. The table steps by
    0.1 ft, so a join on the rounded elevation is accurate to 0.05 ft.
    """
    frame = hypsometry.table()
    conn.register("_hyps", frame)
    conn.execute("""
        CREATE OR REPLACE TABLE gsl_hypsometry AS
        SELECT CAST(elev_ft_ngvd29 AS DECIMAL(10, 1)) AS elev_ft_ngvd29,
               area_km2, volume_kaf
        FROM _hyps ORDER BY elev_ft_ngvd29
    """)
    conn.unregister("_hyps")
