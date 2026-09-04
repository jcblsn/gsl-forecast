"""Daily weather at the Salt Lake City airport from the NCEI daily-summaries service.

The lake evaporates under the conditions over the lake. The nearest long station that
measures those conditions daily is KSLC, GHCN-D station USW00024127, about 8 km from the
south shore. The nClimDiv columns average a whole climate division over a month and are
released around the 8th of the next month, so they cannot force a monthly balance for the
cutoff month. NCEI publishes a daily summary about 1 day after the day, so the cutoff month
is complete when the forecast runs on the 2nd.

Temperatures arrive in tenths of a degree Celsius, wind in tenths of a metre per second and
precipitation in tenths of a millimetre. The reader converts each one on the way in.
"""

import logging
from datetime import date, timedelta
from io import StringIO

import duckdb
import pandas as pd

from src.pipeline.usgs import get_with_retry

NCEI_DAILY = "https://www.ncei.noaa.gov/access/services/data/v1"
KSLC_TABLE = "kslc_daily"
DATA_TYPES = ("TMAX", "TMIN", "AWND", "PRCP")
# NCEI revises a recent day, so a trailing window is read again on every run.
REFETCH_DAYS = 45
# Tenths of the stated unit to the unit this project uses.
SCALES = {
    "TMAX": ("tmax_c", 0.1),
    "TMIN": ("tmin_c", 0.1),
    "AWND": ("wind_mps", 0.1),
    "PRCP": ("prcp_in", 0.1 / 25.4),
}


def fetch_ncei_daily(station: str, start: str, end: str) -> pd.DataFrame:
    """Daily summaries for one station as a frame of converted columns."""
    resp = get_with_retry(
        NCEI_DAILY,
        params={
            "dataset": "daily-summaries",
            "stations": station,
            "startDate": start,
            "endDate": end,
            "dataTypes": ",".join(DATA_TYPES),
            "format": "csv",
        },
        timeout=300,
    )
    if not resp.text.strip():
        return pd.DataFrame(columns=["d", *(c for c, _ in SCALES.values())])
    raw = pd.read_csv(StringIO(resp.text))
    frame = pd.DataFrame({"d": pd.to_datetime(raw["DATE"]).dt.date})
    for code, (column, scale) in SCALES.items():
        values = pd.to_numeric(raw[code], errors="coerce") if code in raw else pd.NA
        frame[column] = values * scale if code in raw else pd.NA
    return frame


def ingest_kslc(conn: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {KSLC_TABLE} (
            d DATE PRIMARY KEY,
            tmax_c DOUBLE, tmin_c DOUBLE, wind_mps DOUBLE, prcp_in DOUBLE
        )
    """)
    max_d = conn.execute(f"SELECT MAX(d) FROM {KSLC_TABLE}").fetchone()[0]
    start = cfg["start"] if max_d is None else str(max_d - timedelta(days=REFETCH_DAYS))
    frame = fetch_ncei_daily(cfg["station"], start, str(date.today()))
    if frame.empty:
        logging.warning(f"NCEI returned no rows for {cfg['station']} from {start}")
        return
    conn.register("_kslc", frame)
    conn.execute(f"INSERT OR REPLACE INTO {KSLC_TABLE} SELECT * FROM _kslc")
    conn.unregister("_kslc")
    logging.info(f"NCEI {cfg['station']}: upserted {len(frame)} daily rows from {start}")
