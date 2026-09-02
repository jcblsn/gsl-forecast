"""Monthly climate-division temperature and precipitation from NOAA nClimDiv.

One fixed-width file per element covers every division since 1895. The file name carries
the release date, so the directory listing is read to find the current one, and the table
is replaced on every run since each release revises the last two years. Values for a month
are released around the 8th of the following month, one week after the forecast runs, so
the cutoff month is always missing at issue time: these columns serve the fit of the water
balance, not the in-season regressors.
"""

import logging
import re

import pandas as pd

from src.pipeline.usgs import get_with_retry

CLIMDIV = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"
ELEMENTS = {"tmpc": ("tavg_f", -99.9), "pcpn": ("prcp_in", -9.99)}


def latest_file(listing: str, element: str) -> str:
    names = sorted(set(re.findall(rf"climdiv-{element}dv-v[\d.]+-\d{{8}}", listing)))
    if not names:
        raise RuntimeError(f"No climdiv-{element}dv file in the nClimDiv listing")
    return names[-1]


def parse_climdiv(
    text: str, state: str, divisions: list[str], missing: float
) -> list[tuple[str, str, float]]:
    """(month, division, value) rows for the wanted state and divisions; the missing code
    differs by element (-99.90 for temperature, -9.99 for precipitation)."""
    rows = []
    for line in text.splitlines():
        if line[:2] != state or line[2:4] not in divisions:
            continue
        year = line[6:10]
        for m, value in enumerate(line[10:].split(), start=1):
            if float(value) > missing:
                rows.append((f"{year}-{m:02d}-01", line[2:4], float(value)))
    return rows


def ingest_climdiv(conn, cfg: dict) -> None:
    listing = get_with_retry(CLIMDIV, timeout=120).text
    frames = []
    for element, (column, missing) in ELEMENTS.items():
        name = latest_file(listing, element)
        text = get_with_retry(CLIMDIV + name, timeout=300).text
        rows = parse_climdiv(text, cfg["state"], cfg["divisions"], missing)
        frames.append(pd.DataFrame(rows, columns=["month", "division", column]))
        logging.info(f"{name}: {len(rows)} rows for divisions {cfg['divisions']}")
    frame = frames[0].merge(frames[1], on=["month", "division"], how="outer")
    frame["month"] = pd.to_datetime(frame["month"]).dt.date
    conn.register("_climdiv", frame)
    conn.execute("""
        CREATE OR REPLACE TABLE climdiv_monthly AS
        SELECT CAST(month AS DATE) AS month, division, CAST(tavg_f AS FLOAT) AS tavg_f,
               CAST(prcp_in AS FLOAT) AS prcp_in
        FROM _climdiv ORDER BY month, division
    """)
    conn.unregister("_climdiv")
