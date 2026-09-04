"""USGS daily values from the Water Data OGC API (the WaterServices replacement).

One page holds up to 50,000 features; longer records are walked by cursor. Values are
provisional for months after collection and can be revised, so callers re-fetch a trailing
window and upsert rather than insert-ignore.
"""

import logging
import os
import time
from datetime import date, timedelta
from urllib.parse import parse_qs, urlparse

import duckdb
import pandas as pd
import requests

from src.pipeline.quality import normalize_flags

USGS_DAILY = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
PAGE = 50000
REFETCH_DAYS = 45


def get_with_retry(url: str, params: dict | None = None, timeout: int = 300, tries: int = 4):
    """USGS and AWDB both return transient 5xx; back off and retry before giving up."""
    for attempt in range(tries):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code < 500:
            resp.raise_for_status()
            return resp
        wait = 10 * 2**attempt
        logging.warning(f"{resp.status_code} from {url[:80]}; retrying in {wait}s")
        time.sleep(wait)
    resp.raise_for_status()
    return resp


def fetch_usgs_daily(
    site: str, parameter: str, start: str, end: str | None = None, url: str = USGS_DAILY
) -> list[tuple[str, float, str]]:
    """(date, value, flags) for one site and parameter code, daily mean statistic.

    Flags are the normalized approval status followed by any qualifier codes. USGS has used
    two vocabularies, so `normalize_flags` maps both to one form.
    """
    params = {
        "monitoring_location_id": f"USGS-{site}",
        "parameter_code": parameter,
        "statistic_id": "00003",
        "datetime": f"{start}/{end or '..'}",
        "limit": PAGE,
        "f": "json",
    }
    if key := os.environ.get("USGS_API_KEY"):
        params["api_key"] = key
    rows = []
    cursor = None
    while True:
        query = {**params, "cursor": cursor} if cursor else params
        body = get_with_retry(url, params=query).json()
        for feature in body.get("features", []):
            p = feature["properties"]
            try:
                value = float(p["value"])
            except (TypeError, ValueError):
                continue
            flags = normalize_flags(p.get("approval_status"), p.get("qualifier"))
            rows.append((p["time"], value, flags))
        links = body.get("links", [])
        nxt = next((link["href"] for link in links if link.get("rel") == "next"), None)
        if not nxt or not body.get("features"):
            return rows
        cursor = parse_qs(urlparse(nxt).query)["cursor"][0]


def upsert(conn: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame) -> None:
    """Bulk INSERT OR REPLACE from a DataFrame; executemany is far too slow for daily data."""
    if frame.empty:
        return
    conn.register("_upsert", frame)
    conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _upsert")
    conn.unregister("_upsert")


def ingest_elevation(
    conn: duckdb.DuckDBPyConnection, table: str, site: str, parameter: str, start: str
) -> None:
    """Daily mean elevation for one gauge into a (d, elevation, qualifiers) table.

    Incremental from the table's max date less a trailing window, so provisional values
    that USGS later revises are replaced.
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            d DATE PRIMARY KEY,
            elevation FLOAT,
            qualifiers VARCHAR
        )
    """)
    max_d = conn.execute(f"SELECT MAX(d) FROM {table}").fetchone()[0]
    if max_d is not None:
        start = str(max_d - timedelta(days=REFETCH_DAYS))
    logging.info(f"Fetching USGS {site} daily elevation from {start}")
    rows = fetch_usgs_daily(site, parameter, start, str(date.today()))
    frame = pd.DataFrame(rows, columns=["d", "elevation", "qualifiers"])
    frame["d"] = pd.to_datetime(frame["d"]).dt.date
    upsert(conn, table, frame)
    logging.info(f"Upserted {len(rows)} rows into {table}")
