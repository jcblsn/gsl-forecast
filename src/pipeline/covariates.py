"""Covariates for the lake forecast: SNOTEL snowpack and USGS tributary discharge.

SNOTEL sites are discovered from the NRCS AWDB station list by hydrologic unit and labelled
by basin (Bear, Weber, Provo-Jordan). Discharge comes from the three gauges the Strike Team
uses for lake inflow. Everything lands in DuckDB next to the elevation tables and is rolled
up to a `monthly_covariates` table aligned with `monthly_elevation`.
"""

import logging
from datetime import datetime, timedelta

import duckdb
import requests

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
USGS_DV = (
    "https://waterservices.usgs.gov/nwis/dv?format=json&siteStatus=all&site={site}"
    "&agencyCd=USGS&statCd=00003&parameterCd={param}&startDT={start}&endDT={end}"
)
SNOTEL_ELEMENTS = ("WTEQ", "PREC")


def basin_for_huc(huc: str, basins: dict[str, str]) -> str | None:
    for prefix, basin in basins.items():
        if huc.startswith(prefix):
            return basin
    return None


def fetch_snotel_sites(states: list[str], basins: dict[str, str]) -> list[dict]:
    sites = []
    for state in states:
        resp = requests.get(
            f"{AWDB}/stations",
            params={"stationTriplets": f"*:{state}:SNTL", "activeOnly": "true"},
            timeout=120,
        )
        resp.raise_for_status()
        for s in resp.json():
            basin = basin_for_huc(str(s.get("huc", "")), basins)
            if basin:
                sites.append(
                    {
                        "station_triplet": s["stationTriplet"],
                        "name": s.get("name"),
                        "basin": basin,
                        "huc": s.get("huc"),
                        "elevation_ft": s.get("elevation"),
                        "latitude": s.get("latitude"),
                        "longitude": s.get("longitude"),
                        "begin_date": str(s.get("beginDate", ""))[:10] or None,
                    }
                )
    return sites


def fetch_snotel_daily(triplets: list[str], start: str, end: str) -> list[tuple]:
    resp = requests.get(
        f"{AWDB}/data",
        params={
            "stationTriplets": ",".join(triplets),
            "elements": ",".join(SNOTEL_ELEMENTS),
            "duration": "DAILY",
            "beginDate": start,
            "endDate": end,
        },
        timeout=300,
    )
    resp.raise_for_status()
    rows: dict[tuple[str, str], dict[str, float | None]] = {}
    for station in resp.json():
        triplet = station["stationTriplet"]
        for series in station.get("data", []):
            element = series["stationElement"]["elementCode"]
            for v in series.get("values", []):
                key = (triplet, v["date"])
                rows.setdefault(key, {e: None for e in SNOTEL_ELEMENTS})[element] = v.get("value")
    return [(t, d, vals["WTEQ"], vals["PREC"]) for (t, d), vals in rows.items()]


def ingest_snotel(conn: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snotel_sites (
            station_triplet VARCHAR PRIMARY KEY,
            name VARCHAR, basin VARCHAR, huc VARCHAR, elevation_ft FLOAT,
            latitude FLOAT, longitude FLOAT, begin_date DATE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snotel_daily (
            station_triplet VARCHAR, d DATE, wteq_in FLOAT, prec_in FLOAT,
            PRIMARY KEY (station_triplet, d)
        )
    """)
    sites = fetch_snotel_sites(cfg["states"], cfg["basins"])
    conn.executemany(
        "INSERT OR REPLACE INTO snotel_sites VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS DATE))",
        [tuple(s.values()) for s in sites],
    )
    logging.info(f"{len(sites)} SNOTEL sites in the GSL basins")

    end = datetime.now().strftime("%Y-%m-%d")
    triplets = [s["station_triplet"] for s in sites]
    total = 0
    for i, triplet in enumerate(triplets, start=1):
        max_d = conn.execute(
            "SELECT MAX(d) FROM snotel_daily WHERE station_triplet = ?", [triplet]
        ).fetchone()[0]
        start = cfg["start"] if max_d is None else str(max_d - timedelta(days=7))
        rows = fetch_snotel_daily([triplet], start, end)
        conn.executemany(
            "INSERT OR REPLACE INTO snotel_daily VALUES (?, CAST(? AS DATE), ?, ?)", rows
        )
        total += len(rows)
        logging.info(f"SNOTEL {i}/{len(triplets)} {triplet}: {len(rows)} rows from {start}")
    logging.info(f"Upserted {total} SNOTEL daily rows")


def fetch_usgs_daily(url: str) -> list[tuple[str, float, str]]:
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    series = resp.json().get("value", {}).get("timeSeries", [])
    if not series:
        return []
    rows = []
    for entry in series[0]["values"][0]["value"]:
        d = entry.get("dateTime", "").split("T")[0]
        try:
            value = float(entry.get("value", ""))
        except ValueError:
            continue
        if value <= -999998:
            continue
        q = entry.get("qualifiers", [""])
        rows.append((d, value, q[0] if isinstance(q, list) else str(q)))
    return rows


def ingest_usgs_discharge(conn: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usgs_discharge_daily (
            site_id VARCHAR, d DATE, discharge_cfs FLOAT, qualifiers VARCHAR,
            PRIMARY KEY (site_id, d)
        )
    """)
    end = datetime.now().strftime("%Y-%m-%d")
    for river, site in cfg["sites"].items():
        max_d = conn.execute(
            "SELECT MAX(d) FROM usgs_discharge_daily WHERE site_id = ?", [site]
        ).fetchone()[0]
        start = cfg["start"] if max_d is None else str(max_d)
        url = USGS_DV.format(site=site, param="00060", start=start, end=end)
        rows = fetch_usgs_daily(url)
        conn.executemany(
            "INSERT OR REPLACE INTO usgs_discharge_daily VALUES (?, CAST(? AS DATE), ?, ?)",
            [(site, d, v, q) for d, v, q in rows],
        )
        logging.info(f"{river} ({site}): upserted {len(rows)} discharge rows from {start}")


def transform_covariates(conn: duckdb.DuckDBPyConnection, sites: dict[str, str]) -> None:
    """Monthly, complete months only: month-end basin SWE and precipitation, inflow in kaf."""
    flow_cols = ",\n".join(
        f"MAX(CASE WHEN site_id = '{site}' THEN kaf END) AS inflow_kaf_{river}"
        for river, site in sites.items()
    )
    total = " + ".join(f"inflow_kaf_{river}" for river in sites)
    conn.execute(f"""
        CREATE OR REPLACE TABLE monthly_covariates AS
        WITH eom AS (
            SELECT s.basin, d.station_triplet, DATE_TRUNC('month', d.d) AS month,
                   arg_max(d.wteq_in, d.d) AS wteq_eom, arg_max(d.prec_in, d.d) AS prec_eom,
                   MAX(d.d) AS last_d
            FROM snotel_daily d JOIN snotel_sites s USING (station_triplet)
            WHERE d.wteq_in IS NOT NULL
            GROUP BY ALL
        ),
        swe AS (
            SELECT month, basin, AVG(wteq_eom) AS swe, AVG(prec_eom) AS prec, COUNT(*) AS n
            FROM eom
            WHERE last_d = LAST_DAY(month)
            GROUP BY ALL
        ),
        swe_wide AS (
            SELECT month,
                   MAX(CASE WHEN basin = 'bear' THEN swe END) AS swe_eom_bear,
                   MAX(CASE WHEN basin = 'weber' THEN swe END) AS swe_eom_weber,
                   MAX(CASE WHEN basin = 'provo_jordan' THEN swe END) AS swe_eom_provo_jordan,
                   MAX(CASE WHEN basin = 'bear' THEN prec END) AS prec_wy_eom_bear,
                   MAX(CASE WHEN basin = 'weber' THEN prec END) AS prec_wy_eom_weber,
                   MAX(CASE WHEN basin = 'provo_jordan' THEN prec END) AS prec_wy_eom_provo_jordan,
                   SUM(swe * n) / SUM(n) AS swe_eom_gsl,
                   SUM(prec * n) / SUM(n) AS prec_wy_eom_gsl,
                   SUM(n) AS n_snotel_sites
            FROM swe GROUP BY month
        ),
        flow AS (
            SELECT DATE_TRUNC('month', d) AS month, site_id,
                   SUM(discharge_cfs) * 86400.0 / 43560.0 / 1000.0 AS kaf, COUNT(*) AS n_days
            FROM usgs_discharge_daily
            GROUP BY ALL
        ),
        flow_wide AS (
            SELECT month,
                   {flow_cols}
            FROM flow WHERE n_days >= 25 GROUP BY month
        )
        SELECT COALESCE(s.month, f.month) AS month, s.* EXCLUDE (month), f.* EXCLUDE (month),
               {total} AS inflow_kaf_total
        FROM swe_wide s FULL OUTER JOIN flow_wide f USING (month)
        WHERE COALESCE(s.month, f.month) < DATE_TRUNC('month', CURRENT_DATE)
        ORDER BY month
    """)


def run_covariates(conn: duckdb.DuckDBPyConnection, config: dict) -> None:
    cov = config["covariates"]
    ingest_snotel(conn, cov["snotel"])
    ingest_usgs_discharge(conn, cov["usgs_discharge"])
    transform_covariates(conn, cov["usgs_discharge"]["sites"])
    n = conn.execute("SELECT COUNT(*) FROM monthly_covariates").fetchone()[0]
    logging.info(f"monthly_covariates has {n} rows")
