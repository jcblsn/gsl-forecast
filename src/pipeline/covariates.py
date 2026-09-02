"""Covariates for the lake forecast: SNOTEL snowpack, USGS tributary discharge, the
causeway breach flow and the north-arm elevation.

SNOTEL sites are discovered from the NRCS AWDB station list by hydrologic unit and labelled
by basin (Bear, Weber, Provo-Jordan). Inflow comes from the three gauges the Strike Team
uses. Everything lands in DuckDB next to the elevation tables and is rolled up to a
`monthly_covariates` table aligned with `monthly_elevation`.
"""

import logging
from datetime import date, timedelta

import duckdb
import pandas as pd

from src.pipeline.climate import ingest_climdiv
from src.pipeline.usgs import (
    REFETCH_DAYS,
    fetch_usgs_daily,
    get_with_retry,
    ingest_elevation,
    upsert,
)

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
DISCHARGE_PARAMETER = "00060"
ELEVATION_PARAMETER = "62614"
NORTH_ARM_TABLE = "usgs_north_arm_elevation_daily"
SNOTEL_ELEMENTS = ("WTEQ", "PREC", "SMS:-8")
SNOTEL_COLUMNS = (
    "wteq_in",
    "prec_in",
    "sms_8_pct",
    "wteq_median_in",
    "prec_median_in",
)


def basin_for_huc(huc: str, basins: dict[str, str]) -> str | None:
    for prefix, basin in basins.items():
        if huc.startswith(prefix):
            return basin
    return None


def fetch_awdb_stations(
    states: list[str], network: str, basins: dict[str, str], elements: str | None = None
) -> list[dict]:
    """Active AWDB stations of one network (SNTL, BOR) inside the configured hydrologic units."""
    sites = []
    for state in states:
        params = {"stationTriplets": f"*:{state}:{network}", "activeOnly": "true"}
        if elements:
            params["elements"] = elements
        resp = get_with_retry(f"{AWDB}/stations", params=params, timeout=120)
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
    """Daily SWE, water-year precipitation and 8-inch soil moisture per site, with the
    1991-2020 median of SWE and precipitation where AWDB has one (young sites have none)."""
    resp = get_with_retry(
        f"{AWDB}/data",
        params={
            "stationTriplets": ",".join(triplets),
            "elements": ",".join(SNOTEL_ELEMENTS),
            "duration": "DAILY",
            "beginDate": start,
            "endDate": end,
            "centralTendencyType": "MEDIAN",
        },
        timeout=300,
    )
    rows: dict[tuple[str, str], dict[str, float | None]] = {}
    for station in resp.json():
        triplet = station["stationTriplet"]
        for series in station.get("data", []):
            element = series["stationElement"]["elementCode"]
            for v in series.get("values", []):
                vals = rows.setdefault((triplet, v["date"]), dict.fromkeys(SNOTEL_COLUMNS))
                if element == "WTEQ":
                    vals["wteq_in"], vals["wteq_median_in"] = v.get("value"), v.get("median")
                elif element == "PREC":
                    vals["prec_in"], vals["prec_median_in"] = v.get("value"), v.get("median")
                elif element == "SMS":
                    vals["sms_8_pct"] = v.get("value")
    return [(t, d, *(vals[c] for c in SNOTEL_COLUMNS)) for (t, d), vals in rows.items()]


def ingest_snotel(conn: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snotel_sites (
            station_triplet VARCHAR PRIMARY KEY,
            name VARCHAR, basin VARCHAR, huc VARCHAR, elevation_ft FLOAT,
            latitude FLOAT, longitude FLOAT, begin_date DATE
        )
    """)
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    if "snotel_daily" in tables:
        cols = {r[0] for r in conn.execute("DESCRIBE snotel_daily").fetchall()}
        if not set(SNOTEL_COLUMNS) <= cols:
            logging.warning("snotel_daily has an old schema; rebuilding it from the source")
            conn.execute("DROP TABLE snotel_daily")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS snotel_daily (
            station_triplet VARCHAR, d DATE, {" FLOAT, ".join(SNOTEL_COLUMNS)} FLOAT,
            PRIMARY KEY (station_triplet, d)
        )
    """)
    sites = fetch_awdb_stations(cfg["states"], "SNTL", cfg["basins"])
    conn.executemany(
        "INSERT OR REPLACE INTO snotel_sites VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS DATE))",
        [tuple(s.values()) for s in sites],
    )
    logging.info(f"{len(sites)} SNOTEL sites in the GSL basins")

    end = str(date.today())
    triplets = [s["station_triplet"] for s in sites]
    total = 0
    for i, triplet in enumerate(triplets, start=1):
        max_d = conn.execute(
            "SELECT MAX(d) FROM snotel_daily WHERE station_triplet = ?", [triplet]
        ).fetchone()[0]
        start = cfg["start"] if max_d is None else str(max_d - timedelta(days=7))
        rows = fetch_snotel_daily([triplet], start, end)
        frame = pd.DataFrame(rows, columns=["station_triplet", "d", *SNOTEL_COLUMNS])
        frame["d"] = pd.to_datetime(frame["d"]).dt.date
        upsert(conn, "snotel_daily", frame)
        total += len(rows)
        logging.info(f"SNOTEL {i}/{len(triplets)} {triplet}: {len(rows)} rows from {start}")
    logging.info(f"Upserted {total} SNOTEL daily rows")


def fetch_reservoir_monthly(triplets: list[str], start: str) -> list[tuple[str, str, float]]:
    """End-of-month reservoir storage (RESC, acre-feet) per station, in kaf."""
    resp = get_with_retry(
        f"{AWDB}/data",
        params={
            "stationTriplets": ",".join(triplets),
            "elements": "RESC",
            "duration": "MONTHLY",
            "beginDate": start,
            "endDate": str(date.today()),
        },
        timeout=300,
    )
    rows = []
    for station in resp.json():
        for series in station.get("data", []):
            for v in series.get("values", []):
                if v.get("value") is not None:
                    month = f"{v['year']}-{v['month']:02d}-01"
                    rows.append((station["stationTriplet"], month, v["value"] / 1000.0))
    return rows


def ingest_reservoirs(conn: duckdb.DuckDBPyConnection, cfg: dict, basins: dict[str, str]) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reservoir_sites (
            station_triplet VARCHAR PRIMARY KEY,
            name VARCHAR, basin VARCHAR, huc VARCHAR, elevation_ft FLOAT,
            latitude FLOAT, longitude FLOAT, begin_date DATE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reservoir_monthly (
            station_triplet VARCHAR, month DATE, storage_kaf FLOAT,
            PRIMARY KEY (station_triplet, month)
        )
    """)
    sites = fetch_awdb_stations(cfg["states"], "BOR", basins, elements="RESC")
    conn.executemany(
        "INSERT OR REPLACE INTO reservoir_sites VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS DATE))",
        [tuple(s.values()) for s in sites],
    )
    max_month = conn.execute("SELECT MAX(month) FROM reservoir_monthly").fetchone()[0]
    start = cfg["start"] if max_month is None else str(max_month - timedelta(days=95))
    rows = fetch_reservoir_monthly([s["station_triplet"] for s in sites], start)
    frame = pd.DataFrame(rows, columns=["station_triplet", "month", "storage_kaf"])
    frame["month"] = pd.to_datetime(frame["month"]).dt.date
    upsert(conn, "reservoir_monthly", frame)
    logging.info(f"{len(sites)} reservoirs; upserted {len(rows)} monthly storage rows from {start}")


def fetch_nrcs_forecasts(station: str, start: str) -> list[tuple]:
    """Every published seasonal inflow forecast for one AWDB forecast point: publication
    date, period, exceedance percent, kaf, and the period normal."""
    resp = get_with_retry(
        f"{AWDB}/forecasts",
        params={
            "stationTriplets": station,
            "elementCodes": "SRVO",
            "beginPublicationDate": start,
            "endPublicationDate": str(date.today()),
        },
        timeout=120,
    )
    rows = []
    for point in resp.json():
        for f in point.get("data", []):
            start_md, end_md = f["forecastPeriod"]
            for pct, kaf in f.get("forecastValues", {}).items():
                rows.append(
                    (
                        f["publicationDate"][:10],
                        start_md,
                        end_md,
                        int(pct),
                        kaf,
                        f.get("periodNormal"),
                    )
                )
    return rows


def ingest_nrcs_forecasts(conn: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nrcs_inflow_forecasts (
            publication_date DATE, period_start VARCHAR, period_end VARCHAR,
            exceedance INTEGER, kaf FLOAT, normal_kaf FLOAT,
            PRIMARY KEY (publication_date, period_start, exceedance)
        )
    """)
    rows = fetch_nrcs_forecasts(cfg["station"], cfg["start"])
    frame = pd.DataFrame(
        rows,
        columns=[
            "publication_date",
            "period_start",
            "period_end",
            "exceedance",
            "kaf",
            "normal_kaf",
        ],
    )
    frame["publication_date"] = pd.to_datetime(frame["publication_date"]).dt.date
    upsert(conn, "nrcs_inflow_forecasts", frame)
    logging.info(f"Upserted {len(rows)} NRCS inflow forecast rows for {cfg['station']}")


def ingest_usgs_discharge(conn: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    """Daily mean discharge for the inflow gauges and the causeway breach, one table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usgs_discharge_daily (
            site_id VARCHAR, d DATE, discharge_cfs FLOAT, qualifiers VARCHAR,
            PRIMARY KEY (site_id, d)
        )
    """)
    end = str(date.today())
    for river, site in {**cfg["inflow"], **cfg.get("exchange", {})}.items():
        max_d = conn.execute(
            "SELECT MAX(d) FROM usgs_discharge_daily WHERE site_id = ?", [site]
        ).fetchone()[0]
        start = cfg["start"] if max_d is None else str(max_d - timedelta(days=REFETCH_DAYS))
        rows = fetch_usgs_daily(site, DISCHARGE_PARAMETER, start, end)
        frame = pd.DataFrame(rows, columns=["d", "discharge_cfs", "qualifiers"])
        frame.insert(0, "site_id", site)
        frame["d"] = pd.to_datetime(frame["d"]).dt.date
        upsert(conn, "usgs_discharge_daily", frame)
        logging.info(f"{river} ({site}): upserted {len(rows)} discharge rows from {start}")


def transform_covariates(
    conn: duckdb.DuckDBPyConnection, discharge: dict, basins: list[str]
) -> None:
    """Monthly, complete months only. Snowpack at month end per basin and pooled: mean SWE
    and precipitation, both as percent of the site medians (so the growing site roster does
    not drift the index), and 8-inch soil moisture. Reservoir storage at month end summed
    over the reporting stations per basin (the roster grows with dam construction, so early
    sums are smaller for a physical reason). Inflow and breach flow in kaf, north-arm mean
    elevation, the south-minus-north head, and climate-division mean temperature and
    precipitation (one month behind at issue time)."""
    inflow = discharge["inflow"]
    exchange = discharge.get("exchange", {})
    flow_cols = ",\n".join(
        [
            f"MAX(CASE WHEN site_id = '{site}' THEN kaf END) AS inflow_kaf_{river}"
            for river, site in inflow.items()
        ]
        + [
            f"MAX(CASE WHEN site_id = '{site}' THEN kaf END) AS {name}_kaf"
            for name, site in exchange.items()
        ]
    )
    total = " + ".join(f"inflow_kaf_{river}" for river in inflow)
    per_basin = {
        "swe_eom": "swe",
        "prec_wy_eom": "prec",
        "swe_pct_median": "100 * swe_sum / swe_med",
        "prec_pct_median": "100 * prec_sum / prec_med",
        "sms_eom": "sms",
    }
    basin_cols = ",\n".join(
        f"MAX(CASE WHEN basin = '{b}' THEN {expr} END) AS {name}_{b}"
        for name, expr in per_basin.items()
        for b in basins
    )
    res_cols = ",\n".join(
        f"MAX(CASE WHEN basin = '{b}' THEN kaf END) AS res_kaf_{b}" for b in basins
    )
    conn.execute(f"""
        CREATE OR REPLACE TABLE monthly_covariates AS
        WITH eom AS (
            SELECT s.basin, DATE_TRUNC('month', d.d) AS month, d.*
            FROM snotel_daily d JOIN snotel_sites s USING (station_triplet)
            WHERE d.d = LAST_DAY(d.d) AND d.wteq_in IS NOT NULL
        ),
        snow AS (
            SELECT month, basin, COUNT(*) AS n,
                   AVG(wteq_in) AS swe, AVG(prec_in) AS prec, AVG(sms_8_pct) AS sms,
                   SUM(wteq_in) FILTER (wteq_median_in IS NOT NULL) AS swe_sum,
                   SUM(wteq_median_in) AS swe_med,
                   SUM(prec_in) FILTER (prec_median_in IS NOT NULL) AS prec_sum,
                   SUM(prec_median_in) AS prec_med
            FROM eom GROUP BY ALL
        ),
        snow_wide AS (
            SELECT month,
                   {basin_cols},
                   SUM(swe * n) / SUM(n) AS swe_eom_gsl,
                   SUM(prec * n) / SUM(n) AS prec_wy_eom_gsl,
                   100 * SUM(swe_sum) / SUM(swe_med) AS swe_pct_median_gsl,
                   100 * SUM(prec_sum) / SUM(prec_med) AS prec_pct_median_gsl,
                   SUM(sms * n) / SUM(n) AS sms_eom_gsl,
                   SUM(n) AS n_snotel_sites
            FROM snow GROUP BY month
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
        ),
        res AS (
            SELECT month, basin, SUM(storage_kaf) AS kaf, COUNT(*) AS n
            FROM reservoir_monthly JOIN reservoir_sites USING (station_triplet)
            GROUP BY ALL
        ),
        res_wide AS (
            SELECT month,
                   {res_cols},
                   SUM(kaf) AS res_kaf_total, SUM(n) AS n_reservoirs
            FROM res GROUP BY month
        ),
        north AS (
            SELECT DATE_TRUNC('month', d) AS month, AVG(elevation) AS north_arm_ft
            FROM {NORTH_ARM_TABLE} GROUP BY month
        ),
        climate AS (
            SELECT month, AVG(tavg_f) AS tavg_f_gsl, AVG(prcp_in) AS prcp_in_gsl
            FROM climdiv_monthly GROUP BY month
        )
        SELECT month, s.* EXCLUDE (month), f.* EXCLUDE (month),
               {total} AS inflow_kaf_total,
               r.* EXCLUDE (month),
               n.north_arm_ft, e.avg_elevation - n.north_arm_ft AS head_diff_ft,
               c.* EXCLUDE (month)
        FROM snow_wide s
        FULL OUTER JOIN flow_wide f USING (month)
        FULL OUTER JOIN res_wide r USING (month)
        FULL OUTER JOIN north n USING (month)
        FULL OUTER JOIN climate c USING (month)
        LEFT JOIN monthly_elevation e USING (month)
        WHERE month < DATE_TRUNC('month', CURRENT_DATE)
        ORDER BY month
    """)


def run_covariates(conn: duckdb.DuckDBPyConnection, config: dict) -> None:
    cov = config["covariates"]
    ingest_snotel(conn, cov["snotel"])
    ingest_reservoirs(conn, cov["reservoirs"], cov["snotel"]["basins"])
    ingest_nrcs_forecasts(conn, cov["nrcs_forecasts"])
    ingest_climdiv(conn, cov["climdiv"])
    ingest_usgs_discharge(conn, cov["usgs_discharge"])
    north = cov["north_arm"]
    ingest_elevation(conn, NORTH_ARM_TABLE, north["site"], ELEVATION_PARAMETER, north["start"])
    transform_covariates(conn, cov["usgs_discharge"], list(cov["snotel"]["basins"].values()))
    n = conn.execute("SELECT COUNT(*) FROM monthly_covariates").fetchone()[0]
    logging.info(f"monthly_covariates has {n} rows")
