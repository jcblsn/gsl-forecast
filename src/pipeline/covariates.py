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
import numpy as np
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
# A month of discharge needs this many daily values. The sum is then scaled to the whole
# month, which assumes the missing days flowed like the days that reported.
MIN_FLOW_DAYS = 25
ELEVATION_PARAMETER = "62614"
NORTH_ARM_TABLE = "usgs_north_arm_elevation_daily"
SNOTEL_ELEMENTS = ("WTEQ", "PREC", "SMS:-8")
# The month-end snow input is the last valid value in the final days of the month, not the
# value on the last day. A site that misses the last day still reports a month-end value.
MONTH_END_WINDOW_DAYS = 5
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
    states: list[str],
    network: str,
    basins: dict[str, str],
    elements: str | None = None,
    station_triplets: list[str] | None = None,
) -> list[dict]:
    """AWDB stations of one network inside the configured hydrologic units.

    A list of station triplets asks AWDB for those stations, active or retired. Without
    that list the function asks for the stations that are active on the day of the run.
    """
    sites = []
    queries = (
        [",".join(station_triplets)]
        if station_triplets
        else [f"*:{state}:{network}" for state in states]
    )
    for query in queries:
        params = {
            "stationTriplets": query,
            "activeOnly": "false" if station_triplets else "true",
        }
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


def roster_stations(cfg: dict) -> list[str]:
    """The station triplets the configured roster names, or an empty list."""
    configured = cfg.get("roster")
    if not configured:
        return []
    return [t for stations in configured["stations"].values() for t in stations]


def create_snotel_roster(conn: duckdb.DuckDBPyConnection, sites: list[dict], cfg: dict) -> None:
    """Write `snotel_roster`: the versioned set of sites the snow features use.

    A roster in the configuration names the sites and the basin weight of each basin.
    The roster makes the features independent of which sites AWDB reports as active on
    the day of the run, and of which sites an earlier run left in `snotel_sites`.

    Without a configured roster the function falls back to the sites discovered today and
    gives each basin the same weight. That fallback is not stable over time. Use it for a
    first run or a test only.
    """
    found = {site["station_triplet"]: site["basin"] for site in sites}
    configured = cfg.get("roster")
    if configured:
        version = configured["version"]
        station_basins = {
            triplet: basin
            for basin, triplets in configured["stations"].items()
            for triplet in triplets
        }
        missing = sorted(set(station_basins) - set(found))
        if missing:
            raise ValueError(f"AWDB did not return these roster stations: {missing}")
        crossed = sorted(t for t, b in station_basins.items() if found[t] != b)
        if crossed:
            raise ValueError(f"These roster stations sit in another basin than declared: {crossed}")
        basin_weights = configured["basin_weights"]
    else:
        version = "discovered-active"
        station_basins = found
        basins = sorted(set(found.values()))
        basin_weights = {basin: 1.0 / len(basins) for basin in basins}
    if set(basin_weights) != set(station_basins.values()):
        raise ValueError("The basin weights must name exactly the basins the roster covers")
    if not np.isclose(sum(basin_weights.values()), 1.0):
        raise ValueError("The basin weights must sum to 1")
    roster = pd.DataFrame(
        [
            {
                "roster_version": version,
                "station_triplet": triplet,
                "basin": basin,
                "basin_weight": float(basin_weights[basin]),
            }
            for triplet, basin in sorted(station_basins.items())
        ]
    )
    conn.register("_snotel_roster", roster)
    conn.execute("CREATE OR REPLACE TABLE snotel_roster AS SELECT * FROM _snotel_roster")
    conn.unregister("_snotel_roster")
    logging.info(f"SNOTEL roster {version}: {len(roster)} sites")


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
    retired = sorted(set(roster_stations(cfg)) - {s["station_triplet"] for s in sites})
    if retired:
        sites += fetch_awdb_stations(cfg["states"], "SNTL", cfg["basins"], station_triplets=retired)
    create_snotel_roster(conn, sites, cfg)
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


# The nClimDiv columns that `transform_covariates` keeps out of `monthly_covariates`. NOAA
# releases a month around the 8th of the next month and the workflow runs on the 2nd, so the
# cutoff month has no value at issue time. `tests/test_leakage.py` checks that no model reads
# one of these names.
UNAVAILABLE_AT_ISSUE = ("tavg_f_gsl", "prcp_in_gsl")


def transform_covariates(
    conn: duckdb.DuckDBPyConnection, discharge: dict, basins: list[str]
) -> None:
    """Monthly, complete months only.

    Snowpack comes from the sites in `snotel_roster`. A site's month-end value is its last
    valid value in the final `MONTH_END_WINDOW_DAYS` days of the month, so a site that
    misses the last day still reports. Each variable takes its own last valid day and its
    own count of reporting sites. The columns are mean SWE, mean water-year precipitation,
    both also as percent of the site medians, and 8-inch soil moisture, per basin and
    pooled. Every pooled column averages the basins under the roster's declared basin
    weights.

    Reservoir storage at month end is summed over the reporting stations per basin. That
    roster grows with dam construction, so early sums are smaller for a physical reason.

    The table also holds inflow and breach flow in kaf, north-arm mean elevation, the
    south-minus-north head, and climate-division mean temperature and precipitation. NOAA
    releases a climate month around the 8th of the next month, so the cutoff month has no
    climate value at issue time. Therefore this table holds only the `_lag1` copies of the
    climate columns. The unlagged values stay in `climdiv_monthly`, where no model reads
    them, so a model cannot use a value that does not exist at issue time."""
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
    inflow_sites = ", ".join(f"'{site}'" for site in inflow.values())
    per_basin = {
        "swe_eom": "swe",
        "prec_wy_eom": "prec",
        "swe_pct_median": "swe_pct",
        "prec_pct_median": "prec_pct",
        "sms_eom": "sms",
    }
    basin_cols = ",\n".join(
        f"MAX(CASE WHEN basin = '{b}' THEN {expr} END) AS {name}_{b}"
        for name, expr in per_basin.items()
        for b in basins
    )
    # Every pooled column is the basin average under the roster's declared basin weights,
    # over the basins that report that column that month. Weighting by the site count
    # instead would let the basin with the most sites decide the index, and would apply the
    # count of reporting SWE sites to the precipitation and soil-moisture averages as well.
    pooled_cols = ",\n".join(
        f"SUM({expr} * w) FILTER ({expr} IS NOT NULL)"
        f" / NULLIF(SUM(w) FILTER ({expr} IS NOT NULL), 0) AS {name}_gsl"
        for name, expr in per_basin.items()
    )
    res_cols = ",\n".join(
        f"MAX(CASE WHEN basin = '{b}' THEN kaf END) AS res_kaf_{b}" for b in basins
    )
    conn.execute(f"""
        CREATE OR REPLACE TABLE monthly_covariates AS
        WITH window_days AS (
            SELECT s.basin, s.basin_weight, s.roster_version,
                   DATE_TRUNC('month', d.d) AS month, d.*
            FROM snotel_daily d JOIN snotel_roster s USING (station_triplet)
            WHERE d.d > LAST_DAY(d.d) - INTERVAL {MONTH_END_WINDOW_DAYS} DAY
        ),
        eom AS (
            SELECT month, station_triplet, basin, basin_weight, roster_version,
                   ARG_MAX(wteq_in, d) AS wteq_in,
                   ARG_MAX(wteq_median_in, d) AS wteq_median_in,
                   ARG_MAX(prec_in, d) AS prec_in,
                   ARG_MAX(prec_median_in, d) AS prec_median_in,
                   ARG_MAX(sms_8_pct, d) AS sms_8_pct
            FROM window_days GROUP BY ALL
        ),
        snow AS (
            SELECT month, basin, roster_version, ANY_VALUE(basin_weight) AS w,
                   COUNT(wteq_in) AS n_swe,
                   COUNT(prec_in) AS n_prec,
                   COUNT(sms_8_pct) AS n_sms,
                   AVG(wteq_in) AS swe, AVG(prec_in) AS prec, AVG(sms_8_pct) AS sms,
                   100 * SUM(wteq_in) FILTER (wteq_median_in IS NOT NULL)
                       / NULLIF(SUM(wteq_median_in) FILTER (wteq_in IS NOT NULL), 0)
                       AS swe_pct,
                   100 * SUM(prec_in) FILTER (prec_median_in IS NOT NULL)
                       / NULLIF(SUM(prec_median_in) FILTER (prec_in IS NOT NULL), 0)
                       AS prec_pct
            FROM eom GROUP BY ALL
        ),
        snow_wide AS (
            SELECT month,
                   {basin_cols},
                   {pooled_cols},
                   SUM(n_swe) AS n_snotel_sites,
                   SUM(n_prec) AS n_snotel_prec,
                   SUM(n_sms) AS n_snotel_sms,
                   ANY_VALUE(roster_version) AS snotel_roster_version
            FROM snow GROUP BY month
        ),
        flow_days AS (
            SELECT DATE_TRUNC('month', d) AS month, site_id,
                   SUM(discharge_cfs) AS cfs_days, COUNT(*) AS n_days,
                   DAY(LAST_DAY(DATE_TRUNC('month', d))) AS month_days,
                   COUNT(*) FILTER (
                       LOWER(COALESCE(qualifiers, '')) LIKE '%provisional%'
                   ) AS n_provisional,
                   COUNT(*) FILTER (
                       LOWER(COALESCE(qualifiers, '')) LIKE '%estimated%'
                   ) AS n_estimated
            FROM usgs_discharge_daily
            GROUP BY ALL
        ),
        flow AS (
            SELECT month, site_id,
                   cfs_days / n_days * month_days * 86400.0 / 43560.0 / 1000.0 AS kaf,
                   n_days::DOUBLE / month_days AS day_coverage,
                   n_provisional, n_estimated
            FROM flow_days WHERE n_days >= {MIN_FLOW_DAYS}
        ),
        flow_wide AS (
            SELECT month,
                   {flow_cols},
                   MIN(day_coverage) FILTER (site_id IN ({inflow_sites}))
                       AS inflow_day_coverage,
                   SUM(n_provisional) FILTER (site_id IN ({inflow_sites}))
                       AS inflow_provisional_days,
                   SUM(n_estimated) FILTER (site_id IN ({inflow_sites}))
                       AS inflow_estimated_days
            FROM flow GROUP BY month
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
        ),
        climate_lag AS (
            SELECT month + INTERVAL 1 MONTH AS month,
                   tavg_f_gsl AS tavg_f_gsl_lag1, prcp_in_gsl AS prcp_in_gsl_lag1
            FROM climate
        )
        SELECT month, s.* EXCLUDE (month), f.* EXCLUDE (month),
               {total} AS inflow_kaf_total,
               r.* EXCLUDE (month),
               n.north_arm_ft, e.avg_elevation - n.north_arm_ft AS head_diff_ft,
               cl.* EXCLUDE (month)
        FROM snow_wide s
        FULL OUTER JOIN flow_wide f USING (month)
        FULL OUTER JOIN res_wide r USING (month)
        FULL OUTER JOIN north n USING (month)
        FULL OUTER JOIN climate c USING (month)
        LEFT JOIN climate_lag cl USING (month)
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
