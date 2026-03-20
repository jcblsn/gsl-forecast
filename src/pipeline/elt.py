import json
import logging
import os
from datetime import datetime
from typing import Optional

import duckdb
import requests


def _load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(base_dir, "config", "config.json")
    with open(config_path) as f:
        return json.load(f)


def ingest_continuous(conn: duckdb.DuckDBPyConnection, source_config: dict) -> None:
    table = "usgs_water_surface_elevation_continuous"

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            agency VARCHAR,
            site VARCHAR,
            dt TIMESTAMP,
            tz VARCHAR,
            elevation FLOAT,
            qualifiers VARCHAR,
            PRIMARY KEY (site, dt)
        )
    """)

    result = conn.execute(f"SELECT MAX(dt) FROM {table}").fetchone()
    max_dt = result[0] if result and result[0] else None

    start_date = datetime(2007, 10, 1) if max_dt is None else max_dt
    end_date = datetime.now()
    site_id = source_config.get("site_id", "10010100")
    url = source_config["url"].format(
        site=site_id,
        start=start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        end=end_date.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    logging.info(f"Fetching continuous USGS data from {start_date.date()} to {end_date.date()}")
    conn.execute("SET force_download=true")
    conn.execute(f"""
        INSERT OR IGNORE INTO {table}
        SELECT
            agency_cd AS agency,
            site_no AS site,
            CAST(datetime || ':00' AS TIMESTAMP) AS dt,
            tz_cd AS tz,
            CAST("144241_62614" AS FLOAT) AS elevation,
            "144241_62614_cd" AS qualifiers
        FROM read_csv_auto(
            '{url}',
            delim='\t',
            skip=1,
            comment='#',
            header=true
        )
        WHERE datetime IS NOT NULL AND datetime != '20d'
    """)


def ingest_daily(conn: duckdb.DuckDBPyConnection, source_config: dict) -> None:
    table = "usgs_water_surface_elevation_daily"

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            d DATE PRIMARY KEY,
            elevation FLOAT,
            qualifiers VARCHAR
        )
    """)

    result = conn.execute(f"SELECT MAX(d) FROM {table}").fetchone()
    max_d = result[0] if result and result[0] else None

    start_date = "1847-10-18" if max_d is None else str(max_d)
    end_date = datetime.now().strftime("%Y-%m-%d")

    site_id = source_config.get("site_id", "10010000")
    url = source_config["url"].format(site=site_id, start=start_date, end=end_date)

    logging.info(f"Fetching daily USGS data from {start_date} to {end_date}")
    response = requests.get(url)
    response.raise_for_status()

    try:
        daily_ts = response.json()["value"]["timeSeries"][0]["values"][0]["value"]
    except (KeyError, IndexError, TypeError) as e:
        logging.warning(f"No daily data found in response: {e}")
        return

    rows = []
    for entry in daily_ts:
        date_str = entry.get("dateTime", "").split("T")[0]
        if not date_str:
            continue
        try:
            elevation_val = float(entry.get("value", ""))
        except ValueError:
            continue
        qualifiers_val = entry.get("qualifiers", [""])[0]
        rows.append((date_str, elevation_val, qualifiers_val))

    if rows:
        conn.executemany(
            f"INSERT OR IGNORE INTO {table} VALUES (CAST(? AS DATE), ?, ?)",
            rows,
        )
        logging.info(f"Inserted {len(rows)} daily records")


def transform(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE OR REPLACE TABLE monthly_elevation AS
        SELECT
            DATE_TRUNC('month', d) AS month,
            AVG(elevation) AS avg_elevation,
            MIN(elevation) AS min_elevation,
            MAX(elevation) AS max_elevation,
            COUNT(*) AS observation_count
        FROM usgs_water_surface_elevation_daily
        GROUP BY month
        ORDER BY month
    """)


def run_pipeline(config_path: Optional[str] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = _load_config(config_path)
    db_path = config["database"]["path"]
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    with duckdb.connect(db_path) as conn:
        try:
            conn.execute("BEGIN TRANSACTION")

            ingest_continuous(conn, config["sources"]["usgs_water_surface_elevation_continuous"])
            ingest_daily(conn, config["sources"]["usgs_water_surface_elevation_daily"])
            transform(conn)

            conn.execute("COMMIT")
            conn.execute("VACUUM")
            logging.info("Pipeline completed successfully")

        except Exception as e:
            conn.execute("ROLLBACK")
            logging.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    run_pipeline()
