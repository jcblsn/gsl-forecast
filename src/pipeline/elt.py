import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import duckdb
import requests

from src.utils.connect_db import get_db_connection
from src.utils.load_config import load_configuration


def get_cache_info(config: Dict) -> Dict[str, Dict[str, Optional[datetime]]]:
    cache_base = config["storage"]["local"]["path"]
    sources = config["sources"].keys()

    cache_info: Dict[str, Dict[str, Optional[datetime]]] = {}

    for source in sources:
        cache_dir = os.path.join(cache_base, source)
        cache_info[source] = {"exists": os.path.isdir(cache_dir), "last_date": None}

        if not cache_info[source]["exists"]:
            continue

        for filename in os.listdir(cache_dir):
            if not (filename.endswith(".csv") or filename.endswith(".parquet")):
                continue

            year_str = filename.split("_")[0]
            try:
                year = int(year_str)
                file_date = datetime(year, 1, 1)

                if (
                    cache_info[source]["last_date"] is None
                    or file_date > cache_info[source]["last_date"]
                ):
                    cache_info[source]["last_date"] = file_date
            except (ValueError, IndexError):
                continue

    return cache_info


def get_db_info(conn: duckdb.DuckDBPyConnection, sources: List[str]) -> Dict[str, bool]:
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]

    return {source: source in table_names for source in sources}


def extract(conn: duckdb.DuckDBPyConnection, config: Dict, cache_info: Dict) -> None:
    for source, source_config in config["sources"].items():
        extract_for_source(conn, source, source_config, cache_info.get(source, {}))


def extract_for_source(
    conn: duckdb.DuckDBPyConnection, source: str, source_config: Dict, cache_info: Dict
) -> None:
    last_date = cache_info.get("last_date")
    end_date = datetime.now()

    if source == "usgs_water_surface_elevation_continuous":
        start_date = (
            datetime(2007, 10, 1)
            if last_date is None
            else datetime(last_date.year + 1, 1, 1)
        )
        site_id = source_config.get("site_id", "10010100")
        url = source_config["url"].format(
            site=site_id,
            start=start_date.strftime("%Y-%m-%dT%H:%M:%S"),
            end=end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        logging.info(f"Fetching continuous USGS water surface data from: {url}")
        try:
            conn.execute("SET force_download=true")
            conn.execute(f"""
                CREATE TEMP TABLE {source}_staging AS
                SELECT *
                FROM read_csv_auto(
                    '{url}',
                    delim='\t',
                    skip=1,
                    comment='#',
                    header=true
                )
                WHERE datetime != '20d'
            """)
            logging.info(f"Successfully loaded new data into {source}_staging.")
        except duckdb.IOException as e:
            logging.error(f"Failed to fetch data from {url}: {str(e)}")
            conn.execute(
                f'CREATE TEMP TABLE IF NOT EXISTS {source}_staging (agency_cd VARCHAR, site_no VARCHAR, datetime VARCHAR, tz_cd VARCHAR, "144241_62614" VARCHAR, "144241_62614_cd" VARCHAR)'
            )

    elif source == "usgs_water_surface_elevation_daily":
        start_date = (
            "1847-10-18"
            if last_date is None
            else datetime(last_date.year + 1, 1, 1).strftime("%Y-%m-%d")
        )
        end_date_str = end_date.strftime("%Y-%m-%d")

        current_year = end_date.year
        if last_date and last_date.year == current_year:
            logging.info(f"Daily data already up to date (last cache: {last_date})")
            conn.execute(
                f"CREATE TEMP TABLE IF NOT EXISTS {source}_staging (date VARCHAR, value VARCHAR, qualifiers VARCHAR)"
            )
            return

        site_id = source_config.get("site_id", "10010000")
        url = source_config["url"].format(
            site=site_id, start=start_date, end=end_date_str
        )

        logging.info(f"Fetching daily USGS water surface data from: {url}")
        try:
            conn.execute(f"CREATE TEMP TABLE {source}_url AS SELECT '{url}' as url")
            conn.execute(
                f"CREATE TEMP TABLE {source}_staging (date VARCHAR, value VARCHAR, qualifiers VARCHAR)"
            )
            logging.info(f"Stored URL for {source}")
        except Exception as e:
            logging.error(f"Error setting up {source} extraction: {e}")
            conn.execute(
                f"CREATE TEMP TABLE IF NOT EXISTS {source}_staging (date VARCHAR, value VARCHAR, qualifiers VARCHAR)"
            )
            conn.execute(
                f"CREATE TEMP TABLE IF NOT EXISTS {source}_url AS SELECT '' as url"
            )


def load(
    conn: duckdb.DuckDBPyConnection, config: Dict, cache_info: Dict, db_info: Dict
) -> None:
    for source in config["sources"].keys():
        load_for_source(
            conn, source, config, cache_info.get(source, {}), db_info.get(source, False)
        )


def load_for_source(
    conn: duckdb.DuckDBPyConnection,
    source: str,
    config: Dict,
    cache_info: Dict,
    table_exists: bool,
) -> None:
    cache_base = config["storage"]["local"]["path"]
    cache_dir = os.path.join(cache_base, source)

    if source == "usgs_water_surface_elevation_continuous":
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {source} (
                agency VARCHAR,
                site VARCHAR,
                dt TIMESTAMP,
                tz VARCHAR,
                elevation FLOAT,
                qualifiers VARCHAR,
                PRIMARY KEY (site, dt)
            )
        """)

        if cache_info.get("exists", False):
            conn.execute(f"""
                INSERT OR IGNORE INTO {source}
                SELECT * FROM read_csv_auto('{cache_dir}/*.csv', header=true, delim=',', union_by_name=true)
            """)
            logging.info(f"Loaded cached data for {source}")

        conn.execute(f"""
            INSERT OR IGNORE INTO {source}
            SELECT
                agency_cd AS agency,
                site_no AS site,
                CAST(datetime || ':00' AS TIMESTAMP) AS dt,
                tz_cd AS tz,
                CAST("144241_62614" AS FLOAT) AS elevation,
                "144241_62614_cd" AS qualifiers
            FROM {source}_staging
            WHERE datetime IS NOT NULL
        """)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        current_year = datetime.now().year
        conn.execute(f"""
            WITH years AS (
                SELECT DISTINCT EXTRACT(YEAR FROM dt) as year
                FROM {source}
                WHERE EXTRACT(YEAR FROM dt) < {current_year}
                ORDER BY year
            )
            SELECT year FROM years
        """)
        years = [row[0] for row in conn.fetchall()]

        for year in years:
            cache_file = os.path.join(cache_dir, f"{year}_{source}.csv")
            if os.path.exists(cache_file):
                continue

            conn.execute(f"""
                COPY (
                    SELECT *
                    FROM {source}
                    WHERE EXTRACT(YEAR FROM dt) = {year}
                ) TO '{cache_file}' (HEADER, DELIMITER ',')
            """)
            logging.info(f"Cached data for {source} for year {year}")
    elif source == "usgs_water_surface_elevation_daily":

        def get_usgs_daily_data(url: str) -> List[tuple]:
            rows: List[tuple] = []
            try:
                response = requests.get(url)
                response.raise_for_status()
                data_json = response.json()

                try:
                    daily_ts = data_json["value"]["timeSeries"][0]["values"][0]["value"]
                except (KeyError, IndexError, TypeError) as e:
                    logging.error(f"Failed to extract daily data from JSON: {e}")
                    return rows

                for entry in daily_ts:
                    date_str = entry.get("dateTime", "").split("T")[0]
                    if not date_str:
                        continue
                    try:
                        elevation_val = float(entry.get("value", ""))
                    except ValueError:
                        continue
                    qualifiers_val = entry.get("qualifiers", [""])[0]
                    rows.append((date_str, str(elevation_val), qualifiers_val))
                return rows
            except Exception as e:
                logging.error(f"Failed to fetch JSON from {url}: {e}")
                return rows

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {source} (
                d DATE PRIMARY KEY,
                elevation FLOAT,
                qualifiers VARCHAR
            )
        """)
        result = conn.execute(f"SELECT url FROM {source}_url").fetchone()
        if not result or not result[0]:
            raise Exception("No URL found for daily data load; skipping")
        else:
            rows = get_usgs_daily_data(result[0])
            if rows:
                conn.executemany(
                    f"INSERT INTO {source}_staging VALUES (?, ?, ?)",
                    rows,
                )
                conn.execute(f"""
                    INSERT OR IGNORE INTO {source}
                    SELECT
                        CAST(date AS DATE) as d,
                        CAST(value AS FLOAT) as elevation,
                        qualifiers
                    FROM {source}_staging
                    WHERE date IS NOT NULL
                """)
    conn.execute(f"DROP TABLE IF EXISTS {source}_url")
    conn.execute(f"DROP TABLE IF EXISTS {source}_staging")
    conn.execute(f"DROP TABLE IF EXISTS {source}_raw_json")
    conn.execute(f"DROP TABLE IF EXISTS {source}_extracted")


def transform(conn: duckdb.DuckDBPyConnection, config: Dict) -> None:
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

    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(base_dir, "config", "config.json")

    config = load_configuration(config_path)

    cache_base = config["storage"]["local"]["path"]
    os.makedirs(cache_base, exist_ok=True)

    with get_db_connection(config["database"]["path"]) as conn:
        cache_info = get_cache_info(config)
        db_info = get_db_info(conn, config["sources"].keys())

        logging.info("Database and cache state:")
        for source in config["sources"].keys():
            source_cache = cache_info.get(source, {})
            cache_exists = source_cache.get("exists", False)
            table_exists = db_info.get(source, False)

            scenario = "unknown"
            if not table_exists and not cache_exists:
                scenario = "No database table exists, no cache files exist"
            elif not table_exists and cache_exists:
                scenario = "No database table exists, cache files exist"
            elif table_exists and not cache_exists:
                scenario = "Database table exists, no cache files exist"
            elif table_exists and cache_exists:
                scenario = "Database table exists, cache files exist"

            logging.info(f"  {source}: {scenario}")

        try:
            conn.execute("BEGIN TRANSACTION")

            extract(conn, config, cache_info)
            load(conn, config, cache_info, db_info)
            transform(conn, config)

            conn.execute("COMMIT")
            conn.execute("VACUUM")
            logging.info("Pipeline completed successfully")

        except Exception as e:
            conn.execute("ROLLBACK")
            logging.error(f"Pipeline failed with error: {str(e)}")
            raise


if __name__ == "__main__":
    run_pipeline()
