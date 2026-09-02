import argparse
import logging
import os

import duckdb

from src.config import load_config
from src.pipeline.covariates import run_covariates
from src.pipeline.usgs import ingest_elevation

SOUTH_ARM_TABLE = "usgs_water_surface_elevation_daily"


def transform(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(f"""
        CREATE OR REPLACE TABLE monthly_elevation AS
        SELECT
            DATE_TRUNC('month', d) AS month,
            AVG(elevation) AS avg_elevation,
            MIN(elevation) AS min_elevation,
            MAX(elevation) AS max_elevation,
            COUNT(*) AS observation_count
        FROM {SOUTH_ARM_TABLE}
        WHERE d < DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY month
        ORDER BY month
    """)


def run_pipeline(config_path: str | None = None, skip_covariates: bool = False) -> None:
    """Elevation first in its own transaction, then covariates in another, so a covariate
    source outage still leaves an up-to-date target series for the univariate models."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(config_path)
    db_path = config["database"]["path"]
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    src = config["sources"]

    with duckdb.connect(db_path) as conn:
        try:
            conn.execute("BEGIN TRANSACTION")
            ingest_elevation(
                conn,
                SOUTH_ARM_TABLE,
                src["south_arm_site"],
                src["elevation_parameter"],
                src["south_arm_start"],
            )
            transform(conn)
            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            logging.error(f"Elevation ingest failed: {e}")
            raise
        if "covariates" in config and not skip_covariates:
            try:
                conn.execute("BEGIN TRANSACTION")
                run_covariates(conn, config)
                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                logging.error(f"Covariate ingest failed; monthly_covariates left as it was: {e}")
        conn.execute("VACUUM")
        logging.info("Pipeline completed successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch USGS data into the local DuckDB")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--skip-covariates", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.config, skip_covariates=args.skip_covariates)


if __name__ == "__main__":
    main()
