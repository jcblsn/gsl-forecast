import argparse
import logging
import os

import duckdb

from src.config import load_config
from src.pipeline.covariates import run_covariates
from src.pipeline.quality import is_provisional_sql
from src.pipeline.usgs import ingest_elevation

SOUTH_ARM_TABLE = "usgs_water_surface_elevation_daily"

# `avg_elevation` is a mean over the days of the month. A water balance closes between two
# instants, not between two means, so a storage model needs an end-of-month state. These are
# the candidate rules for that state. The literal last day is one wind event away from a
# value the whole month does not support, so a short median is offered beside it.
#
# Measured over 1989-2026, more smoothing always closes the balance better: the residual is
# 0.136 ft/month for `last`, 0.129 for `median_3d` and 0.121 for `median_7d`. Forecast
# accuracy does not follow. Lead-1 MAE is 0.094, 0.093 and 0.103 ft for the same 3 rules, so
# `last` and `median_3d` cannot be told apart and `median_7d` is worse. `median_3d` is the
# default because it matches the best forecast and resists a single wind-driven reading.
ENDPOINT_RULES = {
    "last": "last_elevation",
    "median_3d": "endpoint_3d_median",
    "median_7d": "endpoint_7d_median",
}
DEFAULT_ENDPOINT_RULE = "median_3d"


def transform(conn: duckdb.DuckDBPyConnection, endpoint_rule: str = DEFAULT_ENDPOINT_RULE) -> None:
    """Roll the daily gauge up to months.

    `avg_elevation` is the published target and stays the mean over the reporting days.
    `elevation_eom_ft` is the end-of-month state a storage model steps between, chosen by
    `endpoint_rule`.
    """
    if endpoint_rule not in ENDPOINT_RULES:
        raise ValueError(
            f"Unknown endpoint rule {endpoint_rule}; expected one of {sorted(ENDPOINT_RULES)}"
        )
    endpoint_column = ENDPOINT_RULES[endpoint_rule]
    conn.execute(f"""
        CREATE OR REPLACE TABLE monthly_elevation AS
        WITH complete_months AS (
            SELECT *
            FROM {SOUTH_ARM_TABLE}
            WHERE d < DATE_TRUNC('month', CURRENT_DATE)
        )
        SELECT DATE_TRUNC('month', d) AS month,
               AVG(elevation) AS avg_elevation,
               MIN(elevation) AS min_elevation,
               MAX(elevation) AS max_elevation,
               COUNT(*) AS observation_count,
               ARG_MAX(elevation, d) AS last_elevation,
               MAX(d) AS last_observation_date,
               DATE_DIFF('day', MAX(d), LAST_DAY(MAX(d))) AS endpoint_age_days,
               MEDIAN(elevation) FILTER (
                   WHERE d >= LAST_DAY(d) - INTERVAL 2 DAY
               ) AS endpoint_3d_median,
               COUNT(*) FILTER (
                   WHERE d >= LAST_DAY(d) - INTERVAL 2 DAY
               ) AS endpoint_3d_observation_count,
               MEDIAN(elevation) FILTER (
                   WHERE d >= LAST_DAY(d) - INTERVAL 6 DAY
               ) AS endpoint_7d_median,
               COUNT(*) FILTER (
                   WHERE d >= LAST_DAY(d) - INTERVAL 6 DAY
               ) AS endpoint_7d_observation_count,
               COUNT(*) FILTER (
                   WHERE {is_provisional_sql("qualifiers")}
               ) AS provisional_observation_count
        FROM complete_months
        GROUP BY DATE_TRUNC('month', d)
        ORDER BY month
    """)
    conn.execute("""
        ALTER TABLE monthly_elevation
        ADD COLUMN elevation_eom_ft DOUBLE
    """)
    conn.execute(f"""
        UPDATE monthly_elevation
        SET elevation_eom_ft = COALESCE({endpoint_column}, last_elevation)
    """)
    conn.execute(
        "COMMENT ON COLUMN monthly_elevation.elevation_eom_ft IS "
        f"'End-of-month state under rule {endpoint_rule}'"
    )


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
            transform(conn, src.get("endpoint_rule", DEFAULT_ENDPOINT_RULE))
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
