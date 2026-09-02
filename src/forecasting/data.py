import duckdb
import pandas as pd


def load_monthly_data(
    db: str | duckdb.DuckDBPyConnection, train_start: str | None = None
) -> pd.DataFrame:
    """monthly_elevation left-joined to monthly_covariates when that table exists."""
    if isinstance(db, duckdb.DuckDBPyConnection):
        return _load(db, train_start)
    with duckdb.connect(db, read_only=True) as conn:
        return _load(conn, train_start)


def _load(conn: duckdb.DuckDBPyConnection, train_start: str | None) -> pd.DataFrame:
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    if "monthly_covariates" in tables:
        df = conn.execute(
            """
            SELECT e.month, e.avg_elevation, c.* EXCLUDE (month)
            FROM monthly_elevation e LEFT JOIN monthly_covariates c USING (month)
            ORDER BY e.month
            """
        ).fetchdf()
    else:
        df = conn.execute(
            "SELECT month, avg_elevation FROM monthly_elevation ORDER BY month"
        ).fetchdf()
    df["month"] = pd.to_datetime(df["month"])
    if train_start:
        df = df[df["month"] >= pd.Timestamp(train_start)].reset_index(drop=True)
    return df
