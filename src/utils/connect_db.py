from contextlib import contextmanager
from typing import Generator

import duckdb

from .load_config import load_configuration


class DatabaseError(Exception):
    pass


@contextmanager
def get_db_connection(
    config_path: str,
) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    try:
        config = load_configuration(config_path)
        conn = duckdb.connect(config["database"]["path"])
        yield conn
    except Exception as e:
        raise DatabaseError(f"Failed to connect to database: {str(e)}") from e
    finally:
        if "conn" in locals():
            conn.close()
