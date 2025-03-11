from contextlib import contextmanager
from typing import Generator

import duckdb


class DatabaseError(Exception):
    pass


@contextmanager
def get_db_connection(db_path: str) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    try:
        conn = duckdb.connect(db_path)
        yield conn
    except Exception as e:
        raise DatabaseError(f"Failed to connect to database: {str(e)}") from e
    finally:
        if "conn" in locals():
            conn.close()
