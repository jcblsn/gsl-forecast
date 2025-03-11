import os
import tempfile
from unittest.mock import patch

import duckdb
import pytest

from src.utils.connect_db import DatabaseError, get_db_connection


@pytest.fixture
def temp_db_path():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)
    os.rmdir(temp_dir)


class TestDatabaseConnection:
    def test_successful_connection(self, temp_db_path):
        with get_db_connection(temp_db_path) as conn:
            assert isinstance(conn, duckdb.DuckDBPyConnection)
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("INSERT INTO test VALUES (1)")
            result = conn.execute("SELECT * FROM test").fetchall()
            assert result == [(1,)]

    @patch("duckdb.connect")
    def test_connection_error(self, mock_connect):
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(DatabaseError) as exc_info:
            with get_db_connection("invalid_path"):
                pass

        assert "Failed to connect to database: Connection failed" in str(exc_info.value)

    def test_connection_closes_after_use(self, temp_db_path):
        with get_db_connection(temp_db_path) as conn:
            pass

        with pytest.raises(duckdb.ConnectionException):
            conn.execute("SELECT 1")

    def test_connection_closes_after_exception(self, temp_db_path):
        conn = None
        try:
            with get_db_connection(temp_db_path) as conn:
                raise ValueError("Test exception")
        except DatabaseError:
            pass

        with pytest.raises(duckdb.ConnectionException):
            conn.execute("SELECT 1")
