import json
from pathlib import Path

import duckdb
import pytest

from src.utils.connect_db import get_db_connection


def test_valid_db_connection(tmp_path: Path):
    db_path = str(tmp_path / "test.db")
    config = {
        "database": {"path": db_path},
        "storage": {"local": {"path": str(tmp_path / "storage")}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    with get_db_connection(str(config_path)) as conn:
        assert isinstance(conn, duckdb.DuckDBPyConnection)
        result = conn.execute("SELECT 1").fetchone()[0]
        assert result == 1


def test_connection_cleanup(tmp_path: Path):
    db_path = str(tmp_path / "test.db")
    config = {
        "database": {"path": db_path},
        "storage": {"local": {"path": str(tmp_path / "storage")}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    with get_db_connection(str(config_path)) as conn:
        pass

    with pytest.raises(Exception):
        conn.execute("SELECT 1")
