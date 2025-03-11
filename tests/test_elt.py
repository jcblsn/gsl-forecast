import os
import shutil
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.pipeline.elt import (
    extract_for_source,
    get_cache_info,
    get_db_info,
    load_for_source,
    run_pipeline,
    transform,
)
from src.utils.connect_db import get_db_connection


@pytest.fixture
def mock_config():
    return {
        "sources": {
            "usgs_water_surface_elevation_continuous": {
                "url": "https://example.com/continuous?site={site}&startDT={start}&endDT={end}",
                "site_id": "10010100",
            },
            "usgs_water_surface_elevation_daily": {
                "url": "https://example.com/daily?site={site}&startDT={start}&endDT={end}",
                "site_id": "10010000",
            },
        },
        "storage": {"local": {"path": "./data/cache"}},
        "database": {"path": "./data/gsl.db"},
    }


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def db_conn(temp_dir):
    db_path = os.path.join(temp_dir, "test.db")
    with get_db_connection(db_path) as conn:
        yield conn
    if os.path.exists(db_path):
        os.remove(db_path)


class TestCacheInfo:
    def test_get_cache_info_empty(self, mock_config, temp_dir):
        mock_config["storage"]["local"]["path"] = temp_dir

        result = get_cache_info(mock_config)

        assert "usgs_water_surface_elevation_continuous" in result
        assert not result["usgs_water_surface_elevation_continuous"]["exists"]
        assert result["usgs_water_surface_elevation_continuous"]["last_date"] is None

    def test_get_cache_info_with_files(self, mock_config, temp_dir):
        mock_config["storage"]["local"]["path"] = temp_dir
        source = "usgs_water_surface_elevation_continuous"

        cache_dir = os.path.join(temp_dir, source)
        os.makedirs(cache_dir)

        with open(
            os.path.join(cache_dir, "2020_usgs_water_surface_elevation_continuous.csv"),
            "w",
        ) as f:
            f.write("header\n")
        with open(
            os.path.join(cache_dir, "2021_usgs_water_surface_elevation_continuous.csv"),
            "w",
        ) as f:
            f.write("header\n")

        result = get_cache_info(mock_config)

        assert result[source]["exists"]
        assert result[source]["last_date"].year == 2021


class TestDatabaseInfo:
    def test_get_db_info_empty(self, db_conn):
        result = get_db_info(db_conn, ["usgs_water_surface_elevation_continuous"])
        assert not result["usgs_water_surface_elevation_continuous"]

    def test_get_db_info_with_tables(self, db_conn):
        db_conn.execute(
            "CREATE TABLE usgs_water_surface_elevation_continuous (id INTEGER)"
        )

        result = get_db_info(
            db_conn, ["usgs_water_surface_elevation_continuous", "nonexistent_table"]
        )

        assert result["usgs_water_surface_elevation_continuous"]
        assert not result["nonexistent_table"]


class TestExtract:
    @patch("duckdb.DuckDBPyConnection.execute")
    def test_extract_continuous_new_data(self, mock_execute, mock_config, db_conn):
        source = "usgs_water_surface_elevation_continuous"
        source_config = mock_config["sources"][source]
        cache_info = {"last_date": None}

        extract_for_source(db_conn, source, source_config, cache_info)

        assert any(
            "CREATE TEMP TABLE" in call[0][0] for call in mock_execute.call_args_list
        )

    @patch("duckdb.DuckDBPyConnection.execute")
    def test_extract_continuous_incremental(self, mock_execute, mock_config, db_conn):
        source = "usgs_water_surface_elevation_continuous"
        source_config = mock_config["sources"][source]
        cache_info = {"last_date": datetime(2021, 12, 31)}

        extract_for_source(db_conn, source, source_config, cache_info)

        create_calls = [
            call[0][0]
            for call in mock_execute.call_args_list
            if "CREATE TEMP TABLE" in call[0][0]
        ]
        assert any("2022-01-01" in call or "2022" in call for call in create_calls)

    @patch("requests.get")
    @patch("duckdb.DuckDBPyConnection.execute")
    def test_extract_daily_with_api_error(
        self, mock_execute, mock_get, mock_config, db_conn
    ):
        source = "usgs_water_surface_elevation_daily"
        source_config = mock_config["sources"][source]
        cache_info = {"last_date": None}

        mock_get.side_effect = requests.RequestException("API Error")

        extract_for_source(db_conn, source, source_config, cache_info)

        assert any(
            "CREATE TEMP TABLE" in call[0][0] for call in mock_execute.call_args_list
        )

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("duckdb.DuckDBPyConnection.execute")
    @patch("duckdb.DuckDBPyConnection.fetchall")
    def test_load_continuous_first_time(
        self,
        mock_fetchall,
        mock_execute,
        mock_exists,
        mock_makedirs,
        mock_config,
        db_conn,
    ):
        """Test loading continuous data for the first time"""
        source = "usgs_water_surface_elevation_continuous"
        cache_info = {"exists": False}
        table_exists = False

        mock_exists.return_value = False
        mock_fetchall.return_value = []

        load_for_source(db_conn, source, mock_config, cache_info, table_exists)

        create_calls = [
            call[0][0]
            for call in mock_execute.call_args_list
            if "CREATE TABLE IF NOT EXISTS" in call[0][0]
        ]
        assert len(create_calls) > 0

        assert mock_makedirs.called

    @patch("os.path.exists")
    @patch("duckdb.DuckDBPyConnection.execute")
    @patch("duckdb.DuckDBPyConnection.fetchall")
    def test_load_with_cache(
        self, mock_fetchall, mock_execute, mock_exists, mock_config, db_conn
    ):
        source = "usgs_water_surface_elevation_continuous"
        cache_info = {"exists": True}
        table_exists = True

        mock_exists.return_value = True
        mock_fetchall.return_value = [(2020,), (2021,)]

        load_for_source(db_conn, source, mock_config, cache_info, table_exists)

        load_cache_calls = [
            call[0][0]
            for call in mock_execute.call_args_list
            if "INSERT OR IGNORE INTO" in call[0][0] and "read_csv_auto" in call[0][0]
        ]
        assert len(load_cache_calls) > 0

    @patch("requests.get")
    @patch("duckdb.DuckDBPyConnection.execute")
    @patch("duckdb.DuckDBPyConnection.executemany")
    def test_load_daily_with_data(
        self, mock_executemany, mock_execute, mock_get, mock_config, db_conn
    ):
        source = "usgs_water_surface_elevation_daily"
        cache_info = {}
        table_exists = True

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "value": {
                "timeSeries": [
                    {
                        "values": [
                            {
                                "value": [
                                    {
                                        "dateTime": "2022-01-01T00:00:00",
                                        "value": "1234.56",
                                        "qualifiers": ["P"],
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        db_conn.execute(
            f"CREATE TEMP TABLE {source}_url AS SELECT 'http://example.com' as url"
        )

        load_for_source(db_conn, source, mock_config, cache_info, table_exists)

        assert mock_executemany.called


class TestTransform:
    def test_transform_creates_monthly_table(self, db_conn, mock_config):
        db_conn.execute("""
            CREATE TABLE usgs_water_surface_elevation_daily (
                d DATE,
                elevation FLOAT,
                qualifiers VARCHAR
            )
        """)
        db_conn.execute("""
            INSERT INTO usgs_water_surface_elevation_daily VALUES
            ('2022-01-01', 1234.5, 'P'),
            ('2022-01-15', 1235.0, 'P'),
            ('2022-02-01', 1235.5, 'P')
        """)

        transform(db_conn, mock_config)

        result = db_conn.execute(
            "SELECT * FROM monthly_elevation ORDER BY month"
        ).fetchall()
        assert len(result) == 2
        assert result[0][0].strftime("%Y-%m") == "2022-01"
        assert result[1][0].strftime("%Y-%m") == "2022-02"


class TestPipeline:
    @patch("src.pipeline.elt.get_db_connection")
    @patch("src.pipeline.elt.load_configuration")
    @patch("src.pipeline.elt.get_cache_info")
    @patch("src.pipeline.elt.get_db_info")
    @patch("src.pipeline.elt.extract")
    @patch("src.pipeline.elt.load")
    @patch("src.pipeline.elt.transform")
    def test_run_pipeline_success(
        self,
        mock_transform,
        mock_load,
        mock_extract,
        mock_db_info,
        mock_cache_info,
        mock_load_config,
        mock_get_db_connection,
        mock_config,
    ):
        mock_conn = MagicMock()
        mock_get_db_connection.return_value.__enter__.return_value = mock_conn
        mock_load_config.return_value = mock_config
        mock_cache_info.return_value = {
            "usgs_water_surface_elevation_continuous": {
                "exists": True,
                "last_date": datetime(2021, 12, 31),
            }
        }
        mock_db_info.return_value = {"usgs_water_surface_elevation_continuous": True}

        run_pipeline("test_config.json")

        mock_extract.assert_called_once()
        mock_load.assert_called_once()
        mock_transform.assert_called_once()

        commit_calls = [
            call[0][0]
            for call in mock_conn.execute.call_args_list
            if "COMMIT" in call[0][0]
        ]
        assert len(commit_calls) > 0

    @patch("src.pipeline.elt.get_db_connection")
    @patch("src.pipeline.elt.load_configuration")
    @patch("src.pipeline.elt.get_cache_info")
    @patch("src.pipeline.elt.get_db_info")
    @patch("src.pipeline.elt.extract")
    def test_run_pipeline_rollback_on_error(
        self,
        mock_extract,
        mock_db_info,
        mock_cache_info,
        mock_load_config,
        mock_get_db_connection,
        mock_config,
    ):
        mock_conn = MagicMock()
        mock_get_db_connection.return_value.__enter__.return_value = mock_conn
        mock_load_config.return_value = mock_config
        mock_cache_info.return_value = {
            "usgs_water_surface_elevation_continuous": {
                "exists": False,
                "last_date": None,
            }
        }
        mock_db_info.return_value = {"usgs_water_surface_elevation_continuous": False}

        mock_extract.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            run_pipeline("test_config.json")

        rollback_calls = [
            call[0][0]
            for call in mock_conn.execute.call_args_list
            if "ROLLBACK" in call[0][0]
        ]
        assert len(rollback_calls) > 0
