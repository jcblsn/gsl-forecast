import json
from datetime import date
from unittest.mock import patch

import duckdb
import pytest

from src.pipeline.elt import ingest_daily, run_pipeline, transform

SAMPLE_DAILY_RESPONSE = {
    "value": {
        "timeSeries": [
            {
                "values": [
                    {
                        "value": [
                            {
                                "dateTime": "2022-01-01T00:00:00",
                                "value": "4195.5",
                                "qualifiers": ["P"],
                            },
                            {
                                "dateTime": "2022-01-15T00:00:00",
                                "value": "4196.0",
                                "qualifiers": ["P"],
                            },
                            {
                                "dateTime": "2022-02-01T00:00:00",
                                "value": "4196.5",
                                "qualifiers": ["A"],
                            },
                        ]
                    }
                ]
            }
        ]
    }
}

DAILY_SOURCE_CONFIG = {
    "url": "https://example.com/nwis/dv?site={site}&startDT={start}&endDT={end}",
    "site_id": "10010000",
}


@pytest.fixture
def conn():
    with duckdb.connect(":memory:") as conn:
        yield conn


def _mock_requests_get(response_json):
    class MockResponse:
        def json(self):
            return response_json

        def raise_for_status(self):
            pass

    return MockResponse()


class TestIngestDaily:
    @patch("requests.get")
    def test_creates_table_and_inserts_rows(self, mock_get, conn):
        mock_get.return_value = _mock_requests_get(SAMPLE_DAILY_RESPONSE)

        ingest_daily(conn, DAILY_SOURCE_CONFIG)

        rows = conn.execute(
            "SELECT * FROM usgs_water_surface_elevation_daily ORDER BY d"
        ).fetchall()
        assert len(rows) == 3
        assert rows[0] == (date(2022, 1, 1), 4195.5, "P")
        assert rows[2] == (date(2022, 2, 1), 4196.5, "A")

    @patch("requests.get")
    def test_incremental_does_not_duplicate(self, mock_get, conn):
        mock_get.return_value = _mock_requests_get(SAMPLE_DAILY_RESPONSE)

        ingest_daily(conn, DAILY_SOURCE_CONFIG)
        ingest_daily(conn, DAILY_SOURCE_CONFIG)

        count = conn.execute("SELECT COUNT(*) FROM usgs_water_surface_elevation_daily").fetchone()[
            0
        ]
        assert count == 3

    @patch("requests.get")
    def test_incremental_uses_max_date_as_start(self, mock_get, conn):
        mock_get.return_value = _mock_requests_get(SAMPLE_DAILY_RESPONSE)
        ingest_daily(conn, DAILY_SOURCE_CONFIG)

        mock_get.return_value = _mock_requests_get(SAMPLE_DAILY_RESPONSE)
        ingest_daily(conn, DAILY_SOURCE_CONFIG)

        # Check that the second call used a start date based on existing max
        _, kwargs = mock_get.call_args
        called_url = mock_get.call_args[0][0]
        assert "2022-02-01" in called_url

    @patch("requests.get")
    def test_handles_empty_timeseries(self, mock_get, conn):
        mock_get.return_value = _mock_requests_get({"value": {"timeSeries": []}})

        ingest_daily(conn, DAILY_SOURCE_CONFIG)

        count = conn.execute("SELECT COUNT(*) FROM usgs_water_surface_elevation_daily").fetchone()[
            0
        ]
        assert count == 0

    @patch("requests.get")
    def test_skips_invalid_elevation_values(self, mock_get, conn):
        response = {
            "value": {
                "timeSeries": [
                    {
                        "values": [
                            {
                                "value": [
                                    {
                                        "dateTime": "2022-01-01T00:00:00",
                                        "value": "not_a_number",
                                        "qualifiers": ["P"],
                                    },
                                    {
                                        "dateTime": "2022-01-02T00:00:00",
                                        "value": "4196.0",
                                        "qualifiers": ["P"],
                                    },
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        mock_get.return_value = _mock_requests_get(response)

        ingest_daily(conn, DAILY_SOURCE_CONFIG)

        count = conn.execute("SELECT COUNT(*) FROM usgs_water_surface_elevation_daily").fetchone()[
            0
        ]
        assert count == 1


class TestTransform:
    def test_creates_monthly_elevation_table(self, conn):
        conn.execute("""
            CREATE TABLE usgs_water_surface_elevation_daily (
                d DATE PRIMARY KEY,
                elevation FLOAT,
                qualifiers VARCHAR
            )
        """)
        conn.execute("""
            INSERT INTO usgs_water_surface_elevation_daily VALUES
            ('2022-01-01', 4195.5, 'P'),
            ('2022-01-15', 4196.0, 'P'),
            ('2022-02-01', 4196.5, 'P')
        """)

        transform(conn)

        rows = conn.execute("SELECT * FROM monthly_elevation ORDER BY month").fetchall()
        assert len(rows) == 2
        assert rows[0][0].strftime("%Y-%m") == "2022-01"
        assert abs(rows[0][1] - (4195.5 + 4196.0) / 2) < 1e-6
        assert rows[1][0].strftime("%Y-%m") == "2022-02"
        assert rows[1][4] == 1  # observation_count

    def test_replaces_existing_table(self, conn):
        conn.execute("""
            CREATE TABLE usgs_water_surface_elevation_daily (
                d DATE PRIMARY KEY, elevation FLOAT, qualifiers VARCHAR
            )
        """)
        conn.execute(
            "INSERT INTO usgs_water_surface_elevation_daily VALUES ('2022-01-01', 4195.0, 'P')"
        )

        transform(conn)
        transform(conn)  # should not error

        count = conn.execute("SELECT COUNT(*) FROM monthly_elevation").fetchone()[0]
        assert count == 1


class TestRunPipeline:
    @patch("src.pipeline.elt.ingest_continuous")
    @patch("requests.get")
    def test_pipeline_creates_tables(self, mock_get, mock_ingest_continuous, tmp_path):
        mock_get.return_value = _mock_requests_get(SAMPLE_DAILY_RESPONSE)

        config = {
            "sources": {
                "usgs_water_surface_elevation_continuous": {
                    "url": "https://example.com?site={site}&start={start}&end={end}",
                    "site_id": "10010100",
                },
                "usgs_water_surface_elevation_daily": {
                    "url": "https://example.com?site={site}&startDT={start}&endDT={end}",
                    "site_id": "10010000",
                },
            },
            "database": {"path": str(tmp_path / "test.db")},
        }
        config_path = str(tmp_path / "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        run_pipeline(config_path)

        with duckdb.connect(str(tmp_path / "test.db")) as conn:
            tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
            assert "usgs_water_surface_elevation_daily" in tables
            assert "monthly_elevation" in tables

    @patch("src.pipeline.elt.ingest_continuous")
    @patch("src.pipeline.elt.ingest_daily")
    def test_pipeline_rolls_back_on_error(
        self, mock_ingest_daily, mock_ingest_continuous, tmp_path
    ):
        mock_ingest_daily.side_effect = Exception("fetch failed")

        config = {
            "sources": {
                "usgs_water_surface_elevation_continuous": {"url": "x", "site_id": "1"},
                "usgs_water_surface_elevation_daily": {"url": "x", "site_id": "1"},
            },
            "database": {"path": str(tmp_path / "test.db")},
        }
        config_path = str(tmp_path / "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        with pytest.raises(Exception, match="fetch failed"):
            run_pipeline(config_path)


class TestRdbValueColumns:
    def test_selects_by_parameter_code(self):
        from src.pipeline.elt import rdb_value_columns

        cols = ["agency_cd", "site_no", "datetime", "tz_cd", "144241_62614", "144241_62614_cd"]
        assert rdb_value_columns(cols, "62614") == ("144241_62614", "144241_62614_cd")

    def test_raises_when_missing(self):
        import pytest

        from src.pipeline.elt import rdb_value_columns

        with pytest.raises(RuntimeError, match="62614"):
            rdb_value_columns(["agency_cd", "site_no", "datetime"], "62614")
