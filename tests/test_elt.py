import json
from datetime import date
from unittest.mock import patch

import duckdb
import pytest

from src.pipeline.elt import SOUTH_ARM_TABLE, run_pipeline, transform
from src.pipeline.usgs import fetch_usgs_daily, ingest_elevation


def feature(day: str, value: str, qualifier=None, approval="Approved") -> dict:
    return {
        "properties": {
            "time": day,
            "value": value,
            "qualifier": qualifier,
            "approval_status": approval,
        }
    }


def page(features: list[dict], cursor: str | None = None) -> dict:
    links = [{"rel": "next", "href": f"https://example/items?cursor={cursor}"}] if cursor else []
    return {"features": features, "links": links}


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.fixture
def conn():
    with duckdb.connect(":memory:") as conn:
        yield conn


class TestFetchUsgsDaily:
    @patch("src.pipeline.usgs.requests.get")
    def test_walks_cursor_pages_and_skips_bad_values(self, mock_get):
        mock_get.side_effect = [
            FakeResponse(page([feature("2022-01-01", "4195.5", ["ESTIMATED"])], cursor="abc")),
            FakeResponse(page([feature("2022-01-02", "bad"), feature("2022-01-03", "4196")])),
        ]
        rows = fetch_usgs_daily("10010000", "62614", "2022-01-01")
        assert rows == [
            ("2022-01-01", 4195.5, "Approved,ESTIMATED"),
            ("2022-01-03", 4196.0, "Approved"),
        ]
        first, second = (c.kwargs["params"] for c in mock_get.call_args_list)
        assert first["monitoring_location_id"] == "USGS-10010000"
        assert first["datetime"] == "2022-01-01/.."
        assert "cursor" not in first and second["cursor"] == "abc"

    @patch("src.pipeline.usgs.requests.get")
    def test_empty_page_returns_nothing(self, mock_get):
        mock_get.return_value = FakeResponse(page([]))
        assert fetch_usgs_daily("10010000", "62614", "2022-01-01", "2022-02-01") == []


class TestIngestElevation:
    @patch("src.pipeline.usgs.fetch_usgs_daily")
    def test_creates_table_and_upserts(self, mock_fetch, conn):
        mock_fetch.return_value = [("2022-01-01", 4195.5, "P"), ("2022-02-01", 4196.5, "A")]
        ingest_elevation(conn, SOUTH_ARM_TABLE, "10010000", "62614", "1847-10-18")
        assert mock_fetch.call_args[0][2] == "1847-10-18"

        mock_fetch.return_value = [("2022-02-01", 4196.6, "A"), ("2022-03-01", 4197.0, "P")]
        ingest_elevation(conn, SOUTH_ARM_TABLE, "10010000", "62614", "1847-10-18")
        assert mock_fetch.call_args[0][2] == "2021-12-18"

        rows = conn.execute(f"SELECT * FROM {SOUTH_ARM_TABLE} ORDER BY d").fetchall()
        assert rows == [
            (date(2022, 1, 1), 4195.5, "P"),
            (date(2022, 2, 1), pytest.approx(4196.6), "A"),
            (date(2022, 3, 1), 4197.0, "P"),
        ]


class TestTransform:
    def test_creates_monthly_elevation_table(self, conn):
        conn.execute(
            f"CREATE TABLE {SOUTH_ARM_TABLE} (d DATE, elevation FLOAT, qualifiers VARCHAR)"
        )
        conn.execute(f"""
            INSERT INTO {SOUTH_ARM_TABLE} VALUES
            ('2022-01-01', 4195.5, 'P'), ('2022-01-15', 4196.0, 'P'), ('2022-02-01', 4196.5, 'P')
        """)
        transform(conn)
        rows = conn.execute("SELECT * FROM monthly_elevation ORDER BY month").fetchall()
        assert len(rows) == 2
        assert rows[0][0].strftime("%Y-%m") == "2022-01"
        assert abs(rows[0][1] - (4195.5 + 4196.0) / 2) < 1e-6
        assert rows[1][4] == 1

    def test_partial_current_month_is_dropped(self, conn):
        conn.execute(
            f"CREATE TABLE {SOUTH_ARM_TABLE} (d DATE, elevation FLOAT, qualifiers VARCHAR)"
        )
        conn.execute(f"INSERT INTO {SOUTH_ARM_TABLE} VALUES ('2022-01-01', 4195.0, 'A')")
        conn.execute(
            f"INSERT INTO {SOUTH_ARM_TABLE} VALUES (DATE_TRUNC('month', CURRENT_DATE), 4190.0, 'P')"
        )
        transform(conn)
        transform(conn)
        months = [r[0] for r in conn.execute("SELECT month FROM monthly_elevation").fetchall()]
        assert len(months) == 1 and str(months[0]).startswith("2022-01")


@pytest.fixture
def config_path(tmp_path):
    config = {
        "sources": {
            "south_arm_site": "10010000",
            "south_arm_start": "1847-10-18",
            "elevation_parameter": "62614",
        },
        "database": {"path": str(tmp_path / "test.db")},
        "covariates": {"snotel": {}, "usgs_discharge": {}, "north_arm": {}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return str(path)


class TestRunPipeline:
    @patch("src.pipeline.elt.run_covariates")
    @patch("src.pipeline.usgs.fetch_usgs_daily")
    def test_pipeline_creates_tables(self, mock_fetch, mock_cov, config_path, tmp_path):
        mock_fetch.return_value = [("2022-01-01", 4195.5, "P")]
        run_pipeline(config_path)
        with duckdb.connect(str(tmp_path / "test.db")) as conn:
            tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
            assert SOUTH_ARM_TABLE in tables and "monthly_elevation" in tables
        mock_cov.assert_called_once()

    @patch("src.pipeline.elt.run_covariates", side_effect=Exception("awdb down"))
    @patch("src.pipeline.usgs.fetch_usgs_daily")
    def test_covariate_failure_keeps_elevation(self, mock_fetch, mock_cov, config_path, tmp_path):
        mock_fetch.return_value = [("2022-01-01", 4195.5, "P")]
        run_pipeline(config_path)
        with duckdb.connect(str(tmp_path / "test.db")) as conn:
            assert conn.execute("SELECT COUNT(*) FROM monthly_elevation").fetchone()[0] == 1

    @patch("src.pipeline.elt.ingest_elevation", side_effect=Exception("fetch failed"))
    def test_pipeline_raises_on_elevation_failure(self, mock_ingest, config_path):
        with pytest.raises(Exception, match="fetch failed"):
            run_pipeline(config_path)
