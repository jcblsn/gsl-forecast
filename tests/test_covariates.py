import duckdb
import pytest

from src.pipeline import covariates as cov
from src.pipeline import usgs

CFG = {
    "snotel": {
        "states": ["UT"],
        "basins": {"1601": "bear", "160201": "weber", "160202": "provo_jordan"},
        "start": "2020-01-01",
    },
    "reservoirs": {"states": ["UT"], "start": "2020-01-01"},
    "nrcs_forecasts": {"station": "10010000:UT:USGS", "start": "2020-01-01"},
    "usgs_discharge": {
        "inflow": {"bear": "10126000"},
        "exchange": {"breach": "10010020"},
        "start": "2020-01-01",
    },
    "north_arm": {"site": "10010100", "start": "2020-01-01"},
}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def fake_get(url, params=None, timeout=None):
    if url.endswith("/forecasts"):
        return FakeResponse(
            [
                {
                    "stationTriplet": "10010000:UT:USGS",
                    "data": [
                        {
                            "forecastPeriod": ["04-01", "07-31"],
                            "publicationDate": "2020-02-01 00:00",
                            "periodNormal": 450.0,
                            "forecastValues": {"90": 10.0, "50": 300.0, "10": 900.0},
                        }
                    ],
                }
            ]
        )
    if url.endswith("/stations") and "BOR" in params["stationTriplets"]:
        return FakeResponse(
            [
                {"stationTriplet": "9:UT:BOR", "name": "Res A", "huc": "160101010101"},
                {"stationTriplet": "8:UT:BOR", "name": "Res B", "huc": "160101010102"},
            ]
        )
    if url.endswith("/data") and params.get("duration") == "MONTHLY":
        return FakeResponse(
            [
                {
                    "stationTriplet": t,
                    "data": [
                        {
                            "stationElement": {"elementCode": "RESC"},
                            "values": [
                                {"month": 1, "year": 2020, "value": 100000},
                                {"month": 2, "year": 2020, "value": None},
                            ],
                        }
                    ],
                }
                for t in params["stationTriplets"].split(",")
            ]
        )
    if url.endswith("/stations"):
        return FakeResponse(
            [
                {
                    "stationTriplet": "1:UT:SNTL",
                    "name": "A",
                    "huc": "160101010101",
                    "elevation": 9000.0,
                    "latitude": 41.0,
                    "longitude": -111.0,
                    "beginDate": "1980-10-01 00:00",
                },
                {
                    "stationTriplet": "2:UT:SNTL",
                    "name": "B",
                    "huc": "160202030101",
                    "elevation": 8000.0,
                    "latitude": 40.5,
                    "longitude": -111.5,
                    "beginDate": "1990-10-01 00:00",
                },
                {"stationTriplet": "3:UT:SNTL", "name": "Elsewhere", "huc": "160300020301"},
            ]
        )
    if url.endswith("/data"):
        out = []
        for t in params["stationTriplets"].split(","):
            out.append(
                {
                    "stationTriplet": t,
                    "data": [
                        {
                            "stationElement": {"elementCode": "WTEQ"},
                            "values": [
                                {"date": "2020-01-30", "value": 5.0, "median": 10.0},
                                {"date": "2020-01-31", "value": 6.0, "median": 12.0},
                            ],
                        },
                        {
                            "stationElement": {"elementCode": "PREC"},
                            "values": [{"date": "2020-01-31", "value": 10.0}],
                        },
                        {
                            "stationElement": {"elementCode": "SMS", "heightDepth": -8},
                            "values": [{"date": "2020-01-31", "value": 30.0}],
                        },
                    ],
                }
            )
        return FakeResponse(out)
    if url.endswith("/daily/items"):
        site = params["monitoring_location_id"]
        value = {"USGS-10126000": "1000", "USGS-10010020": "-100", "USGS-10010100": "4190"}[site]
        features = [
            {"properties": {"time": f"2020-01-{d:02d}", "value": value, "approval_status": "A"}}
            for d in range(1, 32)
        ] + [{"properties": {"time": "2020-02-01", "value": None, "qualifier": ["Ice"]}}]
        return FakeResponse({"features": features, "links": []})
    raise AssertionError(url)


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(usgs.requests, "get", fake_get)
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE monthly_elevation (month DATE, avg_elevation DOUBLE)")
        conn.execute("INSERT INTO monthly_elevation VALUES ('2020-01-01', 4192.0)")
        cov.run_covariates(conn, {"covariates": CFG})
        yield conn


def test_sites_filtered_by_basin(db):
    rows = db.execute("SELECT station_triplet, basin FROM snotel_sites ORDER BY 1").fetchall()
    assert rows == [("1:UT:SNTL", "bear"), ("2:UT:SNTL", "provo_jordan")]


def test_snotel_rows_merge_elements(db):
    row = db.execute(
        "SELECT wteq_in, prec_in, sms_8_pct, wteq_median_in, prec_median_in FROM snotel_daily "
        "WHERE station_triplet='1:UT:SNTL' AND d='2020-01-31'"
    ).fetchone()
    assert row == (6.0, 10.0, 30.0, 12.0, None)


def test_snotel_old_schema_is_rebuilt(monkeypatch):
    monkeypatch.setattr(usgs.requests, "get", fake_get)
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE snotel_daily (station_triplet VARCHAR, d DATE, wteq_in FLOAT)")
        cov.ingest_snotel(conn, CFG["snotel"])
        cols = {r[0] for r in conn.execute("DESCRIBE snotel_daily").fetchall()}
        assert "wteq_median_in" in cols


def test_discharge_skips_null_values(db):
    (n,) = db.execute("SELECT COUNT(*) FROM usgs_discharge_daily").fetchone()
    assert n == 62


def test_monthly_covariates_month_end_and_kaf(db):
    row = db.execute(
        "SELECT swe_eom_bear, swe_eom_provo_jordan, swe_eom_weber, swe_eom_gsl, n_snotel_sites, "
        "inflow_kaf_bear FROM monthly_covariates WHERE month = DATE '2020-01-01'"
    ).fetchone()
    assert row[0] == 6.0 and row[1] == 6.0 and row[2] is None
    assert row[3] == pytest.approx(6.0) and row[4] == 2
    assert row[5] == pytest.approx(1000 * 31 * 86400 / 43560 / 1000)


def test_reservoir_storage_summed_per_basin(db):
    row = db.execute(
        "SELECT res_kaf_bear, res_kaf_weber, res_kaf_total, n_reservoirs "
        "FROM monthly_covariates WHERE month = DATE '2020-01-01'"
    ).fetchone()
    assert row == (pytest.approx(200.0), None, pytest.approx(200.0), 2)
    assert db.execute("SELECT COUNT(*) FROM reservoir_monthly").fetchone()[0] == 2


def test_nrcs_forecasts_one_row_per_exceedance(db):
    rows = db.execute(
        "SELECT publication_date, period_start, exceedance, kaf, normal_kaf "
        "FROM nrcs_inflow_forecasts ORDER BY exceedance"
    ).fetchall()
    assert len(rows) == 3
    assert rows[1][1:] == ("04-01", 50, 300.0, 450.0) and str(rows[1][0]) == "2020-02-01"


def test_percent_of_median_and_soil_moisture(db):
    row = db.execute(
        "SELECT swe_pct_median_bear, swe_pct_median_gsl, prec_pct_median_gsl, sms_eom_gsl "
        "FROM monthly_covariates WHERE month = DATE '2020-01-01'"
    ).fetchone()
    assert row[0] == pytest.approx(50.0) and row[1] == pytest.approx(50.0)
    assert row[2] is None and row[3] == pytest.approx(30.0)


def test_breach_and_north_arm_columns(db):
    row = db.execute(
        "SELECT breach_kaf, north_arm_ft, head_diff_ft, inflow_kaf_total "
        "FROM monthly_covariates WHERE month = DATE '2020-01-01'"
    ).fetchone()
    assert row[0] == pytest.approx(-100 * 31 * 86400 / 43560 / 1000)
    assert row[1] == pytest.approx(4190.0)
    assert row[2] == pytest.approx(2.0) and row[3] == pytest.approx(row[3])


def test_basin_for_huc():
    assert cov.basin_for_huc("160102", CFG["snotel"]["basins"]) == "bear"
    assert cov.basin_for_huc("170101", CFG["snotel"]["basins"]) is None
