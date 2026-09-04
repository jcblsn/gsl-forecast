from datetime import date, datetime
from io import BytesIO

import duckdb
import openpyxl
import pytest

from src.pipeline import brine, climate, usgs, weather
from src.pipeline import covariates as cov

CFG = {
    "snotel": {
        "states": ["UT"],
        "basins": {"1601": "bear", "160201": "weber", "160202": "provo_jordan"},
        "start": "2020-01-01",
    },
    "reservoirs": {"states": ["UT"], "start": "2020-01-01"},
    "nrcs_forecasts": {"station": "10010000:UT:USGS", "start": "2020-01-01"},
    "climdiv": {"state": "42", "divisions": ["03", "05"]},
    "usgs_discharge": {
        "inflow": {"bear": "10126000"},
        "exchange": {"breach": "10010020"},
        "start": "2020-01-01",
    },
    "north_arm": {"site": "10010100", "start": "2020-01-01"},
    "kslc": {"station": "USW00024127", "start": "2020-01-01"},
    "brine": {"primary_site": "AS2", "start": "2020-01-01"},
}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload

    @property
    def text(self):
        return self.payload

    @property
    def content(self):
        return self.payload


CLIMDIV_LISTING = "climdiv-tmpcdv-v1.0.0-20200206 climdiv-pcpndv-v1.0.0-20200206"
CLIMDIV_LINE = "{}{}2020  30.00  32.00" + " -99.90" * 10


def brine_workbook() -> bytes:
    """A 2-campaign AS2 sheet, so the brine reader can be exercised without the 4 MB file."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "AS2"
    sheet.append(
        ["SITE", "DATE", "DEPTH-FT", "Salinity EOS (g/L)", "LAB-DEN\n(g/cm3)", "LK-ELEV (feet)"]
    )
    sheet.append(["AS2", datetime(2020, 1, 15), 3, 120.0, 1.09, 4193.0])
    sheet.append(["AS2", datetime(2020, 1, 15), 25, 180.0, 1.14, 4193.0])
    sheet.append(["AS2", datetime(2020, 7, 15), 3, 140.0, 1.10, 4192.0])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


KSLC_CSV = (
    '"STATION","DATE","AWND","PRCP","TMAX","TMIN"\n'
    '"USW00024127","2020-01-01","   30","    0","  100","   20"\n'
)


def fake_get(url, params=None, timeout=None):
    if url == brine.UGS_WORKBOOK:
        return FakeResponse(brine_workbook())
    if url.startswith(weather.NCEI_DAILY):
        return FakeResponse(KSLC_CSV)
    if url == climate.CLIMDIV:
        return FakeResponse(CLIMDIV_LISTING)
    if url.startswith(climate.CLIMDIV):
        element = "tmpc" if "tmpc" in url else "pcpn"
        code = {"tmpc": "02", "pcpn": "01"}[element]
        return FakeResponse(
            "\n".join(CLIMDIV_LINE.format(s, code) for s in ("4203", "4205", "4201", "0801"))
        )
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


def seed_elevation(conn) -> None:
    """The elevation tables the covariate rollup joins to; the pipeline builds these first."""
    conn.execute(
        "CREATE TABLE monthly_elevation (month DATE, avg_elevation DOUBLE, elevation_eom_ft DOUBLE)"
    )
    conn.execute("INSERT INTO monthly_elevation VALUES ('2020-01-01', 4192.0, 4192.1)")
    conn.execute(
        "CREATE TABLE usgs_water_surface_elevation_daily "
        "(d DATE, elevation FLOAT, qualifiers VARCHAR)"
    )
    conn.execute(
        "INSERT INTO usgs_water_surface_elevation_daily VALUES "
        "('2020-01-15', 4192.0, 'approved'), ('2020-07-15', 4191.0, 'approved')"
    )


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(usgs.requests, "get", fake_get)
    monkeypatch.setattr(brine.requests, "get", fake_get)
    with duckdb.connect(":memory:") as conn:
        seed_elevation(conn)
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
    monkeypatch.setattr(brine.requests, "get", fake_get)
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


def test_climdiv_mean_of_divisions(db):
    row = db.execute(
        "SELECT AVG(tavg_f), AVG(prcp_in) FROM climdiv_monthly WHERE month = DATE '2020-01-01'"
    ).fetchone()
    assert row == (pytest.approx(30.0), pytest.approx(30.0))
    assert db.execute("SELECT COUNT(*) FROM climdiv_monthly").fetchone()[0] == 4


def test_monthly_covariates_hides_the_unlagged_climate_columns(db):
    """A model reads this table, and the unlagged month does not exist at issue time."""
    columns = {r[0] for r in db.execute("DESCRIBE monthly_covariates").fetchall()}
    assert "tavg_f_gsl" not in columns and "prcp_in_gsl" not in columns
    assert "tavg_f_gsl_lag1" in columns and "prcp_in_gsl_lag1" in columns


def test_climdiv_lag_columns_shift_one_month(db):
    """NOAA publishes a month around the 8th of the next month, so only the lag is safe."""
    row = db.execute(
        "SELECT tavg_f_gsl_lag1, prcp_in_gsl_lag1 FROM monthly_covariates "
        "WHERE month = DATE '2020-02-01'"
    ).fetchone()
    assert row == (pytest.approx(30.0), pytest.approx(30.0))


def test_climdiv_ingest_twice_replaces(db):
    climate.ingest_climdiv(db, CFG["climdiv"])
    assert db.execute("SELECT COUNT(*) FROM climdiv_monthly").fetchone()[0] == 4


def test_climdiv_latest_file_and_parse():
    assert climate.latest_file(
        "x climdiv-tmpcdv-v1.0.0-20250101 climdiv-tmpcdv-v1.0.0-20250201", "tmpc"
    )
    assert climate.latest_file(
        CLIMDIV_LISTING + " climdiv-tmpcdv-v1.0.0-20210101", "tmpc"
    ).endswith("20210101")
    rows = climate.parse_climdiv(CLIMDIV_LINE.format("4203", "02"), "42", ["03"], -99.9)
    assert rows == [("2020-01-01", "03", 30.0), ("2020-02-01", "03", 32.0)]
    line = "4203012020  1.50 -9.99" + " -9.99" * 10
    assert climate.parse_climdiv(line, "42", ["03"], -9.99) == [("2020-01-01", "03", 1.5)]


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


ROSTER = {
    "version": "test-roster-v1",
    "basin_weights": {"bear": 0.7, "provo_jordan": 0.3},
    "stations": {"bear": ["1:UT:SNTL"], "provo_jordan": ["2:UT:SNTL"]},
}


def roster_cfg(**overrides):
    snotel = {**CFG["snotel"], "roster": {**ROSTER, **overrides}}
    return {**CFG, "snotel": snotel}


def test_roster_defaults_to_the_sites_discovered_today(db):
    rows = db.execute("SELECT * FROM snotel_roster ORDER BY station_triplet").fetchall()
    assert [r[:3] for r in rows] == [
        ("discovered-active", "1:UT:SNTL", "bear"),
        ("discovered-active", "2:UT:SNTL", "provo_jordan"),
    ]
    assert [r[3] for r in rows] == [pytest.approx(0.5), pytest.approx(0.5)]


def test_configured_roster_names_its_version_and_weights(monkeypatch):
    monkeypatch.setattr(usgs.requests, "get", fake_get)
    monkeypatch.setattr(brine.requests, "get", fake_get)
    with duckdb.connect(":memory:") as conn:
        cov.ingest_snotel(conn, roster_cfg()["snotel"])
        rows = conn.execute("SELECT * FROM snotel_roster ORDER BY station_triplet").fetchall()
    assert rows == [
        ("test-roster-v1", "1:UT:SNTL", "bear", pytest.approx(0.7)),
        ("test-roster-v1", "2:UT:SNTL", "provo_jordan", pytest.approx(0.3)),
    ]


def test_features_ignore_a_site_the_roster_left_out(monkeypatch):
    """A site an earlier run left in snotel_sites must not reach monthly_covariates."""
    monkeypatch.setattr(usgs.requests, "get", fake_get)
    monkeypatch.setattr(brine.requests, "get", fake_get)
    cfg = roster_cfg(stations={"bear": ["1:UT:SNTL"]}, basin_weights={"bear": 1.0})
    with duckdb.connect(":memory:") as conn:
        seed_elevation(conn)
        cov.run_covariates(conn, {"covariates": cfg})
        row = conn.execute(
            "SELECT n_snotel_sites, swe_eom_provo_jordan, snotel_roster_version "
            "FROM monthly_covariates WHERE month = DATE '2020-01-01'"
        ).fetchone()
    assert row == (1, None, "test-roster-v1")


def test_roster_rejects_a_station_awdb_does_not_return(monkeypatch):
    monkeypatch.setattr(usgs.requests, "get", fake_get)
    monkeypatch.setattr(brine.requests, "get", fake_get)
    cfg = roster_cfg(stations={**ROSTER["stations"], "bear": ["1:UT:SNTL", "99:UT:SNTL"]})
    with duckdb.connect(":memory:") as conn:
        with pytest.raises(ValueError, match="99:UT:SNTL"):
            cov.ingest_snotel(conn, cfg["snotel"])


def test_roster_rejects_weights_that_do_not_sum_to_one(monkeypatch):
    monkeypatch.setattr(usgs.requests, "get", fake_get)
    monkeypatch.setattr(brine.requests, "get", fake_get)
    cfg = roster_cfg(basin_weights={"bear": 0.7, "provo_jordan": 0.7})
    with duckdb.connect(":memory:") as conn:
        with pytest.raises(ValueError, match="sum to 1"):
            cov.ingest_snotel(conn, cfg["snotel"])


def stub_db(daily_rows, roster_rows):
    """A database with hand-written rows, so one aggregation rule can be checked alone."""
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE snotel_daily (station_triplet VARCHAR, d DATE, wteq_in FLOAT, "
        "prec_in FLOAT, sms_8_pct FLOAT, wteq_median_in FLOAT, prec_median_in FLOAT)"
    )
    if daily_rows:
        conn.executemany("INSERT INTO snotel_daily VALUES (?, ?, ?, ?, ?, ?, ?)", daily_rows)
    conn.execute(
        "CREATE TABLE snotel_roster (roster_version VARCHAR, station_triplet VARCHAR, "
        "basin VARCHAR, basin_weight DOUBLE)"
    )
    if roster_rows:
        conn.executemany("INSERT INTO snotel_roster VALUES (?, ?, ?, ?)", roster_rows)
    conn.execute(
        "CREATE TABLE usgs_discharge_daily (site_id VARCHAR, d DATE, discharge_cfs FLOAT,"
        " qualifiers VARCHAR)"
    )
    conn.execute(
        "CREATE TABLE reservoir_monthly (station_triplet VARCHAR, month DATE, storage_kaf FLOAT)"
    )
    conn.execute("CREATE TABLE reservoir_sites (station_triplet VARCHAR, basin VARCHAR)")
    conn.execute(
        "CREATE TABLE reservoir_roster (roster_version VARCHAR, station_triplet VARCHAR, "
        "basin VARCHAR)"
    )
    conn.execute(f"CREATE TABLE {cov.NORTH_ARM_TABLE} (d DATE, elevation DOUBLE)")
    conn.execute(
        "CREATE TABLE climdiv_monthly (month DATE, division VARCHAR, tavg_f DOUBLE, prcp_in DOUBLE)"
    )
    conn.execute(
        "CREATE TABLE monthly_elevation (month DATE, avg_elevation DOUBLE, elevation_eom_ft DOUBLE)"
    )
    conn.execute(
        f"CREATE TABLE {weather.KSLC_TABLE} (d DATE, tmax_c DOUBLE, tmin_c DOUBLE, "
        "wind_mps DOUBLE, prcp_in DOUBLE)"
    )
    conn.execute(
        "CREATE TABLE gsl_salt_mass_monthly (month DATE, salt_mass_mt DOUBLE, "
        "salt_mass_age_days DOUBLE)"
    )
    brine.materialize_hypsometry(conn)
    return conn


def test_month_end_takes_the_last_valid_day_in_the_window():
    """A site that misses the last day of the month still reports a month-end value."""
    rows = [
        ("1:UT:SNTL", "2020-01-29", 4.0, None, None, 10.0, None),
        ("1:UT:SNTL", "2020-01-30", 5.0, None, None, 10.0, None),
        ("2:UT:SNTL", "2020-01-31", 7.0, None, None, 10.0, None),
    ]
    with stub_db(rows, [("v", "1:UT:SNTL", "bear", 1.0)]) as conn:
        cov.transform_covariates(conn, CFG["usgs_discharge"], ["bear"])
        row = conn.execute("SELECT swe_eom_bear, n_snotel_sites FROM monthly_covariates").fetchone()
    assert row == (pytest.approx(5.0), 1)


def test_pooled_columns_use_the_declared_basin_weights():
    """Weber has 1 site and Bear has 3, but the weights and not the counts set the index."""
    rows = [("b1", "2020-01-31", 3.0, None, None, None, None)]
    rows += [(f"a{i}", "2020-01-31", 9.0, None, None, None, None) for i in range(3)]
    roster = [("v", "b1", "weber", 0.25)] + [("v", f"a{i}", "bear", 0.75) for i in range(3)]
    with stub_db(rows, roster) as conn:
        cov.transform_covariates(conn, CFG["usgs_discharge"], ["bear", "weber"])
        row = conn.execute("SELECT swe_eom_gsl, n_snotel_sites FROM monthly_covariates").fetchone()
    assert row == (pytest.approx(0.75 * 9.0 + 0.25 * 3.0), 4)


def test_each_variable_counts_its_own_reporting_sites():
    """Soil moisture came from 1 site per basin; the SWE count must not weight its average."""
    rows = [
        ("a", "2020-01-31", 9.0, None, 40.0, None, None),
        ("b", "2020-01-31", 9.0, None, None, None, None),
        ("c", "2020-01-31", 3.0, None, 20.0, None, None),
    ]
    roster = [("v", "a", "bear", 0.5), ("v", "b", "bear", 0.5), ("v", "c", "weber", 0.5)]
    with stub_db(rows, roster) as conn:
        cov.transform_covariates(conn, CFG["usgs_discharge"], ["bear", "weber"])
        row = conn.execute(
            "SELECT sms_eom_gsl, n_snotel_sites, n_snotel_sms FROM monthly_covariates"
        ).fetchone()
    assert row == (pytest.approx(30.0), 3, 2)


def flow_db(daily_rows):
    conn = stub_db([], [])
    conn.executemany("INSERT INTO usgs_discharge_daily VALUES (?, ?, ?, 'Approved')", daily_rows)
    return conn


def test_a_partial_month_of_discharge_is_scaled_and_declares_its_coverage():
    """A 28-day sum is not a 31-day volume, and a reader must be able to see the gap."""
    rows = [("10126000", f"2020-01-{d:02d}", 1000.0) for d in range(1, 29)]
    with flow_db(rows) as conn:
        cov.transform_covariates(conn, CFG["usgs_discharge"], ["bear"])
        row = conn.execute(
            "SELECT inflow_kaf_bear, inflow_day_coverage FROM monthly_covariates"
        ).fetchone()
    assert row[0] == pytest.approx(1000 * 31 * 86400 / 43560 / 1000)
    assert row[1] == pytest.approx(28 / 31)


def test_a_month_below_the_day_threshold_is_dropped():
    rows = [("10126000", f"2020-01-{d:02d}", 1000.0) for d in range(1, cov.MIN_FLOW_DAYS)]
    with flow_db(rows) as conn:
        cov.transform_covariates(conn, CFG["usgs_discharge"], ["bear"])
        assert conn.execute("SELECT COUNT(*) FROM monthly_covariates").fetchone() == (0,)


def test_provisional_and_estimated_days_reach_the_modelled_table():
    """USGS stores the approval and the qualifier; a model could not see either."""
    rows = [("10126000", f"2020-01-{d:02d}", 1000.0, "Approved") for d in range(1, 26)]
    rows += [("10126000", "2020-01-26", 1000.0, "Provisional")]
    rows += [("10126000", f"2020-01-{d:02d}", 1000.0, "Approved,ESTIMATED") for d in (27, 28)]
    conn = stub_db([], [])
    conn.executemany("INSERT INTO usgs_discharge_daily VALUES (?, ?, ?, ?)", rows)
    with conn:
        cov.transform_covariates(conn, CFG["usgs_discharge"], ["bear"])
        row = conn.execute(
            "SELECT inflow_provisional_days, inflow_estimated_days, inflow_day_coverage "
            "FROM monthly_covariates"
        ).fetchone()
    assert row[:2] == (1, 2) and row[2] == pytest.approx(28 / 31)


def test_reservoir_roster_rejects_a_station_awdb_does_not_return():
    """Storage summed over whichever reservoirs answered today is a different quantity every
    run. The station count in the record moves from 1 to 21 to 19 for that reason."""
    conn = duckdb.connect(":memory:")
    cfg = {"roster": {"version": "v1", "stations": {"bear": ["A:UT:BOR", "GONE:UT:BOR"]}}}
    with pytest.raises(ValueError, match="did not return these roster reservoirs"):
        cov.create_reservoir_roster(conn, [{"station_triplet": "A:UT:BOR", "basin": "bear"}], cfg)


def test_reservoir_roster_rejects_a_station_in_another_basin():
    """A reservoir counted in the wrong basin moves storage between basins silently."""
    conn = duckdb.connect(":memory:")
    cfg = {"roster": {"version": "v1", "stations": {"bear": ["A:UT:BOR"]}}}
    sites = [{"station_triplet": "A:UT:BOR", "basin": "weber"}]
    with pytest.raises(ValueError, match="sit in another basin"):
        cov.create_reservoir_roster(conn, sites, cfg)


def test_reservoir_roster_pins_the_station_set_and_its_version():
    """The roster version must travel with the feature, so a later reader can tell which
    basket of reservoirs a stored value summed."""
    conn = duckdb.connect(":memory:")
    cfg = {"roster": {"version": "gsl-modern-complete-v1", "stations": {"bear": ["A:UT:BOR"]}}}
    sites = [
        {"station_triplet": "A:UT:BOR", "basin": "bear"},
        {"station_triplet": "NEW:UT:BOR", "basin": "bear"},
    ]
    triplets = cov.create_reservoir_roster(conn, sites, cfg)
    assert triplets == ["A:UT:BOR"]
    rows = conn.execute("SELECT roster_version, station_triplet FROM reservoir_roster").fetchall()
    assert rows == [("gsl-modern-complete-v1", "A:UT:BOR")]


def test_the_gauged_total_and_the_lake_delivery_estimate_are_separate_columns():
    """The 3 gauges are terminal gauges and do not measure the whole delivery to the lake.
    Rescaling the column the models already read would change what their coefficients mean."""
    conn = stub_db([], [])
    conn.executemany(
        "INSERT INTO usgs_discharge_daily VALUES (?, ?, ?, ?)",
        [
            (site, date(2020, 1, day), 100.0, "approved")
            for site in ("11", "22", "33")
            for day in range(1, 32)
        ],
    )
    cov.transform_covariates(
        conn,
        {"inflow": {"bear": "11", "weber": "22", "jordan": "33"}},
        ["bear"],
        delivery_ratio=0.8,
    )
    row = conn.execute(
        "SELECT inflow_kaf_total, inflow_kaf_lake, n_inflow_gauges FROM monthly_covariates"
    ).fetchone()
    assert row[0] == pytest.approx(row[1] * 0.8)
    assert row[2] == 3


def test_a_missing_gauge_nulls_the_total_but_not_the_reported_sum():
    """One gauge outage used to null the total for 72 months. The partial sum stays available
    under its own name, with the gauge count beside it, so a reader can see what it holds."""
    conn = stub_db([], [])
    conn.executemany(
        "INSERT INTO usgs_discharge_daily VALUES (?, ?, ?, ?)",
        [
            (site, date(2020, 1, day), 100.0, "approved")
            for site in ("11", "22")
            for day in range(1, 32)
        ],
    )
    cov.transform_covariates(
        conn, {"inflow": {"bear": "11", "weber": "22", "jordan": "33"}}, ["bear"]
    )
    row = conn.execute(
        "SELECT inflow_kaf_total, inflow_kaf_reported, n_inflow_gauges FROM monthly_covariates"
    ).fetchone()
    assert row[0] is None
    assert row[1] > 0
    assert row[2] == 2


def test_ice_affected_discharge_days_are_counted():
    """Ice at the Bear River gauge makes a winter value an estimate. That reaches the water
    balance directly, so the count must travel with the feature."""
    conn = stub_db([], [])
    conn.executemany(
        "INSERT INTO usgs_discharge_daily VALUES (?, ?, ?, ?)",
        [
            ("11", date(2020, 1, day), 100.0, "approved,estimated,ice" if day < 5 else "approved")
            for day in range(1, 32)
        ],
    )
    cov.transform_covariates(conn, {"inflow": {"bear": "11"}}, ["bear"])
    row = conn.execute(
        "SELECT inflow_ice_days, inflow_estimated_days FROM monthly_covariates"
    ).fetchone()
    assert row == (4, 4)
