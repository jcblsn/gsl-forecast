"""Run the CLI-level entry points against a synthetic DuckDB and a temp experiment db."""

import json
import os
import subprocess
import sys
from datetime import date

import duckdb
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.cross_validate import run_cross_validation
from src.forecasting.run_forecast import run_forecasts
from src.forecasting.univariate.naive import NaiveForecaster


@pytest.fixture
def project(tmp_path):
    db_path = tmp_path / "gsl.db"
    start = date(2000, 1, 1)
    rows = [
        (start + relativedelta(months=i), 4200.0 - i * 0.05 + (i % 12) * 0.1, 30)
        for i in range(240)
    ]
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE monthly_elevation "
            "(month DATE, avg_elevation DOUBLE, observation_count INT)"
        )
        conn.executemany("INSERT INTO monthly_elevation VALUES (?, ?, ?)", rows)
    config = {
        "sources": {},
        "database": {"path": str(db_path)},
        "forecasting": {
            "train_start": "2005-01-01",
            "horizon": 6,
            "experiment_db": str(tmp_path / "expt.db"),
            "output_dir": str(tmp_path / "outputs"),
            "cv": {"split": "development"},
        },
        "evaluation_policy": {
            "version": "test-v1",
            "default_split": "development",
            "splits": {
                "development": {
                    "cutoff_start": "2016-12-01",
                    "cutoff_end": "2019-06-01",
                    "horizon": 6,
                    "status": "open_development",
                },
                "confirmation": {"status": "sealed", "maximum_horizon": 6},
            },
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return {"config_path": str(config_path), "db_path": str(db_path), "tmp": tmp_path}


def test_cv_all_cutoffs_writes_parquet_and_tracker(project):
    forecasters = [NaiveForecaster(method="last"), NaiveForecaster(method="seasonal")]
    summary = run_cross_validation(
        config_path=project["config_path"], forecasters=forecasters, make_plots=False
    )
    assert set(summary["model"]) == {"naive_last", "naive_seasonal"}
    assert summary["h"].max() == 6
    parquets = list((project["tmp"] / "outputs").glob("cv_results_*.parquet"))
    assert len(parquets) == 1
    per_cutoff = pd.read_parquet(parquets[0])
    assert per_cutoff["cutoff"].nunique() == 3 * 12 - 6 + 1

    from experiment_tracker import ExperimentTracker

    with ExperimentTracker(str(project["tmp"] / "expt.db")) as tracker:
        exp = tracker.latest_experiment()
        runs = tracker.runs(experiment=exp["experiment_id"])
        assert len(runs) == 2
        assert {r["name"] for r in runs} == {"naive_last", "naive_seasonal"}
        assert all(r["status"] == "completed" for r in runs)

        at_six = tracker.metrics(experiment=exp["experiment_id"], dims={"h": 6})
        assert {"mae", "rmse", "mae_ratio"} <= {r["metric"] for r in at_six}

        # The run records how it was produced, so an id resolves to a vintage.
        assert exp["argv"] and exp["python"]
        tags = tracker.tags("experiment", exp["experiment_id"])
        assert tags["cutoff_policy"] == "development"
        assert tags["evaluation_policy_version"] == "test-v1"
        assert tags["policy_cutoff_start"] == "2016-12-01"
        assert tags["policy_cutoff_end"] == "2019-06-01"

        # The parquet is reachable from the run, so no path file is needed.
        recorded = tracker.tags("experiment", exp["experiment_id"])["cv_parquet"]
        assert os.path.basename(recorded) == parquets[0].name

        # Every prediction row is addressable, and the stored metric follows from them.
        rows = tracker.predictions(experiment=exp["experiment_id"], dims={"h": 6})
        assert rows and all(r["dims"].keys() == {"cutoff", "h"} for r in rows)
        checked = tracker.audit(runs[0]["run_id"], "mae", dims={"h": 6})
        assert checked["agrees"], checked


def test_cv_random_cutoffs_respects_n(project):
    summary = run_cross_validation(
        config_path=project["config_path"],
        n_cutoffs=4,
        forecasters=[NaiveForecaster(method="last")],
        make_plots=False,
    )
    per_cutoff = pd.read_parquet(next((project["tmp"] / "outputs").glob("cv_results_*.parquet")))
    assert per_cutoff["cutoff"].nunique() == 4
    assert len(summary) == 6


def test_run_forecasts_stores_run_identity(project):
    forecasters = [NaiveForecaster(method="last")]
    preds = run_forecasts(config_path=project["config_path"], forecasters=forecasters)
    assert len(preds) == 6
    with duckdb.connect(project["db_path"], read_only=True) as conn:
        rows = conn.execute(
            "SELECT model, run_id, experiment_id, data_max, COUNT(*) FROM forecasts GROUP BY ALL"
        ).fetchall()
    assert len(rows) == 1
    model, run_id, exp_id, data_max, n = rows[0]
    assert model == "naive_last" and run_id is not None and exp_id is not None and n == 6
    assert str(data_max) == "2019-12-01"


def test_forecasts_table_migrates_old_schema(project):
    with duckdb.connect(project["db_path"]) as conn:
        conn.execute(
            "CREATE TABLE forecasts "
            "(month DATE, prediction FLOAT, model VARCHAR, created_at TIMESTAMP)"
        )
        conn.execute("INSERT INTO forecasts VALUES ('2020-01-01', 4190.0, 'old', now())")
    run_forecasts(config_path=project["config_path"], forecasters=[NaiveForecaster(method="last")])
    with duckdb.connect(project["db_path"], read_only=True) as conn:
        n_null = conn.execute("SELECT COUNT(*) FROM forecasts WHERE run_id IS NULL").fetchone()[0]
        n_total = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    assert n_null == 1 and n_total == 7


@pytest.mark.parametrize(
    "module",
    [
        "src.pipeline.elt",
        "src.forecasting.run_forecast",
        "src.forecasting.cross_validate",
        "src.forecasting.plot_forecasts",
        "src.forecasting.view_results",
    ],
)
def test_cli_help_exits_cleanly(module):
    """Argparse must be wired through main(); --help must not run the pipeline."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_data_status_flags_thin_month_and_null_covariates(project):
    from src.forecasting.run_forecast import data_status

    meta, problems = data_status(project["db_path"])
    assert meta["data_max"] == "2019-12-01" and meta["observation_count"] == 30
    assert problems == ["null at cutoff: ['swe_eom_gsl', 'prec_wy_eom_gsl', 'head_diff_ft']"]
    with duckdb.connect(project["db_path"]) as conn:
        conn.execute(
            "UPDATE monthly_elevation SET observation_count = 3 WHERE month = '2019-12-01'"
        )
    _, problems = data_status(project["db_path"])
    assert problems[0].startswith("only 3 daily readings")


def add_column(conn, table: str, column: str, kind: str) -> None:
    """Add a column the small end-to-end fixture leaves out, so one gate can be tested."""
    present = {row[0] for row in conn.execute(f"DESCRIBE {table}").fetchall()}
    if column not in present:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")


def test_data_status_flags_an_inflow_month_the_gauges_did_not_cover(project):
    """A monthly volume built from 25 reporting days is scaled, not measured. The pipeline
    computed the coverage and no reader used it, so a scaled month reached the forecast
    without a word."""
    from src.forecasting.run_forecast import data_status

    with duckdb.connect(project["db_path"]) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS monthly_covariates (month DATE, inflow_day_coverage DOUBLE)"
        )
        add_column(conn, "monthly_covariates", "inflow_day_coverage", "DOUBLE")
        conn.execute("DELETE FROM monthly_covariates")
        conn.execute("INSERT INTO monthly_covariates VALUES (DATE '2019-12-01', 0.81)")
    meta, problems = data_status(project["db_path"])
    assert meta["inflow_day_coverage"] == pytest.approx(0.81)
    assert any("inflow gauges cover 81%" in p for p in problems)


def test_data_status_does_not_flag_a_provisional_cutoff_month(project):
    """USGS approves a month long after it ends, so the cutoff month is provisional at every
    issue. A gate on that would fire every month, and a warning that always fires is one
    nobody reads. The share is recorded instead."""
    from src.forecasting.run_forecast import data_status

    with duckdb.connect(project["db_path"]) as conn:
        add_column(conn, "monthly_elevation", "provisional_observation_count", "BIGINT")
        conn.execute(
            "UPDATE monthly_elevation SET provisional_observation_count = observation_count "
            "WHERE month = '2019-12-01'"
        )
    meta, problems = data_status(project["db_path"])
    assert meta["provisional_observation_count"] == meta["observation_count"]
    assert not any("provisional" in p for p in problems)
