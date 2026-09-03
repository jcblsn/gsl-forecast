"""Run the CLI-level entry points against a synthetic DuckDB and a temp experiment db."""

import json
import subprocess
import sys
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.blend import BLEND_MODEL, ETS_MODEL, SNOW_MODEL
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
            "headline_model": BLEND_MODEL,
            "cv": {"history_years": 3, "cutoffs": "all"},
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

    tracker = ExperimentTracker(str(project["tmp"] / "expt.db"))
    exp = tracker.list_experiments()[0]
    runs = tracker.get_run_history(exp["experiment_id"])
    assert len(runs) == 2
    metrics = tracker.get_metrics(runs[0]["run_id"])
    assert {"mae_h1", "rmse_h6", "mae_ratio_h6"} <= set(metrics)


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


def test_run_forecasts_builds_configured_headline_blend(project):
    class ComponentForecaster(NaiveForecaster):
        def __init__(self, name, offset):
            super().__init__(method="last")
            self.name = name
            self.offset = offset

        def predict(self, h, start_date=None):
            out = super().predict(h, start_date)
            out["pred"] += self.offset
            out["model_name"] = self.name
            return out

        def contributions(self, h):
            predictions = self.predict(h)
            return pd.DataFrame(
                {
                    "month": predictions["month"],
                    "h": range(1, h + 1),
                    "input": "reference_path",
                    "value": None,
                    "reference": None,
                    "contribution_ft": predictions["pred"],
                }
            )

    rows = []
    for year in range(2000, 2008):
        for issue_month in range(1, 13):
            cutoff = pd.Timestamp(year, issue_month, 1) - pd.DateOffset(months=1)
            for h in range(1, 7):
                actual = 4190.0 + h * 0.1
                rows.extend(
                    [
                        {
                            "model": SNOW_MODEL,
                            "cutoff": cutoff,
                            "h": h,
                            "pred": actual + 0.1 * h,
                            "actual": actual,
                        },
                        {
                            "model": ETS_MODEL,
                            "cutoff": cutoff,
                            "h": h,
                            "pred": actual + 0.5,
                            "actual": actual,
                        },
                    ]
                )
    cv_path = project["tmp"] / "components.parquet"
    pd.DataFrame(rows).to_parquet(cv_path, index=False)
    preds = run_forecasts(
        config_path=project["config_path"],
        forecasters=[
            ComponentForecaster(SNOW_MODEL, 0.1),
            ComponentForecaster(ETS_MODEL, 0.5),
        ],
        cv_results=str(cv_path),
    )
    assert set(preds["model_name"]) == {SNOW_MODEL, ETS_MODEL, BLEND_MODEL}
    assert preds.attrs["headline_model"] == BLEND_MODEL
    explained = preds.attrs["contributions"].groupby("h")["contribution_ft"].sum()
    blended = preds[preds["model_name"] == BLEND_MODEL].set_index(np.arange(1, 7))["pred"]
    assert np.allclose(explained, blended)


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
