import json
import os

import duckdb
import pandas as pd
import pytest

from src.site.build import build_site, headlines, latest_issue, render, verification
from src.site.chart import fan_chart


@pytest.fixture
def issue(tmp_path):
    """One dated forecast with intervals, its sidecar, and a matching elevation database."""
    months = pd.date_range("2026-09-01", periods=24, freq="MS")
    rows = []
    for model, level in (("blend", 4190.0), ("swe_regression", 4190.5)):
        for h, month in enumerate(months, start=1):
            pred = level + 0.1 * h
            rows.append(
                {
                    "month": month.date(),
                    "model": model,
                    "pred": pred,
                    "issue": "2026-09-01",
                    "h": h,
                    "q05": pred - 1,
                    "q25": pred - 0.5,
                    "q50": pred,
                    "q75": pred + 0.5,
                    "q95": pred + 1,
                }
            )
    forecast_dir = tmp_path / "forecasts"
    forecast_dir.mkdir()
    pd.DataFrame(rows).to_csv(forecast_dir / "2026-09.csv", index=False)
    meta = {
        "data_max": "2026-08-01",
        "observation_count": 31,
        "n_snotel_sites": 55,
        "missing_covariates": [],
    }
    (forecast_dir / "2026-09.meta.json").write_text(json.dumps(meta))

    history = pd.DataFrame(
        {
            "month": pd.date_range("2024-09-01", periods=24, freq="MS"),
            "avg_elevation": [4189.0 + 0.05 * i for i in range(24)],
        }
    )
    db = tmp_path / "gsl.db"
    with duckdb.connect(str(db)) as conn:
        conn.register("history", history)
        conn.execute("CREATE TABLE monthly_elevation AS SELECT * FROM history")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"database": {"path": str(db)}}))
    return {"dir": str(forecast_dir), "config": str(config), "tmp": tmp_path}


def test_latest_issue_reads_the_sidecar(issue):
    path, frame, meta = latest_issue(issue["dir"])
    assert path.endswith("2026-09.csv")
    assert meta["n_snotel_sites"] == 55
    assert set(frame["model"]) == {"blend", "swe_regression"}


def test_headlines_pick_the_next_peak_and_september(issue):
    _, frame, _ = latest_issue(issue["dir"])
    head = headlines(frame, "blend")
    assert head["peak"]["month"] == pd.Timestamp("2027-06-01")
    assert head["wy_end"]["month"] == pd.Timestamp("2026-09-01")
    assert head["peak"]["band"][0] < head["peak"]["value"] < head["peak"]["band"][1]


def test_build_writes_a_page_with_the_headline_and_the_vintage(issue):
    out = os.path.join(issue["tmp"], "site", "index.html")
    build_site(issue["config"], issue["dir"], out, history_years=100)
    html = open(out).read()
    assert "Great Salt Lake elevation forecast" in html
    assert "Headline model: <code>blend</code>" in html
    assert "Last complete month" in html
    assert "2026-08-01" in html
    assert "No issued forecast has reached its target month yet" in html
    assert "<svg" in html and "polyline" in html


def test_page_notes_a_missing_sidecar(issue):
    os.remove(os.path.join(issue["dir"], "2026-09.meta.json"))
    out = os.path.join(issue["tmp"], "site", "no_meta.html")
    build_site(issue["config"], issue["dir"], out, history_years=100)
    assert "no vintage sidecar" in open(out).read()


def test_verification_table_reaches_the_page(issue):
    pd.DataFrame(
        {"model": ["blend"], "h": [3], "mae": [0.21], "bias": [-0.05], "n": [1], "cov90": [1.0]}
    ).to_csv(os.path.join(issue["dir"], "verification.csv"), index=False)
    assert verification(os.path.join(issue["dir"], "verification.csv")) is not None
    out = os.path.join(issue["tmp"], "site", "verified.html")
    build_site(issue["config"], issue["dir"], out, history_years=100)
    html = open(out).read()
    assert "0.21" in html and "cov90" in html


def test_fan_chart_falls_back_when_no_intervals(issue):
    _, frame, _ = latest_issue(issue["dir"])
    history = pd.DataFrame(
        {
            "month": pd.date_range("2025-09-01", periods=12, freq="MS"),
            "avg_elevation": [4189.0] * 12,
        }
    )
    svg = fan_chart(history, frame.drop(columns=["q05", "q95"]), "blend")
    assert "polygon" not in svg
    assert svg.startswith("<svg")


def test_render_falls_back_to_an_available_model(issue):
    _, frame, meta = latest_issue(issue["dir"])
    history = pd.DataFrame(
        {
            "month": pd.date_range("2025-09-01", periods=12, freq="MS"),
            "avg_elevation": [4189.0] * 12,
        }
    )
    html = render(frame[frame["model"] != "blend"], history, meta, None, model="blend")
    assert "Headline model: <code>swe_regression</code>" in html
