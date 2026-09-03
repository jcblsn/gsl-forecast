from datetime import date

import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.benchmark import compare
from src.forecasting.cross_validate import evaluate_at_cutoff
from src.forecasting.headline import (
    APR_JUN_MONTHLY_MEAN_MAX,
    SEPTEMBER_MONTHLY_MEAN,
    TARGET_LABELS,
    headline_metrics,
    headline_scores,
    issue_month,
    summarize_headline,
)
from src.forecasting.univariate.naive import NaiveForecaster


@pytest.fixture
def seasonal_data():
    """Six years of monthly data peaking in May and bottoming in October."""
    start = date(2015, 1, 1)
    rows = []
    for i in range(72):
        m = start + relativedelta(months=i)
        rows.append(
            {
                "month": pd.Timestamp(m),
                "avg_elevation": 4195.0
                + [0, 0.5, 1, 1.5, 2, 1.8, 1.2, 0.6, 0.2, 0, 0.1, 0.1][m.month - 1],
            }
        )
    return pd.DataFrame(rows)


def test_issue_month_is_month_after_cutoff():
    assert issue_month(pd.Timestamp("2024-12-01")) == 1
    assert issue_month(pd.Timestamp("2025-03-01")) == 4


def test_december_cutoff_scores_peak_and_wy_end(seasonal_data):
    cutoff = pd.Timestamp("2019-12-01")
    cv = evaluate_at_cutoff(seasonal_data, cutoff, [NaiveForecaster(method="last")], horizon=12)
    scores = headline_scores(cv, seasonal_data)
    peak = scores[scores["target"] == APR_JUN_MONTHLY_MEAN_MAX].iloc[0]
    wy = scores[scores["target"] == SEPTEMBER_MONTHLY_MEAN].iloc[0]
    assert peak["issue"] == "jan" and peak["water_year"] == 2020
    assert peak["actual"] == pytest.approx(4197.0)
    assert peak["pred"] == pytest.approx(4195.1)
    assert wy["actual"] == pytest.approx(4195.2)


def test_april_cutoff_uses_observed_april(seasonal_data):
    cutoff = pd.Timestamp("2020-04-01")
    cv = evaluate_at_cutoff(seasonal_data, cutoff, [NaiveForecaster(method="last")], horizon=12)
    scores = headline_scores(cv, seasonal_data)
    peak = scores[scores["target"] == APR_JUN_MONTHLY_MEAN_MAX].iloc[0]
    assert peak["issue"] == "may"
    assert peak["pred"] == pytest.approx(4196.5)
    assert peak["actual"] == pytest.approx(4197.0)


def test_summer_cutoffs_score_only_the_water_year_end(seasonal_data):
    cutoff = pd.Timestamp("2020-07-01")
    cv = evaluate_at_cutoff(seasonal_data, cutoff, [NaiveForecaster(method="last")], horizon=12)
    scores = headline_scores(cv, seasonal_data)
    assert list(scores["target"]) == [SEPTEMBER_MONTHLY_MEAN]
    assert scores.iloc[0]["issue"] == "aug"
    cutoff = pd.Timestamp("2020-08-01")
    cv = evaluate_at_cutoff(seasonal_data, cutoff, [NaiveForecaster(method="last")], horizon=12)
    assert headline_scores(cv, seasonal_data).empty


def test_summary_and_metric_names(seasonal_data):
    frames = [
        evaluate_at_cutoff(seasonal_data, c, [NaiveForecaster(method="last")], horizon=12)
        for c in (pd.Timestamp("2018-12-01"), pd.Timestamp("2019-12-01"))
    ]
    scores = headline_scores(pd.concat(frames), seasonal_data)
    summary = summarize_headline(scores)
    assert set(summary["n"]) == {2}
    logged = list(headline_metrics(summary, "naive_last"))
    assert {(d["target"], d["issue"]) for _, d in logged} == {
        (APR_JUN_MONTHLY_MEAN_MAX, "jan"),
        (SEPTEMBER_MONTHLY_MEAN, "jan"),
    }
    assert all(set(values) == {"mae", "n"} for values, _ in logged)
    assert TARGET_LABELS == {
        APR_JUN_MONTHLY_MEAN_MAX: "Maximum April–June monthly mean",
        SEPTEMBER_MONTHLY_MEAN: "September mean (water-year end)",
    }


def test_compare_joins_on_issue_and_year():
    headline = pd.DataFrame(
        {
            "model": ["m"],
            "issue": ["apr"],
            "water_year": [2025],
            "target": [APR_JUN_MONTHLY_MEAN_MAX],
            "pred": [4193.9],
            "actual": [4193.5],
            "abs_error": [0.4],
        }
    )
    nrcs = pd.DataFrame(
        {"issue_date": ["2025-04-01"], "implied_peak_ft": [4194.0], "actual_peak_ft": [4193.6]}
    )
    out = compare(headline, nrcs, model="m")
    assert out.loc[0, "our_model"] == "m"
    assert out.loc[0, "nrcs_error"] == pytest.approx(0.4)
    assert out.loc[0, "nrcs_error_vs_apr_jun_monthly_mean_max"] == pytest.approx(0.5)


def test_compare_merges_issued_inflow():
    headline = pd.DataFrame(columns=["model", "issue", "water_year", "target", "pred", "actual"])
    nrcs = pd.DataFrame(
        {"issue_date": ["2025-04-01"], "implied_peak_ft": [4194.0], "actual_peak_ft": [4193.6]}
    )
    inflow = pd.DataFrame(
        {"issue_date": [pd.Timestamp("2025-04-01").date()], "nrcs_inflow_p50_kaf": [590.0]}
    )
    out = compare(headline, nrcs, model="m", inflow=inflow)
    assert (
        out.loc[0, "nrcs_inflow_p50_kaf"] == 590.0
        and "our_apr_jun_monthly_mean_max" not in out.columns
    )
