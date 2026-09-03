from datetime import date

import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.config import load_config
from src.forecasting.cross_validate import (
    evaluate_at_cutoff,
    require_empty_snapshot_target,
    resolve_evaluation_split,
    summarize,
)
from src.forecasting.cutoffs import policy_cutoffs, sample_cutoffs, valid_cutoffs
from src.forecasting.univariate.naive import NaiveForecaster


@pytest.fixture
def monthly_data():
    """10 years of monthly data."""
    start = date(2010, 1, 1)
    rows = [
        {"month": pd.Timestamp(start + relativedelta(months=i)), "avg_elevation": 4200.0 - i * 0.1}
        for i in range(120)
    ]
    return pd.DataFrame(rows)


class TestCutoffs:
    def test_all_cutoffs_leave_room_for_horizon(self, monthly_data):
        cutoffs = valid_cutoffs(monthly_data, history_years=8, horizon=12)
        latest = monthly_data["month"].max()
        assert cutoffs == sorted(cutoffs)
        assert cutoffs[-1] == latest - relativedelta(months=12)
        assert cutoffs[0] >= latest - relativedelta(years=8)
        assert len(cutoffs) == 8 * 12 - 12 + 1

    def test_none_returns_all(self, monthly_data):
        assert sample_cutoffs(monthly_data, None, 8, 12) == valid_cutoffs(monthly_data, 8, 12)

    def test_random_sample_is_seeded_and_sorted(self, monthly_data):
        c1 = sample_cutoffs(monthly_data, n=5, history_years=8, horizon=12, seed=99)
        c2 = sample_cutoffs(monthly_data, n=5, history_years=8, horizon=12, seed=99)
        c3 = sample_cutoffs(monthly_data, n=5, history_years=8, horizon=12, seed=1)
        assert len(c1) == 5 and c1 == c2 == sorted(c1) and c1 != c3

    def test_raises_if_not_enough_valid_cutoffs(self, monthly_data):
        with pytest.raises(ValueError, match="valid cutoffs"):
            sample_cutoffs(monthly_data, n=500, history_years=1, horizon=12)

    def test_policy_cutoffs_use_exact_bounds(self, monthly_data):
        cutoffs = policy_cutoffs(monthly_data, "2017-01-01", "2018-12-01", horizon=12)
        assert cutoffs == list(pd.date_range("2017-01-01", "2018-12-01", freq="MS"))


def test_default_policy_is_the_frozen_development_cohort():
    config = load_config()
    name, split, horizon = resolve_evaluation_split(config)
    assert name == "development"
    assert split["cutoff_start"] == "2011-08-01"
    assert split["cutoff_end"] == "2024-08-01"
    assert split["status"] == "open_development"
    assert horizon == 24


def test_sealed_confirmation_split_is_rejected():
    with pytest.raises(ValueError, match="sealed"):
        resolve_evaluation_split(load_config(), "limited_confirmation")


def test_snapshot_target_must_be_empty(tmp_path):
    target = tmp_path / "snapshot"
    target.mkdir()
    require_empty_snapshot_target(str(target))
    (target / "existing.csv").write_text("do not replace\n")
    with pytest.raises(ValueError, match="new, empty directory"):
        require_empty_snapshot_target(str(target))


class TestEvaluateAtCutoff:
    def test_returns_correct_structure(self, monthly_data):
        cutoff = monthly_data["month"].iloc[60]
        result = evaluate_at_cutoff(
            monthly_data, cutoff, [NaiveForecaster(method="last")], horizon=6
        )
        assert set(result.columns) >= {
            "model",
            "cutoff",
            "target_month",
            "h",
            "pred",
            "actual",
            "abs_error",
            "sq_error",
        }
        assert len(result) == 6
        assert (result["h"] == list(range(1, 7))).all()
        assert result["target_month"].tolist() == list(
            pd.date_range(cutoff + pd.DateOffset(months=1), periods=6, freq="MS")
        )

    def test_actuals_are_after_cutoff(self, monthly_data):
        cutoff = monthly_data["month"].iloc[60]
        result = evaluate_at_cutoff(
            monthly_data, cutoff, [NaiveForecaster(method="last")], horizon=6
        )
        expected = monthly_data[monthly_data["month"] > cutoff]["avg_elevation"].head(6).tolist()
        assert result["actual"].tolist() == expected

    def test_missing_target_month_fails_instead_of_shifting_actuals(self, monthly_data):
        cutoff = monthly_data["month"].iloc[60]
        missing = cutoff + pd.DateOffset(months=2)
        incomplete = monthly_data[monthly_data["month"] != missing]

        with pytest.raises(ValueError, match=f"Missing actual target month.*{missing.date()}"):
            evaluate_at_cutoff(
                incomplete, cutoff, [NaiveForecaster(method="last")], horizon=6
            )

    def test_forecaster_target_months_must_match_leads(self, monthly_data):
        class ShiftedForecaster(NaiveForecaster):
            def predict(self, h, start_date=None):
                out = super().predict(h, start_date)
                out["month"] += pd.DateOffset(months=1)
                return out

        cutoff = monthly_data["month"].iloc[60]
        with pytest.raises(ValueError, match="target months that do not match"):
            evaluate_at_cutoff(monthly_data, cutoff, [ShiftedForecaster()], horizon=6)

    def test_train_start_restricts_training(self, monthly_data):
        cutoff = monthly_data["month"].iloc[60]

        class CapturingForecaster(NaiveForecaster):
            def fit(self, data):
                self.fit_data_len = len(data)
                return super().fit(data)

        f_full = CapturingForecaster(method="last")
        f_restricted = CapturingForecaster(method="last")
        evaluate_at_cutoff(monthly_data, cutoff, [f_full], horizon=6, train_start=None)
        evaluate_at_cutoff(
            monthly_data, cutoff, [f_restricted], horizon=6, train_start="2015-01-01"
        )
        assert f_restricted.fit_data_len < f_full.fit_data_len

    def test_failed_forecaster_excluded_from_results(self, monthly_data):
        class FailingForecaster(NaiveForecaster):
            def fit(self, data):
                raise RuntimeError("intentional failure")

        cutoff = monthly_data["month"].iloc[60]
        bad = FailingForecaster(method="last")
        bad.name = "failing_model"
        result = evaluate_at_cutoff(
            monthly_data, cutoff, [NaiveForecaster(method="last"), bad], horizon=3
        )
        assert "failing_model" not in result["model"].values
        assert len(result) == 3

    def test_error_metrics_are_correct(self, monthly_data):
        cutoff = monthly_data["month"].iloc[60]
        result = evaluate_at_cutoff(
            monthly_data, cutoff, [NaiveForecaster(method="last")], horizon=3
        )
        err = result["pred"] - result["actual"]
        assert (result["abs_error"] - err.abs()).abs().max() < 1e-9
        assert (result["sq_error"] - err**2).abs().max() < 1e-9


class TestSummarize:
    def test_ratio_is_one_for_baseline(self, monthly_data):
        cutoff = monthly_data["month"].iloc[60]
        forecasters = [NaiveForecaster(method="last"), NaiveForecaster(method="seasonal")]
        cv_df = evaluate_at_cutoff(monthly_data, cutoff, forecasters, horizon=3)
        summary = summarize(cv_df)
        base = summary[summary["model"] == "naive_last"]
        assert (base["mae_ratio"] - 1.0).abs().max() < 1e-12
        assert set(summary.columns) >= {"model", "h", "mae", "rmse", "mae_ratio"}


def test_non_finite_predictions_are_dropped(caplog):
    import numpy as np
    import pandas as pd

    from src.forecasting.cross_validate import evaluate_at_cutoff
    from src.forecasting.univariate.naive import NaiveForecaster

    class NanForecaster(NaiveForecaster):
        def predict(self, h, start_date=None):
            out = super().predict(h, start_date)
            out["pred"] = np.nan
            return out

    months = pd.date_range("2000-01-01", periods=48, freq="MS")
    data = pd.DataFrame({"month": months, "avg_elevation": 4190.0 + np.arange(48) * 0.01})
    bad = NanForecaster(method="last")
    bad.name = "nan_model"
    out = evaluate_at_cutoff(data, months[30], [NaiveForecaster(method="last"), bad], 6)
    assert set(out["model"]) == {"naive_last"}
