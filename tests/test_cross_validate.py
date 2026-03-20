import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.cross_validate import evaluate_at_cutoff, sample_cutoffs
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


class TestSampleCutoffs:
    def test_returns_correct_count(self, monthly_data):
        cutoffs = sample_cutoffs(monthly_data, n=5, history_years=8, horizon=12, seed=42)
        assert len(cutoffs) == 5

    def test_cutoffs_within_bounds(self, monthly_data):
        horizon = 12
        history_years = 8
        cutoffs = sample_cutoffs(monthly_data, n=5, history_years=history_years, horizon=horizon, seed=42)

        latest_month = monthly_data["month"].max()
        for c in cutoffs:
            assert c <= latest_month - relativedelta(months=horizon)

    def test_cutoffs_are_sorted(self, monthly_data):
        cutoffs = sample_cutoffs(monthly_data, n=5, history_years=8, horizon=12, seed=42)
        assert cutoffs == sorted(cutoffs)

    def test_reproducible_with_same_seed(self, monthly_data):
        c1 = sample_cutoffs(monthly_data, n=5, history_years=8, horizon=12, seed=99)
        c2 = sample_cutoffs(monthly_data, n=5, history_years=8, horizon=12, seed=99)
        assert c1 == c2

    def test_different_seeds_give_different_cutoffs(self, monthly_data):
        c1 = sample_cutoffs(monthly_data, n=5, history_years=8, horizon=12, seed=1)
        c2 = sample_cutoffs(monthly_data, n=5, history_years=8, horizon=12, seed=2)
        assert c1 != c2

    def test_raises_if_not_enough_valid_cutoffs(self, monthly_data):
        with pytest.raises(ValueError, match="valid cutoffs"):
            sample_cutoffs(monthly_data, n=500, history_years=1, horizon=12, seed=42)


class TestEvaluateAtCutoff:
    def test_returns_correct_structure(self, monthly_data):
        cutoff = monthly_data["month"].iloc[60]
        forecasters = [NaiveForecaster(method="last")]
        result = evaluate_at_cutoff(monthly_data, cutoff, forecasters, horizon=6)

        assert set(result.columns) >= {"model", "cutoff", "h", "pred", "actual", "abs_error", "sq_error"}
        assert len(result) == 6
        assert (result["h"] == list(range(1, 7))).all()

    def test_actuals_are_after_cutoff(self, monthly_data):
        cutoff = monthly_data["month"].iloc[60]
        forecasters = [NaiveForecaster(method="last")]
        result = evaluate_at_cutoff(monthly_data, cutoff, forecasters, horizon=6)
        assert (result["actual"] != monthly_data.loc[monthly_data["month"] == cutoff, "avg_elevation"].values[0]).any()

    def test_train_start_restricts_training(self, monthly_data):
        """With train_start, the forecaster should only see data from that date onward."""
        cutoff = monthly_data["month"].iloc[60]
        # Use a forecaster that we can introspect
        class CapturingForecaster(NaiveForecaster):
            def fit(self, data):
                self.fit_data_len = len(data)
                return super().fit(data)

        f_full = CapturingForecaster(method="last")
        f_restricted = CapturingForecaster(method="last")

        evaluate_at_cutoff(monthly_data, cutoff, [f_full], horizon=6, train_start=None)
        evaluate_at_cutoff(monthly_data, cutoff, [f_restricted], horizon=6, train_start="2015-01-01")

        assert f_restricted.fit_data_len < f_full.fit_data_len

    def test_failed_forecaster_excluded_from_results(self, monthly_data):
        class FailingForecaster(NaiveForecaster):
            def fit(self, data):
                raise RuntimeError("intentional failure")

        cutoff = monthly_data["month"].iloc[60]
        good = NaiveForecaster(method="last")
        bad = FailingForecaster(method="last")
        bad.name = "failing_model"

        result = evaluate_at_cutoff(monthly_data, cutoff, [good, bad], horizon=3)
        assert "failing_model" not in result["model"].values
        assert len(result) == 3  # only good model's predictions

    def test_error_metrics_are_correct(self, monthly_data):
        cutoff = monthly_data["month"].iloc[60]
        forecasters = [NaiveForecaster(method="last")]
        result = evaluate_at_cutoff(monthly_data, cutoff, forecasters, horizon=3)

        for _, row in result.iterrows():
            assert abs(row["abs_error"] - abs(row["pred"] - row["actual"])) < 1e-9
            assert abs(row["sq_error"] - (row["pred"] - row["actual"]) ** 2) < 1e-9
