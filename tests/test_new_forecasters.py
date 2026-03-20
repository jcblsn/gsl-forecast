from datetime import date, datetime

import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.univariate.drift import DriftForecaster
from src.forecasting.univariate.exponential_smoothing import HoltWintersForecaster
from src.forecasting.univariate.theta import ThetaForecaster


@pytest.fixture
def sample_data():
    """36 months of synthetic elevation data with trend and seasonality."""
    start = date(2020, 1, 1)
    rows = []
    for i in range(36):
        month = start + relativedelta(months=i)
        # downward trend + seasonal bump in spring
        value = 4200.0 - i * 0.5 + (3.0 if month.month in (4, 5, 6) else 0.0)
        rows.append({"month": month, "avg_elevation": value})
    return pd.DataFrame(rows)


# ── DriftForecaster ──────────────────────────────────────────────────────────

class TestDriftForecaster:
    def test_predicts_correct_slope(self, sample_data):
        f = DriftForecaster(window=12)
        f.fit(sample_data)
        preds = f.predict(h=3)
        assert len(preds) == 3
        assert "pred" in preds.columns
        # slope should be negative (downward trend in fixture)
        assert preds["pred"].iloc[2] < preds["pred"].iloc[0]

    def test_slope_calculation(self):
        df = pd.DataFrame({
            "month": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            "avg_elevation": [100.0, 99.0, 98.0],
        })
        f = DriftForecaster(window=24)
        f.fit(df)
        # slope = -1.0 per period
        assert abs(f.slope - (-1.0)) < 1e-9
        preds = f.predict(h=2)
        assert abs(preds["pred"].iloc[0] - 97.0) < 1e-6
        assert abs(preds["pred"].iloc[1] - 96.0) < 1e-6

    def test_window_larger_than_data_warns(self, caplog, sample_data):
        small = sample_data.head(5)
        f = DriftForecaster(window=24)
        import logging
        with caplog.at_level(logging.WARNING):
            f.fit(small)
        assert "only" in caplog.text.lower() or len(caplog.records) > 0

    def test_raises_on_too_little_data(self):
        df = pd.DataFrame({"month": [date(2020, 1, 1)], "avg_elevation": [100.0]})
        f = DriftForecaster(window=5)
        with pytest.raises(ValueError, match="at least 2"):
            f.fit(df)

    def test_is_fitted_flag(self, sample_data):
        f = DriftForecaster(window=12)
        assert not f.is_fitted
        f.fit(sample_data)
        assert f.is_fitted

    def test_predict_before_fit_raises(self):
        f = DriftForecaster(window=12)
        with pytest.raises(RuntimeError):
            f.predict(h=3)

    def test_output_shape_and_columns(self, sample_data):
        f = DriftForecaster(window=12)
        f.fit(sample_data)
        preds = f.predict(h=6)
        assert len(preds) == 6
        assert set(preds.columns) >= {"month", "pred", "model_name"}


# ── HoltWintersForecaster ────────────────────────────────────────────────────

class TestHoltWintersForecaster:
    def test_additive_seasonal_fit_predict(self, sample_data):
        f = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12)
        f.fit(sample_data)
        assert f.is_fitted
        preds = f.predict(h=12)
        assert len(preds) == 12
        assert not preds["pred"].isna().any()

    def test_damped_trend_fit_predict(self, sample_data):
        f = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12, damped_trend=True)
        f.fit(sample_data)
        preds = f.predict(h=6)
        assert len(preds) == 6

    def test_no_seasonal_fit_predict(self, sample_data):
        f = HoltWintersForecaster(trend="add", seasonal=None)
        f.fit(sample_data)
        preds = f.predict(h=3)
        assert len(preds) == 3

    def test_name_encodes_config(self):
        f_damped = HoltWintersForecaster(trend="add", seasonal="add", damped_trend=True)
        f_plain = HoltWintersForecaster(trend="add", seasonal="add", damped_trend=False)
        assert "damped" in f_damped.name
        assert "damped" not in f_plain.name

    def test_predict_before_fit_raises(self):
        f = HoltWintersForecaster()
        with pytest.raises(RuntimeError):
            f.predict(h=3)

    def test_output_shape_and_columns(self, sample_data):
        f = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12)
        f.fit(sample_data)
        preds = f.predict(h=6)
        assert set(preds.columns) >= {"month", "pred", "model_name"}
        assert len(preds) == 6

    def test_get_metrics_returns_config(self):
        f = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12, damped_trend=True)
        metrics = f.get_metrics()
        assert metrics["damped_trend"] is True
        assert metrics["seasonal_periods"] == 12


# ── ThetaForecaster ──────────────────────────────────────────────────────────

class TestThetaForecaster:
    def test_fit_predict(self, sample_data):
        f = ThetaForecaster()
        f.fit(sample_data)
        assert f.is_fitted
        preds = f.predict(h=6)
        assert len(preds) == 6
        assert not preds["pred"].isna().any()

    def test_captures_downward_trend(self, sample_data):
        """Theta should project below last observed value for a downward-trending series."""
        f = ThetaForecaster()
        f.fit(sample_data)
        last_val = sample_data["avg_elevation"].iloc[-1]
        preds = f.predict(h=12)
        # At least the long-horizon predictions should be below last value
        assert preds["pred"].iloc[-1] < last_val

    def test_predict_before_fit_raises(self):
        f = ThetaForecaster()
        with pytest.raises(RuntimeError):
            f.predict(h=3)

    def test_output_shape_and_columns(self, sample_data):
        f = ThetaForecaster()
        f.fit(sample_data)
        preds = f.predict(h=4)
        assert set(preds.columns) >= {"month", "pred", "model_name"}
        assert len(preds) == 4

    def test_get_metrics_includes_alpha(self, sample_data):
        f = ThetaForecaster()
        f.fit(sample_data)
        metrics = f.get_metrics()
        assert "alpha" in metrics
        assert 0 < metrics["alpha"] < 1
