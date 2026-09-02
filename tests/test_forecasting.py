from datetime import date

import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.base import Forecaster
from src.forecasting.registry import all_forecasters, production_forecasters
from src.forecasting.univariate.drift import DriftForecaster
from src.forecasting.univariate.exponential_smoothing import HoltWintersForecaster
from src.forecasting.univariate.moving_average import MovingAverageForecaster
from src.forecasting.univariate.naive import NaiveForecaster
from src.forecasting.univariate.theta import ThetaForecaster


@pytest.fixture
def sample_data():
    # 24 months of sample data
    dates = []
    values = []

    start_date = date(2023, 1, 1)
    for i in range(24):
        current_date = start_date + relativedelta(months=i)
        dates.append(current_date)
        # base + trend + seasonal component
        values.append(4190 + i * 0.1 + (5 * (i % 12 == 6)))

    return pd.DataFrame({"month": dates, "avg_elevation": values})


class TestForecasterBase:
    """Test common behavior across all forecasters"""

    def test_forecaster_interface(self, sample_data):
        """Test that all forecasters follow the common interface"""
        forecasters = [
            NaiveForecaster(method="last"),
            NaiveForecaster(method="seasonal", seasonal_period=12),
            MovingAverageForecaster(window=3),
        ]

        for f in forecasters:
            # Test the interface
            assert isinstance(f, Forecaster)
            assert f.name is not None
            assert callable(f.fit)
            assert callable(f.predict)
            assert callable(f.get_metrics)

            # Test the fit-predict workflow
            f.fit(sample_data)
            assert f.is_fitted

            # Basic prediction shape checks
            h = 6
            preds = f.predict(h=h)
            assert len(preds) == h
            assert "month" in preds.columns
            assert "pred" in preds.columns
            assert "model_name" in preds.columns


class TestNaiveForecaster:
    def test_naive_last_value_correctness(self, sample_data):
        """Test that last-value forecast returns the last value"""
        forecaster = NaiveForecaster(method="last")
        forecaster.fit(sample_data)

        last_value = sample_data["avg_elevation"].iloc[-1]
        preds = forecaster.predict(h=3)

        assert all(preds["pred"] == last_value)

    def test_naive_seasonal_correctness(self, sample_data):
        """Test that seasonal forecast repeats seasonal patterns"""
        # Use period 6 for a shorter test
        period = 6
        forecaster = NaiveForecaster(method="seasonal", seasonal_period=period)
        forecaster.fit(sample_data)

        preds = forecaster.predict(h=period)

        # Check seasonal pattern is repeated
        for i in range(period):
            expected = sample_data["avg_elevation"].iloc[-(period - i)]
            assert preds["pred"].iloc[i] == expected


class TestMovingAverageForecaster:
    def test_moving_average_correctness(self, sample_data):
        """Test that MA forecast equals average of last N values"""
        window = 3
        forecaster = MovingAverageForecaster(window=window)
        forecaster.fit(sample_data)

        expected_ma = sample_data["avg_elevation"].tail(window).mean()
        preds = forecaster.predict(h=3)

        assert all(abs(preds["pred"] - expected_ma) < 1e-10)

    def test_window_size_boundary(self, sample_data):
        """Test MA forecaster with small and large window sizes"""
        # Test with window = data size
        window = len(sample_data)
        forecaster = MovingAverageForecaster(window=window)
        forecaster.fit(sample_data)

        expected_ma = sample_data["avg_elevation"].mean()
        preds = forecaster.predict(h=1)

        assert abs(preds["pred"].iloc[0] - expected_ma) < 1e-10


@pytest.fixture
def trend_data():
    """36 months of synthetic elevation data with a downward trend and spring bump."""
    start = date(2020, 1, 1)
    rows = []
    for i in range(36):
        month = start + relativedelta(months=i)
        value = 4200.0 - i * 0.5 + (3.0 if month.month in (4, 5, 6) else 0.0)
        rows.append({"month": month, "avg_elevation": value})
    return pd.DataFrame(rows)


# ── DriftForecaster ──────────────────────────────────────────────────────────


class TestDriftForecaster:
    def test_predicts_correct_slope(self, trend_data):
        f = DriftForecaster(window=12)
        f.fit(trend_data)
        preds = f.predict(h=3)
        assert len(preds) == 3
        assert "pred" in preds.columns
        # slope should be negative (downward trend in fixture)
        assert preds["pred"].iloc[2] < preds["pred"].iloc[0]

    def test_slope_calculation(self):
        df = pd.DataFrame(
            {
                "month": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
                "avg_elevation": [100.0, 99.0, 98.0],
            }
        )
        f = DriftForecaster(window=24)
        f.fit(df)
        # slope = -1.0 per period
        assert abs(f.slope - (-1.0)) < 1e-9
        preds = f.predict(h=2)
        assert abs(preds["pred"].iloc[0] - 97.0) < 1e-6
        assert abs(preds["pred"].iloc[1] - 96.0) < 1e-6

    def test_window_larger_than_data_warns(self, caplog, trend_data):
        small = trend_data.head(5)
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

    def test_is_fitted_flag(self, trend_data):
        f = DriftForecaster(window=12)
        assert not f.is_fitted
        f.fit(trend_data)
        assert f.is_fitted

    def test_predict_before_fit_raises(self):
        f = DriftForecaster(window=12)
        with pytest.raises(RuntimeError):
            f.predict(h=3)

    def test_output_shape_and_columns(self, trend_data):
        f = DriftForecaster(window=12)
        f.fit(trend_data)
        preds = f.predict(h=6)
        assert len(preds) == 6
        assert set(preds.columns) >= {"month", "pred", "model_name"}


# ── HoltWintersForecaster ────────────────────────────────────────────────────


class TestHoltWintersForecaster:
    def test_additive_seasonal_fit_predict(self, trend_data):
        f = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12)
        f.fit(trend_data)
        assert f.is_fitted
        preds = f.predict(h=12)
        assert len(preds) == 12
        assert not preds["pred"].isna().any()

    def test_damped_trend_fit_predict(self, trend_data):
        f = HoltWintersForecaster(
            trend="add", seasonal="add", seasonal_periods=12, damped_trend=True
        )
        f.fit(trend_data)
        preds = f.predict(h=6)
        assert len(preds) == 6

    def test_no_seasonal_fit_predict(self, trend_data):
        f = HoltWintersForecaster(trend="add", seasonal=None)
        f.fit(trend_data)
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

    def test_output_shape_and_columns(self, trend_data):
        f = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12)
        f.fit(trend_data)
        preds = f.predict(h=6)
        assert set(preds.columns) >= {"month", "pred", "model_name"}
        assert len(preds) == 6

    def test_get_metrics_returns_config(self):
        f = HoltWintersForecaster(
            trend="add", seasonal="add", seasonal_periods=12, damped_trend=True
        )
        metrics = f.get_metrics()
        assert metrics["damped_trend"] is True
        assert metrics["seasonal_periods"] == 12


# ── ThetaForecaster ──────────────────────────────────────────────────────────


class TestThetaForecaster:
    def test_fit_predict(self, trend_data):
        f = ThetaForecaster()
        f.fit(trend_data)
        assert f.is_fitted
        preds = f.predict(h=6)
        assert len(preds) == 6
        assert not preds["pred"].isna().any()

    def test_captures_downward_trend(self, trend_data):
        """Theta should project below last observed value for a downward-trending series."""
        f = ThetaForecaster()
        f.fit(trend_data)
        last_val = trend_data["avg_elevation"].iloc[-1]
        preds = f.predict(h=12)
        # At least the long-horizon predictions should be below last value
        assert preds["pred"].iloc[-1] < last_val

    def test_predict_before_fit_raises(self):
        f = ThetaForecaster()
        with pytest.raises(RuntimeError):
            f.predict(h=3)

    def test_output_shape_and_columns(self, trend_data):
        f = ThetaForecaster()
        f.fit(trend_data)
        preds = f.predict(h=4)
        assert set(preds.columns) >= {"month", "pred", "model_name"}
        assert len(preds) == 4

    def test_get_metrics_includes_alpha(self, trend_data):
        f = ThetaForecaster()
        f.fit(trend_data)
        metrics = f.get_metrics()
        assert "alpha" in metrics
        assert 0 < metrics["alpha"] < 1


class TestThetaAnchoring:
    def test_h1_close_to_last_level_on_long_nonstationary_series(self):
        """Theta must anchor at the current level, not at a global OLS line.
        A 40-year series with a level break makes the OLS line sit far from the last value."""
        start = date(1980, 1, 1)
        rows = []
        for i in range(480):
            level = 4205.0 if i < 300 else 4193.0
            rows.append({"month": start + relativedelta(months=i), "avg_elevation": level})
        df = pd.DataFrame(rows)
        preds = ThetaForecaster().fit(df).predict(h=12)
        assert abs(preds["pred"].iloc[0] - 4193.0) < 0.5


class TestRegistry:
    def test_names_unique_and_fresh_instances(self):
        names = [f.name for f in all_forecasters()]
        assert len(names) == len(set(names))
        assert all_forecasters()[0] is not all_forecasters()[0]

    def test_production_is_subset(self):
        all_names = {f.name for f in all_forecasters()}
        prod = production_forecasters()
        assert prod and {f.name for f in prod} <= all_names
        assert "naive_last" in {f.name for f in prod}
