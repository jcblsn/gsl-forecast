from datetime import date

import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.base import Forecaster
from src.forecasting.univariate.moving_average import MovingAverageForecaster
from src.forecasting.univariate.naive import NaiveForecaster


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
