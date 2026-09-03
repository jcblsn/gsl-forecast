import numpy as np
import pandas as pd
import pytest

from src.forecasting.univariate.endpoint_seasonal import EndpointSeasonalForecaster


def endpoint_data(periods: int = 15 * 12) -> pd.DataFrame:
    months = pd.date_range("2000-01-01", periods=periods, freq="MS")
    level = 4190.0 + 0.01 * np.arange(periods) + 0.4 * np.sin(2 * np.pi * months.month / 12)
    noise = np.where(np.arange(periods) % 2, 0.5, -0.5)
    return pd.DataFrame(
        {
            "month": months,
            "avg_elevation": level,
            "last_elevation": level + noise,
            "endpoint_3d_median": level + noise / 4,
            "endpoint_7d_median": level,
        }
    )


def test_selects_endpoint_with_lowest_expanding_error():
    model = EndpointSeasonalForecaster().fit(endpoint_data())

    assert model.anchor_column == "endpoint_7d_median"
    assert model.anchor_selection_n > 12
    assert model.anchor_selection_mae == pytest.approx(0.0, abs=1e-10)


def test_forecast_uses_endpoint_plus_same_issue_month_change():
    data = endpoint_data()
    model = EndpointSeasonalForecaster().fit(data)
    forecast = model.predict(2)

    expected = data["endpoint_7d_median"].iloc[-1] + np.median(
        data["avg_elevation"].iloc[13::12].to_numpy()
        - data["endpoint_7d_median"].iloc[11:-2:12].to_numpy()
    )
    assert forecast["pred"].iloc[1] == pytest.approx(expected)
    assert forecast["month"].tolist() == list(
        pd.date_range(data["month"].iloc[-1] + pd.DateOffset(months=1), periods=2, freq="MS")
    )


def test_monthly_mean_is_a_compatible_fallback():
    data = endpoint_data().drop(
        columns=["last_elevation", "endpoint_3d_median", "endpoint_7d_median"]
    )
    model = EndpointSeasonalForecaster(min_obs=2).fit(data)

    assert model.anchor_column == "avg_elevation"
    assert np.isfinite(model.predict(6)["pred"]).all()


def test_analogs_take_the_origins_whose_level_was_nearest():
    """The change from a 4,190 ft origin is not the change from a 4,200 ft origin."""
    months, levels = [], []
    for year in range(1990, 2020):
        for month in range(1, 13):
            months.append(pd.Timestamp(year=year, month=month, day=1))
            # A level that alternates between a low regime and a high regime by year, and a
            # January-to-February change that depends on which regime the origin sits in.
            base = 4190.0 if year % 2 else 4200.0
            levels.append(base + (1.0 if month == 2 and base == 4190.0 else 0.0))
    data = pd.DataFrame({"month": months, "avg_elevation": levels})
    january = data[data["month"] <= pd.Timestamp("2019-01-01")]

    everything = EndpointSeasonalForecaster(min_obs=2).fit(january).predict(1)["pred"].iloc[0]
    analog = (
        EndpointSeasonalForecaster(min_obs=2, n_analogs=5, name="endpoint_analog")
        .fit(january)
        .predict(1)["pred"]
        .iloc[0]
    )
    assert january["avg_elevation"].iloc[-1] == pytest.approx(4190.0)
    assert analog == pytest.approx(4191.0)
    assert everything < analog


def test_the_analog_count_travels_with_the_fit():
    data = pd.DataFrame(
        {
            "month": pd.date_range("2000-01-01", periods=60, freq="MS"),
            "avg_elevation": np.linspace(4190.0, 4195.0, 60),
        }
    )
    assert EndpointSeasonalForecaster().fit(data).get_metrics()["n_analogs"] == "all"
    assert EndpointSeasonalForecaster(n_analogs=8).fit(data).get_metrics()["n_analogs"] == 8
