"""The experimental structural state-space model evolves hypsometric storage."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting import hypsometry
from src.forecasting.multivariate.state_space import STORAGE_SCALE_KAF, StateSpaceForecaster


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(4)
    dates = pd.date_range("1990-01-01", periods=30 * 12, freq="MS")
    month = dates.month.to_numpy()
    innovations = rng.normal(0, 12, len(dates))
    storage = 7_000 + np.cumsum(innovations) + 180 * np.sin(2 * np.pi * (month - 3) / 12)
    return pd.DataFrame({"month": dates, "avg_elevation": hypsometry.elevation_ft(storage)})


@pytest.fixture(scope="module")
def fitted(series):
    return StateSpaceForecaster().fit(series)


def test_fits_a_standard_structural_model_on_storage(series, fitted):
    expected = np.asarray(hypsometry.volume_kaf(series["avg_elevation"])) / STORAGE_SCALE_KAF
    assert np.allclose(np.asarray(fitted._result.model.endog).ravel(), expected)
    assert fitted.get_metrics() == {
        "state": "south_arm_storage_kaf",
        "structure": "local_linear_trend_with_monthly_seasonality",
        "converged": True,
        "n_params": 2,
    }


def test_predict_shape_and_first_month(fitted):
    preds = fitted.predict(6)
    assert list(preds.columns) == ["month", "target", "pred", "model_name"]
    assert preds["month"].iloc[0] == pd.Timestamp("2020-01-01")
    assert preds["pred"].notna().all()


def test_native_interval_widens_with_lead(fitted):
    quantiles = fitted.predict_quantiles(24)
    width = quantiles["q95"] - quantiles["q05"]
    assert (width.diff().dropna() > 0).all()
    assert (quantiles["q05"] <= quantiles["q50"]).all()
    assert (quantiles["q50"] <= quantiles["q95"]).all()


def test_has_no_deterministic_future_inflow_dependency(fitted):
    assert fitted.feature_columns() == []


def test_rejects_elevations_outside_the_hypsometry_table(series):
    invalid = series.copy()
    invalid.loc[0, "avg_elevation"] = 4300.0
    with pytest.raises(ValueError, match="hypsometry domain"):
        StateSpaceForecaster().fit(invalid)


def test_requires_two_years_of_monthly_observations(series):
    with pytest.raises(ValueError, match="at least 24"):
        StateSpaceForecaster().fit(series.head(23))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fitted"):
        StateSpaceForecaster().predict(6)
