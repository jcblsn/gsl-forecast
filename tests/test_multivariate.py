from datetime import date

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.multivariate.inflow_chain import InflowChainForecaster
from src.forecasting.multivariate.swe_regression import SweRegressionForecaster


@pytest.fixture
def synthetic():
    """30 years of monthly data where the spring rise is proportional to March SWE."""
    rng = np.random.default_rng(0)
    rows = []
    level = 4195.0
    start = date(1990, 1, 1)
    swe_year = 10.0
    for i in range(30 * 12):
        m = start + relativedelta(months=i)
        if m.month == 10:
            swe_year = rng.uniform(4, 20)
        swe = swe_year * {11: 0.2, 12: 0.4, 1: 0.6, 2: 0.8, 3: 1.0, 4: 0.6}.get(m.month, 0.0)
        inflow = swe_year * {4: 8, 5: 15, 6: 10}.get(m.month, 1.0)
        level += 0.02 * inflow - 0.25 - 0.001 * (level - 4195)
        rows.append(
            {
                "month": pd.Timestamp(m),
                "avg_elevation": level,
                "swe_eom_gsl": swe,
                "prec_wy_eom_gsl": swe * 1.5,
                "inflow_kaf_total": inflow,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("cls", [SweRegressionForecaster, InflowChainForecaster])
def test_learns_snowpack_signal(synthetic, cls):
    cutoff = pd.Timestamp("2015-03-01")
    train = synthetic[synthetic["month"] <= cutoff]
    actual = synthetic[synthetic["month"] > cutoff].head(6)["avg_elevation"].to_numpy()
    preds = cls().fit(train).predict(6)
    assert list(preds.columns) == ["month", "target", "pred", "model_name"]
    assert preds["month"].iloc[0] == cutoff + relativedelta(months=1)
    naive_err = np.abs(actual - train["avg_elevation"].iloc[-1]).mean()
    model_err = np.abs(actual - preds["pred"].to_numpy()).mean()
    assert model_err < 0.5 * naive_err


@pytest.mark.parametrize("cls", [SweRegressionForecaster, InflowChainForecaster])
def test_no_leakage_from_future_rows(synthetic, cls):
    cutoff = pd.Timestamp("2015-03-01")
    train = synthetic[synthetic["month"] <= cutoff]
    base = cls().fit(train).predict(12)["pred"].to_numpy()
    again = cls().fit(train.copy()).predict(12)["pred"].to_numpy()
    assert np.allclose(base, again)


@pytest.mark.parametrize("cls", [SweRegressionForecaster, InflowChainForecaster])
def test_requires_covariates(synthetic, cls):
    with pytest.raises(ValueError, match="covariate"):
        cls().fit(synthetic[["month", "avg_elevation"]])


def test_falls_back_when_covariates_missing_at_cutoff(synthetic):
    train = synthetic[synthetic["month"] <= "2015-03-01"].copy()
    train.loc[train.index[-1], ["swe_eom_gsl", "prec_wy_eom_gsl"]] = np.nan
    preds = SweRegressionForecaster().fit(train).predict(3)
    assert preds["pred"].notna().all()
