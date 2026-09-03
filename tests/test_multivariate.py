from datetime import date

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.multivariate.blend import (
    BlendForecaster,
    default_weights,
    monotone_decreasing,
)
from src.forecasting.multivariate.inflow_chain import InflowChainForecaster
from src.forecasting.multivariate.regression import fallback_reason, gcv_alpha, ridge_fit
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


def test_ridge_matches_least_squares_with_a_tiny_penalty():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(60), rng.normal(size=(60, 2))])
    y = 3.0 + 2.0 * X[:, 1] - 1.5 * X[:, 2] + rng.normal(scale=0.1, size=60)
    beta = ridge_fit(X, y, alpha=1e-10)
    expected = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(beta, expected, atol=1e-6)


def test_ridge_predictions_are_invariant_to_feature_scale():
    rng = np.random.default_rng(2)
    X = np.column_stack([np.ones(60), rng.normal(size=(60, 2))])
    y = 4192.0 + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=60)
    scaled = X.copy()
    scaled[:, 1] *= 1000.0
    plain = X @ ridge_fit(X, y, alpha=1.0)
    rescaled = scaled @ ridge_fit(scaled, y, alpha=1.0)
    assert np.allclose(plain, rescaled, atol=1e-8)


def test_gcv_picks_a_larger_penalty_for_pure_noise():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, 3))])
    signal = 4192.0 + 5.0 * X[:, 1]
    noise = 4192.0 + rng.normal(scale=1.0, size=40)
    assert gcv_alpha(*_centered(X, noise)) > gcv_alpha(*_centered(X, signal))


def _centered(X, y):
    features = X[:, 1:]
    z = (features - features.mean(axis=0)) / features.std(axis=0)
    return z, y - y.mean()


def test_fallback_reason_states_the_rule():
    assert fallback_reason(20, 10, True) is None
    assert "NULL at the cutoff" in fallback_reason(20, 10, False)
    assert "min_obs=10" in fallback_reason(4, 10, True)


def _blend(**kw):
    return BlendForecaster(horizon=12, history_years=8, max_cutoffs=15, min_cutoffs=6, **kw)


def test_blend_weights_fall_with_the_lead(synthetic):
    model = _blend().fit(synthetic[synthetic["month"] <= pd.Timestamp("2015-03-01")])
    w = model.weights
    assert len(w) == 12
    assert ((w >= 0) & (w <= 1)).all()
    assert (np.diff(w) <= 1e-9).all()
    assert model.n_weight_cutoffs >= 6


def test_blend_uses_no_rows_after_the_cutoff(synthetic):
    cutoff = pd.Timestamp("2015-03-01")
    train = synthetic[synthetic["month"] <= cutoff]
    base = _blend().fit(train)
    tampered = synthetic.copy()
    future = tampered["month"] > cutoff
    tampered.loc[future, "avg_elevation"] += 25.0
    tampered.loc[future, "swe_eom_gsl"] *= 3.0
    again = _blend().fit(tampered[tampered["month"] <= cutoff])
    assert np.allclose(base.weights, again.weights)
    assert np.allclose(base.predict(12)["pred"], again.predict(12)["pred"])


def test_blend_lies_between_its_components(synthetic):
    train = synthetic[synthetic["month"] <= pd.Timestamp("2015-03-01")]
    model = _blend().fit(train)
    snow, univariate = (f.predict(12)["pred"].to_numpy() for f in model._fitted)
    blended = model.predict(12)["pred"].to_numpy()
    assert (blended >= np.minimum(snow, univariate) - 1e-9).all()
    assert (blended <= np.maximum(snow, univariate) + 1e-9).all()


def test_monotone_decreasing_pools_violators():
    out = monotone_decreasing(np.array([0.2, 0.8, 0.5]))
    assert (np.diff(out) <= 1e-9).all()
    assert np.isclose(out.mean(), 0.5)


def test_default_weights_ramp_from_one_to_zero():
    w = default_weights(24)
    assert w[0] == 1.0 and w[5] == 1.0
    assert w[-1] == 0.0
    assert (np.diff(w) <= 1e-9).all()
