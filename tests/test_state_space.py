"""The state-space water balance: the path is one recursion, so it carries its own variance."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.multivariate.state_space import StateSpaceForecaster


@pytest.fixture(scope="module")
def synthetic():
    """30 years where the level follows a bucket: inflow raises it, each month loses some."""
    rng = np.random.default_rng(0)
    rows = []
    level = 4195.0
    swe_year = 10.0
    for i in range(30 * 12):
        m = date(1990, 1, 1) + relativedelta(months=i)
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


@pytest.fixture(scope="module")
def fitted(synthetic):
    return StateSpaceForecaster().fit(synthetic[synthetic["month"] <= "2015-03-01"])


def test_recovers_the_bucket(synthetic, fitted):
    """The generator raises the level 0.02 ft per unit of inflow, so the fit must find it."""
    metrics = fitted.get_metrics()
    assert metrics["converged"]
    assert metrics["inflow_ft_per_kaf"] == pytest.approx(0.02, rel=0.35)
    assert 0.9 < metrics["phi"] <= 1.0


def test_beats_a_repeat_of_the_last_value(synthetic):
    cutoff = pd.Timestamp("2015-03-01")
    train = synthetic[synthetic["month"] <= cutoff]
    actual = synthetic[synthetic["month"] > cutoff].head(12)["avg_elevation"].to_numpy()
    preds = StateSpaceForecaster().fit(train).predict(12)["pred"].to_numpy()
    naive = np.abs(actual - train["avg_elevation"].iloc[-1]).mean()
    assert np.abs(actual - preds).mean() < 0.6 * naive


def test_predict_shape_and_first_month(fitted):
    preds = fitted.predict(6)
    assert list(preds.columns) == ["month", "target", "pred", "model_name"]
    assert preds["month"].iloc[0] == pd.Timestamp("2015-04-01")
    assert preds["pred"].notna().all()


def test_the_interval_widens_with_the_lead(fitted):
    """No rule makes it widen. It widens because the state variance grows with each step."""
    q = fitted.predict_quantiles(24)
    width = q["q95"] - q["q05"]
    assert (width.diff().dropna() > 0).all()
    assert width.iloc[0] < width.iloc[-1]


def test_the_path_moves_no_faster_than_the_record(synthetic, fitted):
    """24 independent fits can step further than 1 month allows. A recursion cannot."""
    observed = synthetic[synthetic["month"] <= "2015-03-01"]["avg_elevation"].to_numpy()
    preds = fitted.predict(24)["pred"].to_numpy()
    assert np.abs(np.diff(preds)).max() <= np.abs(np.diff(observed)).max()


def test_missing_inflow_does_not_stop_the_filter(synthetic):
    train = synthetic[synthetic["month"] <= "2015-03-01"].copy()
    train.loc[train.index[:60], "inflow_kaf_total"] = np.nan
    preds = StateSpaceForecaster().fit(train).predict(6)
    assert preds["pred"].notna().all()


def test_requires_the_covariates_it_names(synthetic):
    model = StateSpaceForecaster()
    with pytest.raises(ValueError, match="covariate"):
        model.fit(synthetic.drop(columns=model.feature_columns()))


def test_feature_columns_matches_stage_one(fitted):
    assert fitted.feature_columns() == ["inflow_kaf_total", "swe_eom_gsl", "prec_wy_eom_gsl"]


def test_a_monthly_term_lands_on_its_own_month():
    """statsmodels writes alpha_(t+1) = c_t + T alpha_t, so the intercept at t drives t + 1.

    Without a shift the fit puts each monthly term 1 month early, and the forecast then
    peaks and troughs 1 month before the record does. This builds a series whose only steps
    are a rise into May and a fall into September, and requires both to land on their month.
    """
    import warnings

    from src.forecasting.multivariate.state_space import WaterBalanceSSM

    rng = np.random.default_rng(0)
    step = np.zeros(12)
    step[4], step[8] = 2.0, -1.0
    level, months, values = 100.0, [], []
    for d in pd.date_range("1980-01-01", periods=40 * 12, freq="MS"):
        level = 0.995 * (level - 100.0) + 100.0 + step[d.month - 1] + rng.normal(0, 0.01)
        months.append(d.month)
        values.append(level)
    values = np.array(values)

    model = WaterBalanceSSM(values - values.mean(), np.zeros(len(values)), np.array(months))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp=False, maxiter=300)
    monthly = model.parts(np.asarray(result.params))["monthly"]
    assert int(np.argmax(monthly)) == 4
    assert int(np.argmin(monthly)) == 8


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fitted"):
        StateSpaceForecaster().predict(6)
