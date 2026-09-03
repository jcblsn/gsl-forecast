from datetime import date

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.multivariate.blend import _CACHE as blend_cache
from src.forecasting.multivariate.blend import (
    SEASON_MONTHS,
    BlendForecaster,
    covariate_share,
    default_weights,
    issue_season,
    monotone_weight_path,
    simplex_grid,
)
from src.forecasting.multivariate.inflow_chain import InflowChainForecaster
from src.forecasting.multivariate.regression import fallback_reason, gcv_alpha, ridge_fit
from src.forecasting.multivariate.swe_regression import SweRegressionForecaster
from src.forecasting.univariate.exponential_smoothing import HoltWintersForecaster


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
                "head_diff_ft": 0.5 + 0.02 * (level - 4195.0),
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
def test_fit_is_deterministic(synthetic, cls):
    """The same rows give the same forecast. Leakage itself is tested in test_leakage.py."""
    cutoff = pd.Timestamp("2015-03-01")
    train = synthetic[synthetic["month"] <= cutoff]
    base = cls().fit(train).predict(12)["pred"].to_numpy()
    again = cls().fit(train.copy()).predict(12)["pred"].to_numpy()
    assert np.allclose(base, again)


@pytest.mark.parametrize("cls", [SweRegressionForecaster, InflowChainForecaster])
def test_feature_columns_names_every_covariate_the_model_reads(synthetic, cls):
    """The availability test in test_leakage.py reads this list, so it must be complete."""
    model = cls()
    columns = model.feature_columns()
    assert columns
    stripped = synthetic.drop(columns=columns)
    with pytest.raises((ValueError, KeyError)):
        model.fit(stripped)


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


def _centered(X, y):
    features = X[:, 1:]
    z = (features - features.mean(axis=0)) / features.std(axis=0)
    return z, y - y.mean()


def test_gcv_picks_a_larger_penalty_for_pure_noise():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, 3))])
    signal = 4192.0 + 5.0 * X[:, 1]
    noise = 4192.0 + rng.normal(scale=1.0, size=40)
    assert gcv_alpha(*_centered(X, noise)) > gcv_alpha(*_centered(X, signal))


def test_fallback_reason_states_the_rule():
    assert fallback_reason(20, 10, True) is None
    assert "NULL at the cutoff" in fallback_reason(20, 10, False)
    assert "min_obs=10" in fallback_reason(4, 10, True)


def _blend(**kw):
    return BlendForecaster(horizon=12, history_years=10, max_cutoffs=60, min_rows=4, **kw)


def test_issue_seasons_cover_every_month():
    assert set().union(*SEASON_MONTHS.values()) == set(range(1, 13))
    assert issue_season(pd.Timestamp("2025-12-01")) == "accumulation"
    assert issue_season(pd.Timestamp("2025-03-01")) == "melt"
    assert issue_season(pd.Timestamp("2025-08-01")) == "recession"


def test_blend_weights_fall_with_the_lead_in_every_fitted_season(synthetic):
    model = _blend().fit(synthetic[synthetic["month"] <= pd.Timestamp("2015-03-01")])
    assert model.fitted_seasons
    for season in model.fitted_seasons:
        w = model.weights[season]
        assert w.shape == (12, 2)
        assert ((w >= 0) & (w <= 1)).all()
        assert np.allclose(w.sum(axis=1), 1.0)
        assert (np.diff(1.0 - w[:, -1]) <= 1e-9).all()


def test_blend_lies_between_its_components(synthetic):
    train = synthetic[synthetic["month"] <= pd.Timestamp("2015-03-01")]
    model = _blend().fit(train)
    snow, univariate = (f.predict(12)["pred"].to_numpy() for f in model._fitted)
    blended = model.predict(12)["pred"].to_numpy()
    assert (blended >= np.minimum(snow, univariate) - 1e-9).all()
    assert (blended <= np.maximum(snow, univariate) + 1e-9).all()


def test_blend_selects_the_curve_for_the_issue_season(synthetic):
    model = _blend().fit(synthetic[synthetic["month"] <= pd.Timestamp("2015-03-01")])
    model.weights["accumulation"] = np.tile([0.0, 1.0], (12, 1))
    model.weights["melt"] = np.tile([1.0, 0.0], (12, 1))
    snow, univariate = (f.predict(12)["pred"].to_numpy() for f in model._fitted)
    assert issue_season(pd.Timestamp("2015-03-01")) == "melt"
    assert np.allclose(model.predict(12)["pred"], snow)
    december = model.predict(12, start_date=pd.Timestamp("2015-12-01"))
    assert issue_season(pd.Timestamp("2015-12-01")) == "accumulation"
    assert np.allclose(december["pred"], univariate)


@pytest.mark.parametrize("k, step", [(2, 0.01), (3, 0.05)])
def test_monotone_path_beats_a_per_lead_fit_on_the_stated_objective(k, step):
    """The constrained search must minimise the loss it claims, not project a free fit."""
    rng = np.random.default_rng(7)
    grid = simplex_grid(k, step)
    share = covariate_share(grid)
    loss = rng.random((6, len(grid)))
    path = monotone_weight_path(loss, share)
    assert (np.diff(share[path]) <= 1e-9).all()
    chosen = sum(loss[lead, j] for lead, j in enumerate(path))
    free = sorted(loss.argmin(axis=1), key=lambda j: -share[j])
    projected = sum(loss[lead, j] for lead, j in enumerate(free))
    assert chosen <= projected + 1e-9


def test_simplex_grid_covers_the_simplex_in_share_order():
    for k, step in ((2, 0.01), (3, 0.05)):
        grid = simplex_grid(k, step)
        assert np.allclose(grid.sum(axis=1), 1.0)
        assert (grid >= 0).all()
        assert (np.diff(covariate_share(grid)) >= -1e-12).all()
        assert len(set(map(tuple, np.round(grid, 6)))) == len(grid)


@pytest.mark.parametrize("k", [2, 3])
def test_default_weights_ramp_from_one_to_zero(k):
    w = default_weights(24, k)
    share = 1.0 - w[:, -1]
    assert share[0] == 1.0 and share[5] == 1.0
    assert share[-1] == 0.0
    assert (np.diff(share) <= 1e-9).all()
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(w[:, :-1], (share / (k - 1))[:, None])


def test_snowpack_contributions_sum_to_the_prediction(synthetic):
    cutoff = pd.Timestamp("2015-03-01")
    model = SweRegressionForecaster().fit(synthetic[synthetic["month"] <= cutoff])
    predictions = model.predict(6).set_index("month")["pred"]
    explained = model.contributions(6).groupby("month")["contribution_ft"].sum()
    assert np.allclose(predictions, explained)


def test_blend_contributions_sum_to_the_blended_prediction(synthetic):
    model = _blend().fit(synthetic[synthetic["month"] <= pd.Timestamp("2015-03-01")])
    predictions = model.predict(12).set_index("month")["pred"]
    explained = model.contributions(12).groupby("month")["contribution_ft"].sum()
    assert np.allclose(predictions, explained)


def _blend3(synthetic):
    from src.forecasting.multivariate.inflow_chain import InflowChainForecaster

    return BlendForecaster(
        components=[
            ("swe_head", lambda: SweRegressionForecaster(name="swe_head")),
            ("inflow_chain", InflowChainForecaster),
            (
                "ets_damped_s12",
                lambda: HoltWintersForecaster(
                    trend="add", seasonal="add", seasonal_periods=12, damped_trend=True
                ),
            ),
        ],
        horizon=12,
        history_years=8,
        max_cutoffs=40,
        min_rows=4,
        name="blend3",
    )


def test_three_component_blend_mixes_on_the_simplex(synthetic):
    model = _blend3(synthetic).fit(synthetic[synthetic["month"] <= pd.Timestamp("2015-03-01")])
    assert model.k == 3
    for season in model.fitted_seasons:
        w = model.weights[season]
        assert w.shape == (12, 3)
        assert np.allclose(w.sum(axis=1), 1.0)
        assert (np.diff(1.0 - w[:, -1]) <= 1e-9).all()


def test_three_component_prediction_is_the_weighted_sum(synthetic):
    model = _blend3(synthetic).fit(synthetic[synthetic["month"] <= pd.Timestamp("2015-03-01")])
    parts = np.stack([f.predict(12)["pred"].to_numpy(dtype=float) for f in model._fitted])
    w = model.weights_for(12)
    assert np.allclose(model.predict(12)["pred"].to_numpy(), (w.T * parts).sum(axis=0))


def test_three_component_contributions_sum_to_the_prediction(synthetic):
    model = _blend3(synthetic).fit(synthetic[synthetic["month"] <= pd.Timestamp("2015-03-01")])
    predictions = model.predict(12).set_index("month")["pred"]
    explained = model.contributions(12).groupby("month")["contribution_ft"].sum()
    assert np.allclose(predictions, explained)


def test_blends_with_different_components_do_not_share_the_cache(synthetic):
    """The memo is keyed on the component name, so `blend` and `blend_swe` stay separate."""
    train = synthetic[synthetic["month"] <= pd.Timestamp("2015-03-01")]
    blend_cache.clear()
    _blend().fit(train)
    _blend(snow_features=["swe_eom_gsl", "prec_wy_eom_gsl"], snow_name="swe_regression").fit(train)
    labels = {key[0].split("|")[0] for key in blend_cache}
    assert labels == {"swe_head", "swe_regression", "ets_damped_s12"}
