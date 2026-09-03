import numpy as np
import pandas as pd
import pytest

from src.forecasting.cutoffs import SEASON_MONTHS
from src.forecasting.quantiles import (
    SEASON_SCALE_PRIOR,
    apply_intervals,
    error_quantiles,
    interval_pairs,
    leave_one_year_out_scores,
    pinball,
    probabilistic_scores,
    weighted_interval_score,
    with_season,
)


@pytest.fixture
def cv_df():
    rng = np.random.default_rng(1)
    rows = []
    for year in range(2010, 2020):
        for h in (1, 2):
            rows.append(
                {
                    "model": "m",
                    "cutoff": pd.Timestamp(year=year, month=3, day=1),
                    "h": h,
                    "pred": 4195.0,
                    "actual": 4195.0 + rng.normal(0, h),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def seasonal_cv_df():
    """Wide errors from an accumulation issue and narrow errors from a recession issue."""
    rng = np.random.default_rng(7)
    scale = {1: 4.0, 8: 0.5}
    rows = [
        {
            "model": "m",
            "cutoff": pd.Timestamp(year=year, month=month, day=1),
            "h": h,
            "pred": 4195.0,
            "actual": 4195.0 + rng.normal(0, scale[month]),
        }
        for year in range(1990, 2020)
        for month in scale
        for h in (1, 2)
    ]
    return pd.DataFrame(rows)


def test_error_quantiles_widen_with_horizon(cv_df):
    eq = error_quantiles(cv_df).query("issue_season == 'melt'").set_index("h")
    assert eq.loc[2, "q95"] - eq.loc[2, "q05"] > eq.loc[1, "q95"] - eq.loc[1, "q05"]


def test_error_quantiles_carry_every_season(cv_df):
    """A forecast issued in a season the cross-validation never covered still gets a band."""
    eq = error_quantiles(cv_df)
    assert set(eq["issue_season"]) == set(SEASON_MONTHS)


def test_the_band_is_wider_in_the_season_whose_errors_are_wider(seasonal_cv_df):
    eq = error_quantiles(seasonal_cv_df).set_index(["issue_season", "h"])
    width = eq["q95"] - eq["q05"]
    assert width.loc["accumulation", 1] > 2 * width.loc["recession", 1]


def _by_season(df, prior):
    eq = error_quantiles(df, prior=prior)
    scored = pd.concat(
        [
            apply_intervals(g, eq, "m", season)
            for season, g in with_season(df).groupby("issue_season")
        ],
        ignore_index=True,
    )
    return probabilistic_scores(scored, by=("issue_season", "h")).set_index(["issue_season", "h"])


def test_season_calibration_narrows_the_band_where_the_errors_are_narrow(seasonal_cv_df):
    """One band over every issue month is too wide in one season and too narrow in another."""
    pooled = _by_season(seasonal_cv_df, prior=1e12)
    season = _by_season(seasonal_cv_df, prior=SEASON_SCALE_PRIOR)

    assert pooled.loc["accumulation", 1]["width90"] == pytest.approx(
        pooled.loc["recession", 1]["width90"]
    )
    assert season.loc["recession", 1]["width90"] < 0.5 * season.loc["accumulation", 1]["width90"]

    def spread(table):
        lead_one = table.xs(1, level="h")["cov90"]
        return lead_one.max() - lead_one.min()

    assert spread(season) < spread(pooled)


def test_apply_intervals_adds_columns(cv_df):
    eq = error_quantiles(cv_df)
    out = apply_intervals(cv_df, eq, "m")
    assert {"q05", "q25", "q50", "q75", "q95"} <= set(out.columns)
    assert (out["q05"] <= out["q95"]).all()


def test_pinball_is_asymmetric():
    assert pinball(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.9)
    assert pinball(np.array([0.0]), np.array([1.0]), 0.9)[0] == pytest.approx(0.1)


def test_scores_have_expected_shape(cv_df):
    scored = apply_intervals(cv_df, error_quantiles(cv_df), "m")
    s = probabilistic_scores(scored)
    assert list(s.columns) == [
        "model",
        "h",
        "mean_pinball_loss",
        "wis",
        "cov90",
        "width90",
        "n_scored",
    ]
    losses = np.column_stack(
        [
            pinball(scored["actual"].to_numpy(), scored[f"q{q:02d}"].to_numpy(), q / 100)
            for q in (5, 25, 50, 75, 95)
        ]
    )
    assert s["mean_pinball_loss"].iloc[0] == pytest.approx(losses[scored["h"] == 1].mean())
    assert "crps" not in s.columns
    assert (s["cov90"] >= 0.8).all()
    loyo = leave_one_year_out_scores(cv_df)
    assert set(loyo["h"]) == {1, 2}


def test_wis_is_twice_the_mean_pinball_loss_for_a_symmetric_quantile_set(cv_df):
    """WIS carries a recognized name for the number the 5 pinball losses already gave."""
    scored = apply_intervals(cv_df, error_quantiles(cv_df), "m")
    s = probabilistic_scores(scored)
    assert (s["wis"] / s["mean_pinball_loss"]).round(9).eq(2.0).all()


def test_wis_penalises_an_actual_outside_the_band():
    """The penalty is 2/alpha times the distance outside, so a wide miss costs more."""
    inside = pd.DataFrame(
        {"actual": [0.0], "q05": [-1.0], "q25": [-0.5], "q50": [0.0], "q75": [0.5], "q95": [1.0]}
    )
    outside = inside.assign(actual=[3.0])
    # Inside: (0.5*0 + 0.05*2 + 0.25*1) / 2.5. Outside adds 2/alpha per foot beyond each edge.
    assert weighted_interval_score(np.array([0.0]), inside)[0] == pytest.approx(0.14)
    assert weighted_interval_score(np.array([3.0]), outside)[0] == pytest.approx(2.54)


def test_interval_pairs_reads_the_central_intervals():
    assert interval_pairs((0.05, 0.25, 0.5, 0.75, 0.95)) == [
        (0.05, 0.95, 0.1),
        (0.25, 0.75, 0.5),
    ]
    assert interval_pairs((0.1, 0.5, 0.9)) == [(0.1, 0.9, 0.2)]
