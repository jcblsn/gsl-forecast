import numpy as np
import pandas as pd
import pytest

from src.forecasting.quantiles import (
    apply_intervals,
    error_quantiles,
    leave_one_year_out_scores,
    pinball,
    probabilistic_scores,
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


def test_error_quantiles_widen_with_horizon(cv_df):
    eq = error_quantiles(cv_df).set_index("h")
    assert eq.loc[2, "q95"] - eq.loc[2, "q05"] > eq.loc[1, "q95"] - eq.loc[1, "q05"]


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
    assert list(s.columns) == ["model", "h", "mean_pinball_loss", "cov90"]
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
