"""The harness must give each model only the rows that existed at its cutoff.

Two rules protect a dated forecast. First, a prediction at a cutoff must not change when a
later row changes; cross-validation reads a finished table, so a model that reaches past its
cutoff scores well and then gives a different answer at issue time. Second, a model must not
read a column that has no value on the issue date, even though the finished table holds one.

The first rule is tested through `evaluate_at_cutoff`, which is the function that truncates
the frame. It receives the complete series, so a model that read a later row would change
its prediction. The second rule is tested against `feature_columns()`.
"""

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.cross_validate import evaluate_at_cutoff
from src.forecasting.multivariate.blend import BlendForecaster
from src.forecasting.registry import production_forecasters
from src.pipeline.covariates import UNAVAILABLE_AT_ISSUE

CUTOFF = pd.Timestamp("2015-03-01")
HORIZON = 6


@pytest.fixture(scope="module")
def series():
    """25 years of monthly data with each column that a production model reads."""
    rng = np.random.default_rng(11)
    rows = []
    level = 4195.0
    swe_year = 10.0
    for i in range(25 * 12):
        m = date(1995, 1, 1) + relativedelta(months=i)
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


def tampered(series: pd.DataFrame) -> pd.DataFrame:
    """The same series, with a large change to each value after the cutoff."""
    out = series.copy()
    future = out["month"] > CUTOFF
    out.loc[future, "avg_elevation"] += 25.0
    out.loc[future, "swe_eom_gsl"] *= 3.0
    out.loc[future, "prec_wy_eom_gsl"] *= 3.0
    out.loc[future, "head_diff_ft"] -= 2.0
    out.loc[future, "inflow_kaf_total"] *= 4.0
    return out


def _fresh(model):
    """A copy that is not fitted, so that the second fit cannot use data from the first.

    The blend runs an inner walk-forward pass. This test gives it a shorter pass.
    """
    if isinstance(model, BlendForecaster):
        return BlendForecaster(
            snow_features=model.snow_features,
            snow_name=model.snow_name,
            horizon=HORIZON,
            history_years=8,
            max_cutoffs=40,
            min_rows=4,
            name=model.name,
        )
    return copy.deepcopy(model)


@pytest.mark.parametrize("factory", production_forecasters(), ids=lambda f: f.name)
def test_predictions_ignore_rows_after_the_cutoff(series, factory):
    """The harness sees the complete series, so a later row must not move the forecast."""
    honest = evaluate_at_cutoff(series, CUTOFF, [_fresh(factory)], HORIZON)
    leaked = evaluate_at_cutoff(tampered(series), CUTOFF, [_fresh(factory)], HORIZON)
    assert not honest.empty, f"{factory.name} produced no rows at the cutoff"
    assert len(honest) == len(leaked)
    assert np.allclose(
        honest["pred"].to_numpy(dtype=float),
        leaked["pred"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_the_tamper_actually_changes_the_later_rows(series):
    """Without this the test above passes on 2 identical frames, which proves nothing."""
    later = series["month"] > CUTOFF
    assert not series[later].equals(tampered(series)[later])
    assert series[~later].equals(tampered(series)[~later])
    honest = evaluate_at_cutoff(series, CUTOFF, [_fresh(production_forecasters()[0])], HORIZON)
    leaked = evaluate_at_cutoff(
        tampered(series), CUTOFF, [_fresh(production_forecasters()[0])], HORIZON
    )
    assert not np.allclose(honest["actual"].to_numpy(), leaked["actual"].to_numpy())


@pytest.mark.parametrize("factory", production_forecasters(), ids=lambda f: f.name)
def test_no_model_reads_a_column_that_is_absent_at_issue_time(factory):
    forbidden = set(UNAVAILABLE_AT_ISSUE) & set(factory.feature_columns())
    assert not forbidden, f"{factory.name} reads {sorted(forbidden)}, which is 1 month late"
