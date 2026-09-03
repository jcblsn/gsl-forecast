"""Each production model must ignore the rows after its cutoff.

A fit at a cutoff can use only the data that was available on that date. Cross-validation
reads a complete table. Therefore a model that reads a later row gets a good score, and then
gives a different result at issue time. This test applies to each production model.
"""

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.forecasting.multivariate.blend import BlendForecaster
from src.forecasting.registry import production_forecasters

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
    train = series[series["month"] <= CUTOFF]
    later = tampered(series)
    honest = _fresh(factory).fit(train)
    leaked = _fresh(factory).fit(later[later["month"] <= CUTOFF])
    if isinstance(honest, BlendForecaster):
        for season in honest.weights:
            assert np.allclose(honest.weights[season], leaked.weights[season])
    assert np.allclose(
        honest.predict(HORIZON)["pred"].to_numpy(dtype=float),
        leaked.predict(HORIZON)["pred"].to_numpy(dtype=float),
        equal_nan=True,
    )
