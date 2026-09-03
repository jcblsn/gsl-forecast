from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.forecasting.hindcast import hindcast_frame, score
from src.forecasting.univariate.naive import NaiveForecaster


def test_hindcast_frame_and_score():
    rows = [
        {
            "month": pd.Timestamp(date(2015, 1, 1) + relativedelta(months=i)),
            "avg_elevation": 4195 + (i % 12) * 0.1,
        }
        for i in range(60)
    ]
    data = pd.DataFrame(rows)
    cutoff = pd.Timestamp("2018-03-01")
    cv = pd.DataFrame(
        {
            "model": ["naive_last"] * 4,
            "cutoff": ["2016-01-01", "2016-02-01", "2017-01-01", "2017-02-01"],
            "h": [1, 1, 2, 2],
            "pred": [0, 0, 0, 0],
            "actual": [-0.1, 0.1, -0.2, 0.2],
        }
    )
    fc = hindcast_frame(data, cutoff, [NaiveForecaster(method="last")], 2, cv)
    assert list(fc["h"]) == [1, 2] and fc["actual"].notna().all()
    assert (fc["q05"] <= fc["pred"]).all() and (fc["q95"] >= fc["pred"]).all()
    s = score(fc, cutoff)
    assert (
        s.loc[0, "model"] == "naive_last"
        and "apr_jun_monthly_mean_max_pred" in s.columns
        and "cov90" in s.columns
    )
