"""The dated forecast is a range, not one number, so it does not go out without one."""

import os

import numpy as np
import pandas as pd
import pytest

from src.forecasting.run_forecast import export_forecasts, require_intervals


def cv_frame(models: list[str], horizon: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for model in models:
        for cutoff in pd.date_range("2020-01-01", periods=12, freq="MS"):
            for h in range(1, horizon + 1):
                pred = 4190.0 + rng.normal(0, 0.1)
                rows.append(
                    {"model": model, "cutoff": cutoff, "h": h, "pred": pred, "actual": pred + 0.2}
                )
    return pd.DataFrame(rows)


def predictions(models: list[str], horizon: int = 3) -> pd.DataFrame:
    rows = [
        {
            "month": pd.Timestamp("2026-09-01") + pd.DateOffset(months=h - 1),
            "model_name": model,
            "pred": 4190.0,
        }
        for model in models
        for h in range(1, horizon + 1)
    ]
    return pd.DataFrame(rows)


def test_export_writes_intervals_for_every_model(tmp_path):
    cv = tmp_path / "cv.parquet"
    cv_frame(["blend", "naive_last"]).to_parquet(cv, index=False)
    path = str(tmp_path / "2026-09-01.csv")
    out = export_forecasts(predictions(["blend", "naive_last"]), path, str(cv))
    assert os.path.exists(path)
    assert out[["q05", "q95"]].notna().all().all()


def test_export_refuses_when_the_cv_file_omits_a_model(tmp_path):
    """The monthly job picked its interval file by modification time and could get a stale
    one. A model missing from it used to publish NaN quantiles in silence."""
    cv = tmp_path / "cv.parquet"
    cv_frame(["naive_last"]).to_parquet(cv, index=False)
    path = str(tmp_path / "2026-09-01.csv")
    with pytest.raises(SystemExit, match="No interval for \\['blend'\\]"):
        export_forecasts(predictions(["blend", "naive_last"]), path, str(cv))
    assert not os.path.exists(path)


def test_export_without_a_cv_file_writes_point_forecasts(tmp_path):
    """`gsl-plot` and a hindcast use the point path, so the guard applies only with --intervals."""
    path = str(tmp_path / "2026-09-01.csv")
    out = export_forecasts(predictions(["blend"]), path, None)
    assert "q05" not in out.columns and os.path.exists(path)


def test_require_intervals_names_every_model_without_one():
    out = pd.DataFrame(
        {
            "model": ["a", "a", "b", "c"],
            "q05": [1.0, 1.0, np.nan, 3.0],
            "q95": [2.0, 2.0, 4.0, np.nan],
        }
    )
    with pytest.raises(SystemExit, match="\\['b', 'c'\\]"):
        require_intervals(out, "cv.parquet")
