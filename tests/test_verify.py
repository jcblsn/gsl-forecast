import pandas as pd
import pytest

from src.forecasting.run_forecast import export_forecasts
from src.forecasting.verify import summarize, verify


def test_export_and_verify_roundtrip(tmp_path):
    preds = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-09-01", "2026-10-01"]),
            "target": "avg_elevation",
            "pred": [4191.0, 4190.5],
            "model_name": "m",
        }
    )
    cv = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "h": [1, 1, 2, 2],
            "pred": [0, 0, 0, 0],
            "actual": [-0.2, 0.2, -0.5, 0.5],
        }
    )
    cv_path = tmp_path / "cv.parquet"
    cv.to_parquet(cv_path)
    out = export_forecasts(preds, str(tmp_path / "2026-09-01.csv"), str(cv_path))
    assert list(out["h"]) == [1, 2] and str(out["issue"].iloc[0]) == "2026-09-01"
    assert out["q95"].iloc[1] > out["q05"].iloc[1]

    issued = pd.read_csv(tmp_path / "2026-09-01.csv")
    issued["issue"] = pd.to_datetime(issued["issue"])
    issued["month"] = pd.to_datetime(issued["month"])
    observed = pd.DataFrame({"month": pd.to_datetime(["2026-09-01"]), "avg_elevation": [4190.8]})
    scored = verify(issued, observed)
    assert len(scored) == 1
    assert scored["abs_error"].iloc[0] == pytest.approx(0.2)
    summary = summarize(scored)
    assert set(summary.columns) == {"model", "h", "mae", "bias", "n", "cov90"}
