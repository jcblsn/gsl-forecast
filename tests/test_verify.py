import pandas as pd
import pytest

from src.forecasting.run_forecast import export_forecasts
from src.forecasting.verify import load_issued, summarize, verify

META = {
    "schema_version": 1,
    "issue_status": "experimental",
    "forecast_version": "prototype-test",
    "code_commit": "abc123",
    "code_dirty": False,
    "evaluation_policy_version": "test-v1",
}


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
    out = export_forecasts(preds, str(tmp_path / "2026-09-01.csv"), str(cv_path), META)
    assert list(out["h"]) == [1, 2] and str(out["issue"].iloc[0]) == "2026-09-01"
    assert out["q95"].iloc[1] > out["q05"].iloc[1]

    issued = load_issued(str(tmp_path))
    observed = pd.DataFrame({"month": pd.to_datetime(["2026-09-01"]), "avg_elevation": [4190.8]})
    scored = verify(issued, observed)
    assert len(scored) == 1
    assert scored["abs_error"].iloc[0] == pytest.approx(0.2)
    summary = summarize(scored)
    assert set(summary.columns) == {
        "issue_status",
        "forecast_version",
        "model",
        "h",
        "mae",
        "bias",
        "n",
        "cov90",
    }
    assert summary.loc[0, "issue_status"] == "experimental"
    assert summary.loc[0, "forecast_version"] == "prototype-test"


def test_verification_refuses_a_forecast_without_metadata(tmp_path):
    pd.DataFrame(
        {"issue": ["2026-09-01"], "month": ["2026-09-01"], "model": ["m"], "pred": [1.0]}
    ).to_csv(tmp_path / "2026-09-01.csv", index=False)
    with pytest.raises(ValueError, match="Missing or malformed issue metadata"):
        load_issued(str(tmp_path))


def test_summary_keeps_forecast_versions_and_statuses_separate():
    scored = pd.DataFrame(
        {
            "issue_status": ["experimental", "release"],
            "forecast_version": ["v0", "v1"],
            "model": ["m", "m"],
            "h": [1, 1],
            "error": [1.0, 3.0],
            "abs_error": [1.0, 3.0],
        }
    )
    summary = summarize(scored)
    assert len(summary) == 2
    assert dict(zip(summary["forecast_version"], summary["mae"], strict=True)) == {
        "v0": 1.0,
        "v1": 3.0,
    }
