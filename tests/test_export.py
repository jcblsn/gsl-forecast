"""The dated forecast is a range, not one number, so it does not go out without one."""

import os

import numpy as np
import pandas as pd
import pytest

from src.forecasting.run_forecast import export_forecasts, require_intervals


def metadata():
    return {
        "schema_version": 1,
        "issue_status": "experimental",
        "forecast_version": "prototype-test",
        "code_commit": "abc123",
        "code_dirty": False,
        "evaluation_policy_version": "test-v1",
    }


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
    out = export_forecasts(predictions(["blend", "naive_last"]), path, str(cv), metadata())
    assert os.path.exists(path)
    assert out[["q05", "q95"]].notna().all().all()


def test_export_refuses_when_the_cv_file_omits_a_model(tmp_path):
    """The monthly job picked its interval file by modification time and could get a stale
    one. A model missing from it used to publish NaN quantiles in silence."""
    cv = tmp_path / "cv.parquet"
    cv_frame(["naive_last"]).to_parquet(cv, index=False)
    path = str(tmp_path / "2026-09-01.csv")
    with pytest.raises(SystemExit, match="No interval for \\['blend'\\]"):
        export_forecasts(predictions(["blend", "naive_last"]), path, str(cv), metadata())
    assert not os.path.exists(path)


def test_export_without_a_cv_file_writes_point_forecasts(tmp_path):
    """`gsl-plot` and a hindcast use the point path, so the guard applies only with --intervals."""
    path = str(tmp_path / "2026-09-01.csv")
    out = export_forecasts(predictions(["blend"]), path, None, metadata())
    assert "q05" not in out.columns and os.path.exists(path)


def test_dated_export_is_write_once_without_partial_replacement(tmp_path):
    path = str(tmp_path / "2026-09-01.csv")
    export_forecasts(predictions(["blend"]), path, None, metadata())
    before = {
        artifact: (tmp_path / artifact).read_bytes()
        for artifact in ("2026-09-01.csv", "2026-09-01.meta.json")
    }

    with pytest.raises(FileExistsError, match="write-once"):
        export_forecasts(predictions(["blend"]), path, None, metadata())

    assert not (tmp_path / "2026-09-01.explain.json").exists()
    assert all(
        (tmp_path / artifact).read_bytes() == content for artifact, content in before.items()
    )


@pytest.mark.parametrize("suffix", (".csv", ".meta.json", ".explain.json"))
def test_any_existing_issue_artifact_blocks_the_whole_export(tmp_path, suffix):
    stem = tmp_path / "2026-09-01"
    existing = tmp_path / f"{stem.name}{suffix}"
    existing.write_text("keep\n")
    with pytest.raises(FileExistsError, match="write-once"):
        export_forecasts(predictions(["blend"]), str(stem) + ".csv", None, metadata())
    assert existing.read_text() == "keep\n"
    assert len(list(tmp_path.iterdir())) == 1


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


def test_the_vintage_carries_a_content_address(tmp_path):
    """A maximum date says when the data stops, not what the values were."""
    from src.forecasting.run_forecast import table_fingerprint

    df = pd.DataFrame(
        {"month": pd.date_range("2020-01-01", periods=3, freq="MS"), "avg_elevation": [1.0, 2, 3]}
    )
    first = table_fingerprint(df)
    assert first["n_rows"] == 3 and first["columns"] == ["month", "avg_elevation"]
    assert first == table_fingerprint(df.iloc[::-1])

    revised = df.copy()
    revised.loc[0, "avg_elevation"] = 1.001
    assert table_fingerprint(revised)["sha256"] != first["sha256"]

    widened = df.assign(swe_eom_gsl=0.0)
    assert table_fingerprint(widened)["sha256"] != first["sha256"]


def test_the_config_digest_moves_when_the_roster_moves():
    from src.forecasting.run_forecast import config_fingerprint

    base = {"covariates": {"snotel": {"roster": {"stations": {"bear": ["1:UT:SNTL"]}}}}}
    other = {"covariates": {"snotel": {"roster": {"stations": {"bear": ["2:UT:SNTL"]}}}}}
    assert config_fingerprint(base) != config_fingerprint(other)
    assert config_fingerprint(base) == config_fingerprint(dict(base))
