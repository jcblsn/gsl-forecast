import numpy as np
import pandas as pd
import pytest

from src.forecasting.blend import (
    BLEND_MODEL,
    ETS_MODEL,
    SEASON_MONTHS,
    SNOW_MODEL,
    blend_contributions,
    blend_forward,
    cross_fitted_predictions,
    fit_weights,
    issue_season,
    weight_metadata,
)


def component_cv(years=range(2000, 2006), horizon=4):
    rows = []
    for year in years:
        for issue_month in range(1, 13):
            cutoff = pd.Timestamp(year, issue_month, 1) - pd.DateOffset(months=1)
            season = issue_season(cutoff)
            for h in range(1, horizon + 1):
                actual = 4190.0 + year * 0.001 + issue_month * 0.01 + h * 0.1
                snow_error = {"accumulation": 0.1, "melt": 0.2, "recession": 0.3}[season] * h
                ets_error = 0.5
                rows.extend(
                    [
                        {
                            "model": SNOW_MODEL,
                            "cutoff": cutoff,
                            "h": h,
                            "pred": actual + snow_error,
                            "actual": actual,
                        },
                        {
                            "model": ETS_MODEL,
                            "cutoff": cutoff,
                            "h": h,
                            "pred": actual + ets_error,
                            "actual": actual,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def test_issue_seasons_cover_each_month():
    assert set().union(*SEASON_MONTHS.values()) == set(range(1, 13))
    assert issue_season(pd.Timestamp("2025-12-01")) == "accumulation"
    assert issue_season(pd.Timestamp("2025-03-01")) == "melt"
    assert issue_season(pd.Timestamp("2025-08-01")) == "recession"


def test_weights_are_bounded_and_nonincreasing():
    weights = fit_weights(component_cv(), min_rows=3)
    assert weights["swe_weight"].between(0, 1).all()
    for _, group in weights.groupby("season"):
        assert np.all(np.diff(group.sort_values("h")["swe_weight"]) <= 0)


def test_cross_fitted_predictions_have_blend_errors():
    out = cross_fitted_predictions(component_cv(), min_rows=3)
    assert set(out["model"]) == {BLEND_MODEL}
    assert out["cutoff"].dt.year.nunique() == 7
    assert np.allclose(out["abs_error"], (out["pred"] - out["actual"]).abs())


def test_forward_blend_uses_issue_season():
    cv = component_cv()
    weights = fit_weights(cv, min_rows=3)
    months = pd.date_range("2026-01-01", periods=4, freq="MS")
    predictions = pd.concat(
        [
            pd.DataFrame(
                {"month": months, "target": "avg_elevation", "pred": 1.0, "model_name": model}
            )
            for model in (SNOW_MODEL, ETS_MODEL)
        ]
    )
    predictions.loc[predictions["model_name"] == ETS_MODEL, "pred"] = 2.0
    out = blend_forward(predictions, weights)
    selected = weights[weights["season"] == "accumulation"].sort_values("h")
    assert np.allclose(out["pred"], 2.0 - selected["swe_weight"].to_numpy())


def test_blended_contributions_reconcile_to_forward_path():
    months = pd.date_range("2026-01-01", periods=2, freq="MS")
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "month": months,
                    "target": "avg_elevation",
                    "pred": values,
                    "model_name": model,
                }
            )
            for model, values in ((SNOW_MODEL, [11.0, 12.0]), (ETS_MODEL, [20.0, 22.0]))
        ]
    )
    terms = pd.DataFrame(
        [
            {
                "month": month,
                "h": h,
                "input": name,
                "value": None,
                "reference": None,
                "contribution_ft": value,
            }
            for month, h, reference, feature in zip(
                months, [1, 2], [10.0, 10.0], [1.0, 2.0], strict=True
            )
            for name, value in (("reference_path", reference), ("snow", feature))
        ]
    )
    weights = pd.DataFrame({"season": "accumulation", "h": [1, 2], "swe_weight": [0.5, 0.5]})
    explained = blend_contributions(terms, predictions, weights)
    total = explained.groupby("h")["contribution_ft"].sum().to_numpy()
    blended = blend_forward(predictions, weights)["pred"].to_numpy()
    assert np.allclose(total, blended)


def test_calibration_requires_both_components():
    with pytest.raises(ValueError, match=SNOW_MODEL):
        fit_weights(component_cv().query("model != @SNOW_MODEL"), min_rows=1)


def test_weight_metadata_includes_held_out_blend_scores():
    cv = component_cv()
    blended = cross_fitted_predictions(cv, min_rows=3)
    scored = pd.concat([cv, blended], ignore_index=True)
    metadata = weight_metadata(fit_weights(cv, min_rows=3), scored)
    assert [row["h"] for row in metadata["held_out_mae"]] == [1, 2, 3, 4]
    assert all(row["mae_ft"] >= 0 for row in metadata["held_out_mae"])
