import json

import numpy as np
import pandas as pd
import pytest

from src.forecasting.multivariate.blend import BlendForecaster, default_weights
from src.forecasting.run_forecast import (
    explanation_payload,
    export_site_data,
    headline_calibration,
)

HEADLINE = "blend"


def issued_frame():
    return pd.DataFrame(
        {
            "month": [pd.Timestamp("2026-09-01")],
            "model": [HEADLINE],
            "pred": [4190.25],
            "issue": [pd.Timestamp("2026-09-01")],
            "h": [1],
            "q05": [4189.75],
            "q95": [4190.75],
        }
    )


def predictions_with_contributions():
    frame = pd.DataFrame({"month": [pd.Timestamp("2026-09-01")], "pred": [4190.25]})
    frame.attrs["contributions"] = pd.DataFrame(
        [
            {
                "month": pd.Timestamp("2026-09-01"),
                "h": 1,
                "input": "reference_path",
                "value": None,
                "reference": None,
                "contribution_ft": 4190.0,
                "snow_weight": 0.5,
            },
            {
                "month": pd.Timestamp("2026-09-01"),
                "h": 1,
                "input": "swe_eom_gsl",
                "value": 2.0,
                "reference": 1.0,
                "contribution_ft": 0.25,
                "snow_weight": 0.5,
            },
        ]
    )
    frame.attrs["calibration"] = {"weights": {}}
    return frame


def test_explanation_terms_sum_to_the_published_number():
    payload = explanation_payload(issued_frame(), predictions_with_contributions(), HEADLINE)
    target = payload["targets"][0]
    assert sum(term["contribution_ft"] for term in target["contributions"]) == target["pred"]
    assert payload["headline_model"] == HEADLINE
    assert payload["schema_version"] == 2


def test_incomplete_issue_keeps_the_last_complete_bundle(tmp_path):
    latest = tmp_path / "latest.json"
    latest.write_text('{"issue": "2026-08-01"}\n')
    observed = pd.DataFrame({"month": [pd.Timestamp("2026-08-01")], "avg_elevation": [4190.0]})
    meta = {"data_max": "2026-08-01", "problems": ["null at cutoff: ['head_diff_ft']"]}
    export_site_data(str(tmp_path), issued_frame(), observed, meta, None)
    assert json.loads(latest.read_text())["issue"] == "2026-08-01"
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["available"] is False
    assert status["problems"] == meta["problems"]


def test_complete_issue_replaces_the_bundle(tmp_path):
    observed = pd.DataFrame({"month": [pd.Timestamp("2026-08-01")], "avg_elevation": [4190.0]})
    payload = explanation_payload(issued_frame(), predictions_with_contributions(), HEADLINE)
    meta = {"data_max": "2026-08-01", "observation_count": 31, "problems": []}
    export_site_data(str(tmp_path), issued_frame(), observed, meta, payload)
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["issue"] == "2026-09-01"
    assert latest["observations"][0]["elevation"] == 4190.0
    assert latest["vintage"]["observation_count"] == 31
    assert latest["inputs"]["columns"] == ["month", "avg_elevation"]
    assert latest["inputs"]["rows"][0]["avg_elevation"] == 4190.0


def _fitted_blend(season, fitted_seasons):
    model = BlendForecaster(horizon=6)
    model.is_fitted = True
    model.last_date = {
        "accumulation": pd.Timestamp("2026-11-01"),
        "melt": pd.Timestamp("2026-03-01"),
    }[season]
    model.fitted_seasons = list(fitted_seasons)
    model.n_weight_cutoffs = 120 if fitted_seasons else 0
    model.weights = {name: default_weights(6) for name in model.weights}
    return model


def test_calibration_records_the_weights_that_were_fitted():
    model = _fitted_blend("melt", ["accumulation", "melt", "recession"])
    calibration = headline_calibration(model, 6)
    assert calibration["issue_season"] == "melt"
    assert calibration["constraint"] == "nonincreasing by lead"
    assert len(calibration["weights"]["melt"]) == 6


def test_calibration_refuses_a_curve_that_was_never_fitted():
    """A blend that found no cutoffs holds the fixed ramp, which must not reach an issue."""
    model = _fitted_blend("melt", [])
    assert np.allclose(model.weights["melt"], default_weights(6))
    with pytest.raises(ValueError, match="refusing to publish a headline"):
        headline_calibration(model, 6)
