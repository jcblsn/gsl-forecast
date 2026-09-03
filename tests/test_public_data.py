import json

import pandas as pd

from src.forecasting.blend import BLEND_MODEL
from src.forecasting.run_forecast import explanation_payload, export_site_data


def issued_frame():
    return pd.DataFrame(
        {
            "month": [pd.Timestamp("2026-09-01")],
            "model": [BLEND_MODEL],
            "pred": [4190.25],
            "issue": [pd.Timestamp("2026-09-01")],
            "h": [1],
            "q05": [4189.75],
            "q95": [4190.75],
        }
    )


def contribution_frame():
    return pd.DataFrame(
        [
            {
                "month": pd.Timestamp("2026-09-01"),
                "h": 1,
                "input": "reference_path",
                "value": None,
                "reference": None,
                "contribution_ft": 4190.0,
                "swe_weight": 0.5,
            },
            {
                "month": pd.Timestamp("2026-09-01"),
                "h": 1,
                "input": "swe_eom_gsl",
                "value": 2.0,
                "reference": 1.0,
                "contribution_ft": 0.25,
                "swe_weight": 0.5,
            },
        ]
    )


def test_explanation_payload_reconciles():
    payload = explanation_payload(
        issued_frame(), contribution_frame(), {"weights": {}}, BLEND_MODEL
    )
    target = payload["targets"][0]
    total = sum(term["contribution_ft"] for term in target["contributions"])
    assert total == target["pred"]
    assert payload["headline_model"] == BLEND_MODEL


def test_incomplete_issue_keeps_last_valid_bundle(tmp_path):
    latest = tmp_path / "latest.json"
    latest.write_text('{"issue": "2026-08-01"}\n')
    observed = pd.DataFrame({"month": [pd.Timestamp("2026-08-01")], "avg_elevation": [4190.0]})
    meta = {"data_max": "2026-08-01", "problems": ["missing head difference"]}
    export_site_data(str(tmp_path), issued_frame(), observed, meta, explanation=None)
    assert json.loads(latest.read_text())["issue"] == "2026-08-01"
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["available"] is False


def test_complete_issue_replaces_site_bundle(tmp_path):
    observed = pd.DataFrame({"month": [pd.Timestamp("2026-08-01")], "avg_elevation": [4190.0]})
    payload = explanation_payload(issued_frame(), contribution_frame(), {}, BLEND_MODEL)
    meta = {"data_max": "2026-08-01", "problems": []}
    export_site_data(str(tmp_path), issued_frame(), observed, meta, payload)
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["issue"] == "2026-09-01"
    assert latest["observations"][0]["elevation"] == 4190.0
