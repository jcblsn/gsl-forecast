"""`gsl-results` is step 1 of the loop in docs/program.md, so it must not raise."""

import json
import os

import pandas as pd
import pytest
from experiment_tracker import ExperimentTracker

from src.forecasting import results, view_results


@pytest.fixture
def tracker_db(tmp_path):
    path = str(tmp_path / "experiments.db")
    tracker = ExperimentTracker(path)
    exp_id = tracker.create_experiment("GSL_CV_test", "a run")
    for name, mae in (("swe_head", 0.51), ("naive_last", 1.33)):
        run_id = tracker.start_run(exp_id)
        tracker.log_model(run_id, name, {"last_cutoff_alpha": 0.1})
        tracker.log_metrics(run_id, {"mae_h6": mae, "rmse_h6": mae * 1.3, "peak_mae_feb": mae})
        tracker.end_run(run_id)
    empty = tracker.start_run(exp_id)
    tracker.log_model(empty, "broken", {})
    tracker.end_run(empty, success=False, error="Failed at all cutoffs during CV")
    return path, exp_id


def test_ranks_by_a_metric_that_exists(tracker_db, capsys):
    path, exp_id = tracker_db
    df = view_results.view_experiment(exp_id, path)
    assert list(df["model"])[:2] == ["swe_head", "naive_last"]
    assert "Models ranked by mae_h6" in capsys.readouterr().out


def test_a_run_without_the_metric_does_not_raise(tracker_db):
    path, exp_id = tracker_db
    df = view_results.view_experiment(exp_id, path, metric="mae_h24")
    assert df is not None and "mae_h24" not in df.columns


def test_unknown_experiment_returns_none(tracker_db):
    path, _ = tracker_db
    assert view_results.view_experiment(9999, path) is None


def _artifact(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "model": m,
                "h": h,
                "mae": mae,
                "rmse": mae,
                "mae_ratio": r,
                "crps": mae / 3,
                "cov90": 0.89,
            }
            for m, base, r in (("blend", 0.13, 0.38), ("naive_last", 0.34, 1.0))
            for h, mae in ((1, base), (6, base * 4), (12, base * 8), (24, base * 15))
        ]
    )
    headline = pd.DataFrame(
        [
            {"model": m, "issue": "feb", "target": "peak", "mae": v, "n": 13}
            for m, v in (("blend", 0.57), ("naive_last", 1.39))
        ]
    )
    meta = {
        "run_label": "GSL_CV_test",
        "headline_model": "blend",
        "n_cutoffs": 157,
        "first_cutoff": "2011-08-01",
        "last_cutoff": "2024-08-01",
        "horizon": 24,
        "train_start": "1960-01-01",
        "data_max": "2026-08-01",
        "git_commit": "abcdef123456789",
    }
    return results.write_results(summary, headline, meta, str(tmp_path / "results"))


def test_write_and_read_round_trip(tmp_path):
    directory = _artifact(tmp_path)
    assert set(os.listdir(directory)) == {
        results.CV_SUMMARY,
        results.HEADLINE_SUMMARY,
        results.META,
    }
    summary, headline, meta = results.read_results(directory)
    assert meta["run_label"] == "GSL_CV_test"
    assert len(summary) == 8 and len(headline) == 2
    with open(os.path.join(directory, results.META)) as stream:
        assert json.load(stream)["horizon"] == 24


def test_tables_render_from_the_committed_files(tmp_path, capsys):
    directory = _artifact(tmp_path)
    text = view_results.print_tables(directory, ["blend", "naive_last"])
    assert "| Lead | blend | naive_last | Ratio (blend) |" in text
    assert "| 6 | 0.52 | 1.36 | 0.38 |" in text
    assert "| Spring peak | Feb 1 | 0.57 | 1.39 |" in text
    assert "GSL_CV_test" in capsys.readouterr().out
