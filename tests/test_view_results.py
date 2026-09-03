"""`gsl-results` is step 1 of the loop in docs/program.md, so it must not raise."""

import json
import os

import pandas as pd
import pytest
from experiment_tracker import ExperimentTracker

from src.forecasting import results, view_results
from src.forecasting.headline import APR_JUN_MONTHLY_MEAN_MAX


@pytest.fixture
def tracker_db(tmp_path):
    path = str(tmp_path / "experiments.db")
    with ExperimentTracker(path) as tracker:
        exp_id = tracker.experiment("GSL_CV_test", "a run")
        for name, mae in (("swe_head", 0.51), ("naive_last", 1.33)):
            with tracker.run(exp_id, name=name, params={"last_cutoff_alpha": 0.1}) as run:
                run.log_metrics({"mae": mae, "rmse": mae * 1.3}, dims={"h": 6})
                run.log_metrics(
                    {"mae": mae},
                    dims={"target": APR_JUN_MONTHLY_MEAN_MAX, "issue": "feb"},
                )
        broken = tracker.start_run(exp_id, name="broken")
        tracker.end_run(broken, success=False, error="Failed at all cutoffs during CV")
    return path, exp_id


def test_ranks_by_a_metric_that_exists(tracker_db, capsys):
    path, exp_id = tracker_db
    df = view_results.view_experiment(exp_id, path)
    assert list(df["model"])[:2] == ["swe_head", "naive_last"]
    assert "Models ranked by mae_h6" in capsys.readouterr().out


def test_a_run_without_the_metric_still_appears(tracker_db):
    """A model that failed at every cutoff is a result, not a row to drop."""
    path, exp_id = tracker_db
    df = view_results.view_experiment(exp_id, path, metric="mae_h24")
    assert df is not None
    assert len(df) == 3
    assert "broken" in set(df["model"])
    assert "mae_h24" not in df.columns


def test_the_failed_run_sorts_last_when_the_metric_exists(tracker_db):
    path, exp_id = tracker_db
    df = view_results.view_experiment(exp_id, path)
    assert list(df["model"])[-1] == "broken"


def test_unknown_experiment_returns_none(tracker_db):
    path, _ = tracker_db
    assert view_results.view_experiment(9999, path) is None


def test_labels_round_trip_through_parse(tracker_db):
    for metric, dims in view_results.DISPLAY_METRICS:
        name = view_results.label(metric, dims)
        assert view_results.parse_label(name) == (metric, dims)


def test_label_names_match_the_ledger():
    assert view_results.label("mae", {"h": 6}) == "mae_h6"
    assert (
        view_results.label("mae", {"target": "apr_jun_monthly_mean_max", "issue": "feb"})
        == "apr_jun_monthly_mean_max_mae_feb"
    )
    assert (
        view_results.label("mae", {"target": "september_monthly_mean", "issue": "apr"})
        == "september_monthly_mean_mae_apr"
    )


def _snapshot(tmp_path):
    """A snapshot written the way `gsl-cv` writes one."""
    directory = str(tmp_path / "results")
    with ExperimentTracker(str(tmp_path / "snap.db")) as tracker:
        exp_id = tracker.experiment(
            "GSL_CV_test",
            "walk-forward",
            tags={
                "headline_model": "blend",
                "n_cutoffs": 157,
                "first_cutoff": "2011-08-01",
                "last_cutoff": "2024-08-01",
                "horizon": 24,
                "train_start": "1960-01-01",
                "data_max": "2026-08-01",
            },
        )
        for model, base, ratio in (("blend", 0.13, 0.38), ("naive_last", 0.34, 1.0)):
            with tracker.run(exp_id, name=model) as run:
                for h, mae in ((1, base), (6, base * 4), (12, base * 8), (24, base * 15)):
                    run.log_metrics(
                        {
                            "mae": mae,
                            "rmse": mae,
                            "mae_ratio": ratio,
                            "mean_pinball_loss": mae / 3,
                            "cov90": 0.89,
                        },
                        dims={"h": h},
                    )
                run.log_metrics(
                    {"mae": 0.57 if model == "blend" else 1.39, "n": 13},
                    dims={"target": APR_JUN_MONTHLY_MEAN_MAX, "issue": "feb"},
                )
        tracker.snapshot(exp_id, directory)
    return directory


def test_read_results_pivots_the_snapshot(tmp_path):
    directory = _snapshot(tmp_path)
    assert set(os.listdir(directory)) == {"experiment.json", "runs.csv", "metrics.csv"}
    summary, headline, meta = results.read_results(directory)
    assert meta["run_label"] == "GSL_CV_test"
    assert meta["horizon"] == "24"
    assert len(summary) == 8 and len(headline) == 2
    assert set(summary.columns) >= {
        "model",
        "h",
        "mae",
        "rmse",
        "mae_ratio",
        "mean_pinball_loss",
        "cov90",
    }
    assert summary[(summary["model"] == "blend") & (summary["h"] == 6)]["mae"].iloc[0] == 0.52


def test_the_commit_reaches_the_committed_record(tmp_path):
    directory = _snapshot(tmp_path)
    with open(os.path.join(directory, "experiment.json")) as stream:
        assert "git_commit" in json.load(stream)


def test_tables_render_from_the_committed_files(tmp_path, capsys):
    directory = _snapshot(tmp_path)
    text = view_results.print_tables(directory, ["blend", "naive_last"])
    assert "| Lead | blend | naive_last | Ratio (blend) |" in text
    assert "| 6 | 0.52 | 1.36 | 0.38 |" in text
    assert "| Maximum April–June monthly mean | Feb 1 | 0.57 | 1.39 |" in text
    assert "GSL_CV_test" in capsys.readouterr().out


def test_a_snapshot_round_trips_to_the_same_numbers(tmp_path):
    directory = _snapshot(tmp_path)
    summary, _, _ = results.read_results(directory)
    at_24 = summary[(summary["model"] == "naive_last") & (summary["h"] == 24)]
    assert at_24["mae"].iloc[0] == pytest.approx(0.34 * 15)
    assert pd.api.types.is_integer_dtype(summary["h"])


def test_frozen_development_snapshot_matches_its_manifest():
    manifest = results.verify_manifest()
    assert manifest["snapshot_status"] == "frozen_development_only"
    assert manifest["source_run"] == "GSL_CV_20260903_0004"
    assert manifest["numeric_value_count"] == 2774
    summary, headline, _ = results.read_results()
    assert "mean_pinball_loss" in summary and "crps" not in summary
    assert set(headline["target"]) == {
        "apr_jun_monthly_mean_max",
        "september_monthly_mean",
    }
