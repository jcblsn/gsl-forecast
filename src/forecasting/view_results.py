"""Print the metrics of one cross-validation run.

The command has 2 modes. With an experiment id it reads the experiment tracker database,
which is the working file for a run in progress. With `--tables` it reads the frozen
development snapshot under `data/results/`.

Metrics are stored with their dimensions, so a lead is data and not part of a name. Labels
like `mae_h6` exist only for display and for the command line.
"""

import argparse

import pandas as pd
from experiment_tracker import ExperimentTracker

from src.forecasting.headline import APR_JUN_MONTHLY_MEAN_MAX, SEPTEMBER_MONTHLY_MEAN
from src.forecasting.results import RESULTS_DIR, read_results, render_tables, verify_manifest

DEFAULT_METRIC = "mae_h6"

# The historical research view, as (metric, dims) pairs. Every other metric stays in the
# database and comes back with --all-metrics.
DISPLAY_METRICS = (
    ("mae", {"h": 1}),
    ("mae", {"h": 3}),
    ("mae", {"h": 6}),
    ("mae", {"h": 12}),
    ("mae", {"h": 18}),
    ("mae", {"h": 24}),
    ("wis", {"h": 6}),
    ("wis", {"h": 12}),
    ("cov90", {"h": 12}),
    ("mae", {"target": APR_JUN_MONTHLY_MEAN_MAX, "issue": "feb"}),
    ("mae", {"target": SEPTEMBER_MONTHLY_MEAN, "issue": "apr"}),
    ("mae", {"target": SEPTEMBER_MONTHLY_MEAN, "issue": "aug"}),
)

TARGETS = {APR_JUN_MONTHLY_MEAN_MAX, SEPTEMBER_MONTHLY_MEAN}


def label(metric: str, dims: dict) -> str:
    """A metric's display name, such as mae_h6 or apr_jun_monthly_mean_max_mae_feb."""
    if "h" in dims:
        return f"{metric}_h{int(dims['h'])}"
    if "target" in dims and "issue" in dims:
        return f"{dims['target']}_{metric}_{dims['issue']}"
    return metric


def parse_label(name: str) -> tuple[str, dict]:
    """Turn a display name back into a metric and its dims.

    This is the inverse of `label` and must stay so because command-line metric selectors
    use these flattened names.
    """
    for target in TARGETS:
        if name.startswith(f"{target}_"):
            rest = name[len(target) + 1 :]
            metric, _, issue = rest.rpartition("_")
            return metric, {"target": target, "issue": issue}
    metric, sep, lead = name.rpartition("_h")
    if sep and lead.isdigit():
        return metric, {"h": int(lead)}
    return name, {}


def metrics_frame(tracker: ExperimentTracker, experiment_id: int) -> pd.DataFrame:
    """One row per run, one column per metric label.

    A run that logged nothing still gets a row, because a model that failed at every cutoff
    is a result worth seeing rather than one to drop.
    """
    runs = tracker.runs(experiment=experiment_id)
    if not runs:
        return pd.DataFrame()
    rows = {
        run["run_id"]: {"run_id": run["run_id"], "model": run["name"] or "unknown"} for run in runs
    }
    for row in tracker.metrics(experiment=experiment_id):
        rows[row["run_id"]][label(row["metric"], row["dims"])] = row["value"]
    return pd.DataFrame(list(rows.values()))


def display_columns(df: pd.DataFrame, metric: str) -> list[str]:
    """The identity columns, ranking metric, and compact default metric set."""
    wanted = ["run_id", "model", metric, *(label(m, d) for m, d in DISPLAY_METRICS)]
    seen, out = set(), []
    for name in wanted:
        if name in df.columns and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def view_experiment(
    experiment_id: int,
    experiment_db: str = "forecast_experiments.db",
    metric: str = DEFAULT_METRIC,
    all_metrics: bool = False,
) -> pd.DataFrame | None:
    with ExperimentTracker(experiment_db) as tracker:
        experiment = tracker.get_experiment(experiment_id)
        if not experiment:
            print(f"No experiment found with ID {experiment_id}")
            return None

        print(f"Experiment: {experiment['name']}")
        print(f"Description: {experiment['description']}")
        print(f"Created at: {experiment['created_at']}")
        if experiment["git_commit"]:
            dirty = " (dirty tree)" if experiment["git_dirty"] else ""
            print(f"Commit: {experiment['git_commit'][:12]}{dirty}")

        metrics_df = metrics_frame(tracker, experiment_id)

    if metrics_df.empty:
        print("\nNo runs")
        return None
    if metric in metrics_df.columns:
        metrics_df = metrics_df.sort_values(metric, na_position="last")
        print(f"\nModels ranked by {metric}:")
    else:
        print(f"\nNo run logged {metric}; the table keeps the order of the runs:")
    shown = metrics_df if all_metrics else metrics_df[display_columns(metrics_df, metric)]
    print(shown.round(3).to_string(index=False))
    return metrics_df


def print_tables(results_dir: str = RESULTS_DIR, models: list[str] | None = None) -> str:
    """The markdown tables for the README and for `docs/model-spec.md`."""
    summary, headline, meta = read_results(results_dir)
    text = render_tables(summary, headline, meta, models)
    print(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="View experiment results")
    parser.add_argument(
        "experiment_id", type=int, nargs="?", help="Experiment ID to view; omit with --tables"
    )
    parser.add_argument(
        "--db", default="forecast_experiments.db", help="Path to experiment database"
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric to rank runs by (default {DEFAULT_METRIC})",
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Print markdown tables from the frozen development snapshot",
    )
    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="Verify the frozen development snapshot's SHA-256 manifest",
    )
    parser.add_argument("--all-metrics", action="store_true", help="Print every logged metric")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directory of those files")
    parser.add_argument("--models", help="Comma-separated model names, in the column order")
    args = parser.parse_args()
    if args.verify_manifest:
        manifest = verify_manifest(args.results_dir)
        print(f"Verified frozen snapshot: {manifest['source_run']}")
        return
    if args.tables:
        print_tables(args.results_dir, args.models.split(",") if args.models else None)
        return
    if args.experiment_id is None:
        parser.error("give an experiment id, or --tables")
    view_experiment(args.experiment_id, args.db, args.metric, args.all_metrics)


if __name__ == "__main__":
    main()
