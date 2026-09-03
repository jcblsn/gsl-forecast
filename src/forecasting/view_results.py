"""Print the metrics of one cross-validation run.

The command has 2 modes. With an experiment id it reads the experiment tracker database,
which is the working scratchpad for a run in progress. With `--tables` it reads the
committed files under `data/results/`, which are the record behind every published number.
"""

import argparse

import pandas as pd
from experiment_tracker import ExperimentTracker

from src.forecasting.results import RESULTS_DIR, read_results, render_tables

DEFAULT_METRIC = "mae_h6"

# The columns the autoresearch loop compares. Every other metric stays in the database and
# comes back with --all-metrics.
LOOP_METRICS = (
    "mae_h1",
    "mae_h3",
    "mae_h6",
    "mae_h12",
    "mae_h18",
    "mae_h24",
    "crps_h6",
    "crps_h12",
    "cov90_h12",
    "peak_mae_feb",
    "wyend_mae_apr",
    "wyend_mae_aug",
)


def loop_columns(df: pd.DataFrame, metric: str) -> list[str]:
    """The identity columns, the ranking metric, and the metrics the loop compares."""
    wanted = ["run_id", "model", metric, *LOOP_METRICS]
    seen, out = set(), []
    for name in wanted:
        if name in df.columns and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def get_run_metrics(tracker: ExperimentTracker, run_id: int) -> dict | None:
    try:
        model = tracker.get_model(run_id)
        metrics = tracker.get_metrics(run_id)
        model_name = model["model_name"] if model else "Unknown"
        return {"run_id": run_id, "model": model_name, **metrics}
    except (ValueError, IndexError):
        return None


def create_metrics_summary(
    tracker: ExperimentTracker, experiment_id: int, metric: str = DEFAULT_METRIC
) -> pd.DataFrame | None:
    """Every run in the experiment, ranked by `metric`.

    Warning: a run can lack the metric, because a model that failed at every cutoff logs
    none. Such a run goes to the end of the table instead of stopping the command.
    """
    runs = tracker.get_run_history(experiment_id)
    metrics_list = [get_run_metrics(tracker, run["run_id"]) for run in runs]
    metrics_list = [m for m in metrics_list if m is not None]

    if not metrics_list:
        return None

    df = pd.DataFrame(metrics_list)
    if metric not in df.columns:
        return df
    return df.sort_values(metric, na_position="last")


def view_experiment(
    experiment_id: int,
    experiment_db: str = "forecast_experiments.db",
    metric: str = DEFAULT_METRIC,
    all_metrics: bool = False,
) -> pd.DataFrame | None:
    tracker = ExperimentTracker(experiment_db)
    experiment = tracker.get_experiment(experiment_id)

    if not experiment:
        print(f"No experiment found with ID {experiment_id}")
        return None

    print(f"Experiment: {experiment['experiment_name']}")
    print(f"Description: {experiment['experiment_description']}")
    print(f"Created at: {experiment['created_time']}")

    metrics_df = create_metrics_summary(tracker, experiment_id, metric)
    if metrics_df is None:
        print("\nNo runs with metrics")
        return None
    if metric in metrics_df.columns:
        print(f"\nModels ranked by {metric}:")
    else:
        print(f"\nNo run logged {metric}; the table keeps the order of the runs:")
    shown = metrics_df if all_metrics else metrics_df[loop_columns(metrics_df, metric)]
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
        help="Print the published markdown tables from the committed results files",
    )
    parser.add_argument(
        "--all-metrics", action="store_true", help="Print every logged metric, not the loop set"
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directory of those files")
    parser.add_argument("--models", help="Comma-separated model names, in the column order")
    args = parser.parse_args()
    if args.tables:
        print_tables(args.results_dir, args.models.split(",") if args.models else None)
        return
    if args.experiment_id is None:
        parser.error("give an experiment id, or --tables")
    view_experiment(args.experiment_id, args.db, args.metric, args.all_metrics)


if __name__ == "__main__":
    main()
