import argparse
from typing import Dict, Optional

import pandas as pd
from experiment_tracker import ExperimentTracker


def get_experiment_info(
    tracker: ExperimentTracker, experiment_id: int
) -> Optional[Dict]:
    return tracker.get_experiment(experiment_id)


def get_run_metrics(tracker: ExperimentTracker, run_id: int) -> Optional[Dict]:
    try:
        models = tracker.get_models(run_id)
        metrics = tracker.get_metrics(run_id)
        model_name = models[0]["name"] if models else "Unknown"
        return {"run_id": run_id, "model": model_name, **metrics}
    except (ValueError, IndexError):
        return None


def create_metrics_summary(
    tracker: ExperimentTracker, experiment_id: int
) -> Optional[pd.DataFrame]:
    runs = tracker.get_run_history(experiment_id)
    metrics_list = [get_run_metrics(tracker, run["id"]) for run in runs]
    metrics_list = [m for m in metrics_list if m is not None]

    if not metrics_list:
        return None

    return pd.DataFrame(metrics_list).sort_values("rmse")


def view_experiment(
    experiment_id: int, experiment_db: str = "forecast_experiments.db"
) -> Optional[pd.DataFrame]:
    tracker = ExperimentTracker(experiment_db)
    experiment = get_experiment_info(tracker, experiment_id)

    if not experiment:
        print(f"No experiment found with ID {experiment_id}")
        return None

    print(f"Experiment: {experiment['name']}")
    print(f"Description: {experiment['description']}")
    print(f"Created at: {experiment['created_at']}")

    metrics_df = create_metrics_summary(tracker, experiment_id)
    if metrics_df is not None:
        print("\nModels Ranked by RMSE:")
        print(metrics_df.to_string(index=False))

    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View experiment results")
    parser.add_argument("--experiment_id", type=int, help="Experiment ID to view")
    parser.add_argument(
        "--db", default="forecast_experiments.db", help="Path to experiment database"
    )

    args = parser.parse_args()
    view_experiment(args.experiment_id, args.db)
