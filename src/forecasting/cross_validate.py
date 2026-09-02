import argparse
import logging
import os
import random

import pandas as pd
from dateutil.relativedelta import relativedelta
from experiment_tracker import ExperimentTracker

from src.config import load_config
from src.forecasting.base import Forecaster
from src.forecasting.data import load_monthly_data
from src.forecasting.headline import (
    headline_metrics,
    headline_scores,
    print_headline,
    summarize_headline,
)
from src.forecasting.quantiles import leave_one_year_out_scores
from src.forecasting.registry import BASELINE, all_forecasters


def valid_cutoffs(data: pd.DataFrame, history_years: int, horizon: int) -> list[pd.Timestamp]:
    """Every month in the last `history_years` that has `horizon` months of actuals after it."""
    latest_month = data["month"].max()
    earliest_cutoff = latest_month - relativedelta(years=history_years)
    latest_cutoff = latest_month - relativedelta(months=horizon)
    mask = (data["month"] >= earliest_cutoff) & (data["month"] <= latest_cutoff)
    return list(data.loc[mask, "month"])


def sample_cutoffs(
    data: pd.DataFrame,
    n: int | None,
    history_years: int,
    horizon: int,
    seed: int = 42,
) -> list[pd.Timestamp]:
    """All valid cutoffs when n is None, otherwise a seeded random sample of n."""
    valid = valid_cutoffs(data, history_years, horizon)
    if n is None:
        return valid
    if len(valid) < n:
        raise ValueError(f"Only {len(valid)} valid cutoffs available, requested {n}")
    return sorted(random.Random(seed).sample(valid, n))


def evaluate_at_cutoff(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    forecasters: list[Forecaster],
    horizon: int,
    train_start: str | None = None,
) -> pd.DataFrame:
    train = data[data["month"] <= cutoff]
    if train_start:
        train = train[train["month"] >= pd.Timestamp(train_start)]
    train = train.copy()
    actuals = data[data["month"] > cutoff].head(horizon).reset_index(drop=True)
    actuals["h"] = range(1, len(actuals) + 1)

    frames = []
    for forecaster in forecasters:
        try:
            preds = forecaster.fit(train).predict(h=horizon).reset_index(drop=True)
        except Exception as e:
            logging.warning(f"Forecaster {forecaster.name} failed at cutoff {cutoff}: {e}")
            continue
        preds["h"] = range(1, len(preds) + 1)
        merged = preds.merge(actuals[["h", "avg_elevation"]], on="h")
        frames.append(
            pd.DataFrame(
                {
                    "model": forecaster.name,
                    "cutoff": cutoff,
                    "h": merged["h"].astype(int),
                    "pred": merged["pred"],
                    "actual": merged["avg_elevation"],
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=["model", "cutoff", "h", "pred", "actual", "abs_error", "sq_error"]
        )
    out = pd.concat(frames, ignore_index=True)
    err = out["pred"] - out["actual"]
    out["abs_error"] = err.abs()
    out["sq_error"] = err**2
    return out


def summarize(cv_df: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Per model and horizon: MAE, RMSE, and MAE relative to the baseline model."""
    summary = (
        cv_df.groupby(["model", "h"])
        .agg(mae=("abs_error", "mean"), rmse=("sq_error", lambda x: x.mean() ** 0.5))
        .reset_index()
    )
    base = summary[summary["model"] == baseline].set_index("h")["mae"]
    summary["mae_ratio"] = summary["mae"] / summary["h"].map(base)
    return summary


def log_to_tracker(
    tracker: ExperimentTracker,
    exp_id: int,
    forecasters: list[Forecaster],
    cv_df: pd.DataFrame,
    summary: pd.DataFrame,
    headline_summary: pd.DataFrame | None = None,
) -> None:
    for forecaster in forecasters:
        run_id = tracker.start_run(exp_id)
        tracker.log_model(run_id, forecaster.name, forecaster.get_metrics())
        model_df = cv_df[cv_df["model"] == forecaster.name]
        if model_df.empty:
            tracker.end_run(run_id, success=False, error="Failed at all cutoffs during CV")
            logging.warning(f"No results for {forecaster.name}; logged as failed run")
            continue
        tracker.log_predictions(
            run_id, predictions=model_df["pred"].tolist(), actual_values=model_df["actual"].tolist()
        )
        model_summary = summary[summary["model"] == forecaster.name].set_index("h")
        metrics = {}
        for h, row in model_summary.iterrows():
            metrics[f"mae_h{h}"] = float(row["mae"])
            metrics[f"rmse_h{h}"] = float(row["rmse"])
            metrics[f"mae_ratio_h{h}"] = float(row["mae_ratio"])
            if "crps" in row and pd.notna(row["crps"]):
                metrics[f"crps_h{h}"] = float(row["crps"])
                metrics[f"cov90_h{h}"] = float(row["cov90"])
        if headline_summary is not None:
            metrics.update(headline_metrics(headline_summary, forecaster.name))
        tracker.log_metrics(run_id, metrics)
        tracker.end_run(run_id)


def print_summary(summary: pd.DataFrame, horizon: int) -> None:
    pivot = summary.pivot(index="model", columns="h", values="mae").round(3)
    pivot.columns = [f"h={c}" for c in pivot.columns]
    print("\nMean absolute error (ft) by model and horizon:")
    print(pivot.to_string())
    if "crps" in summary.columns:
        crps = summary.pivot(index="model", columns="h", values="crps").round(3)
        crps.columns = [f"h={c}" for c in crps.columns]
        print("\nCRPS (ft, leave-one-year-out empirical intervals) by model and horizon:")
        print(crps.to_string())
    print("\nBest model at each horizon (MAE ratio to naive_last in parentheses):")
    for h in range(1, horizon + 1):
        best = summary[summary["h"] == h].sort_values("mae").iloc[0]
        print(f"  h={h:2d}: {best['model']:<20} MAE={best['mae']:.3f} ({best['mae_ratio']:.2f})")


def run_cross_validation(
    config_path: str | None = None,
    n_cutoffs: int | None = None,
    horizon: int | None = None,
    history_years: int | None = None,
    train_start: str | None = None,
    experiment_db: str | None = None,
    seed: int = 42,
    output_dir: str | None = None,
    forecasters: list[Forecaster] | None = None,
    make_plots: bool = True,
) -> pd.DataFrame:
    """Walk-forward CV. Any argument left as None falls back to config/config.json."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(config_path)
    fc = config["forecasting"]
    horizon = horizon or fc["horizon"]
    history_years = history_years or fc["cv"]["history_years"]
    train_start = train_start or fc["train_start"]
    experiment_db = experiment_db or fc["experiment_db"]
    output_dir = output_dir or fc["output_dir"]
    if n_cutoffs is None and fc["cv"]["cutoffs"] != "all":
        n_cutoffs = int(fc["cv"]["cutoffs"])
    os.makedirs(output_dir, exist_ok=True)

    data = load_monthly_data(config["database"]["path"])
    cutoffs = sample_cutoffs(data, n_cutoffs, history_years, horizon, seed)
    logging.info(f"{len(cutoffs)} cutoffs from {cutoffs[0].date()} to {cutoffs[-1].date()}")
    if train_start:
        logging.info(f"Training data restricted to {train_start} onward")

    forecasters = forecasters if forecasters is not None else all_forecasters()
    results = []
    for i, cutoff in enumerate(cutoffs, start=1):
        logging.info(f"Cutoff {i}/{len(cutoffs)}: {cutoff.date()}")
        results.append(evaluate_at_cutoff(data, cutoff, forecasters, horizon, train_start))
    cv_df = pd.concat(results, ignore_index=True)
    summary = summarize(cv_df)
    headline = headline_scores(cv_df, data)
    headline_summary = summarize_headline(headline)
    prob = leave_one_year_out_scores(cv_df)
    summary = summary.merge(prob, on=["model", "h"], how="left")

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    cutoff_desc = "all month-end" if n_cutoffs is None else f"{n_cutoffs} random"
    tracker = ExperimentTracker(experiment_db)
    exp_id = tracker.create_experiment(
        f"GSL_CV_{stamp}",
        f"Walk-forward CV: {len(cutoffs)} cutoffs ({cutoff_desc}), {horizon}-month horizon, "
        f"last {history_years} years, training from {train_start or 'series start'}",
    )
    tags = {
        "data_min": str(data["month"].min().date()),
        "data_max": str(data["month"].max().date()),
        "n_months_available": str(len(data)),
        "n_cutoffs": str(len(cutoffs)),
        "cutoff_policy": cutoff_desc,
        "train_start": train_start or "",
    }
    for k, v in tags.items():
        tracker.log_tag("experiment", exp_id, k, v)
    log_to_tracker(tracker, exp_id, forecasters, cv_df, summary, headline_summary)

    per_cutoff_path = os.path.join(output_dir, f"cv_results_{stamp}.parquet")
    cv_df.to_parquet(per_cutoff_path, index=False)
    headline.to_parquet(os.path.join(output_dir, f"headline_{stamp}.parquet"), index=False)
    logging.info(f"Saved per-cutoff results to {per_cutoff_path} (experiment {exp_id})")

    if make_plots:
        from src.forecasting.plots import plot_cv_mae, plot_cv_ratio

        subtitle = f"{len(cutoffs)} {cutoff_desc} cutoffs, last {history_years} years"
        plot_cv_mae(summary, os.path.join(output_dir, "gsl_cv_mae.png"), subtitle)
        plot_cv_ratio(summary, os.path.join(output_dir, "gsl_cv_ratio.png"), subtitle)
        logging.info(f"Saved CV plots to {output_dir}")

    print_summary(summary, horizon)
    print_headline(headline_summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward CV for GSL forecasters")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument(
        "--n-cutoffs",
        type=int,
        default=None,
        help="Random sample of cutoffs; omit to use every month-end cutoff (config default)",
    )
    parser.add_argument("--horizon", type=int, help="Forecast horizon in months")
    parser.add_argument("--history-years", type=int, help="Years back to draw cutoffs from")
    parser.add_argument("--train-start", help="Earliest training date, e.g. 1960-01-01")
    parser.add_argument("--experiment-db", help="Path to experiment database")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", help="Directory for parquet and PNG outputs")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_cross_validation(
        config_path=args.config,
        n_cutoffs=args.n_cutoffs,
        horizon=args.horizon,
        history_years=args.history_years,
        train_start=args.train_start,
        experiment_db=args.experiment_db,
        seed=args.seed,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
