import argparse
import logging
import os

import numpy as np
import pandas as pd
from experiment_tracker import ExperimentTracker

from src.config import load_config
from src.forecasting.base import Forecaster
from src.forecasting.cutoffs import policy_cutoffs
from src.forecasting.data import load_monthly_data
from src.forecasting.headline import (
    headline_metrics,
    headline_scores,
    print_headline,
    summarize_headline,
)
from src.forecasting.quantiles import leave_one_year_out_scores
from src.forecasting.registry import BASELINE, all_forecasters


def resolve_evaluation_split(
    config: dict, split_name: str | None = None, horizon: int | None = None
) -> tuple[str, dict, int]:
    """Resolve one open, date-bounded CV split from the versioned policy."""
    policy = config.get("evaluation_policy")
    if not policy:
        raise ValueError("Configuration does not define evaluation_policy")
    name = split_name or config["forecasting"]["cv"].get("split") or policy["default_split"]
    splits = policy["splits"]
    if name not in splits:
        raise ValueError(f"Unknown evaluation split {name!r}; choose from {sorted(splits)}")
    split = splits[name]
    if split.get("status") == "sealed":
        raise ValueError(f"Evaluation split {name!r} is sealed and cannot be opened by gsl-cv")
    if split.get("status") != "open_development":
        raise ValueError(f"Evaluation split {name!r} is not an open development split")
    required = {"cutoff_start", "cutoff_end", "horizon"}
    missing = sorted(required - split.keys())
    if missing:
        raise ValueError(f"Evaluation split {name!r} is missing {missing}")
    policy_horizon = int(split["horizon"])
    if horizon is not None and horizon != policy_horizon:
        raise ValueError(f"Evaluation split {name!r} fixes horizon={policy_horizon}; got {horizon}")
    return name, split, policy_horizon


def require_empty_snapshot_target(results_dir: str) -> None:
    """Refuse to replace any existing snapshot file."""
    if not os.path.exists(results_dir):
        return
    if not os.path.isdir(results_dir) or os.listdir(results_dir):
        raise ValueError(f"Results snapshot target must be a new, empty directory: {results_dir}")


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
        if not np.isfinite(preds["pred"].to_numpy(dtype=float)).all():
            logging.warning(f"Forecaster {forecaster.name} gave non-finite values at {cutoff}")
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
    """One run per model, with the lead and the issue month carried as dimensions."""
    for forecaster in forecasters:
        # get_metrics describes the fit at the last cutoff only, not the whole walk, so the
        # key says so. A blend logs the weights it held at that one cutoff.
        last_fit = {f"last_cutoff_{k}": v for k, v in forecaster.get_metrics().items()}
        run = tracker.run(exp_id, name=forecaster.name, params=last_fit)
        model_df = cv_df[cv_df["model"] == forecaster.name]
        if model_df.empty:
            tracker.end_run(run.run_id, success=False, error="Failed at all cutoffs during CV")
            logging.warning(f"No results for {forecaster.name}; logged as failed run")
            continue
        with run:
            # The cutoff and the lead are what make a row addressable. Without them the
            # rows cannot be read back, and a metric cannot be checked against them.
            run.log_predictions(
                model_df["pred"].tolist(),
                model_df["actual"].tolist(),
                dims=[
                    {"cutoff": str(pd.Timestamp(cutoff).date()), "h": int(h)}
                    for cutoff, h in zip(model_df["cutoff"], model_df["h"], strict=True)
                ],
            )
            model_summary = summary[summary["model"] == forecaster.name].set_index("h")
            for h, row in model_summary.iterrows():
                values = {
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "mae_ratio": float(row["mae_ratio"]),
                }
                if "mean_pinball_loss" in row and pd.notna(row["mean_pinball_loss"]):
                    values["mean_pinball_loss"] = float(row["mean_pinball_loss"])
                    values["cov90"] = float(row["cov90"])
                run.log_metrics(values, dims={"h": int(h)})
            if headline_summary is not None:
                for values, dims in headline_metrics(headline_summary, forecaster.name):
                    run.log_metrics(values, dims=dims)


def print_summary(summary: pd.DataFrame, horizon: int) -> None:
    pivot = summary.pivot(index="model", columns="h", values="mae").round(3)
    pivot.columns = [f"h={c}" for c in pivot.columns]
    print("\nMean absolute error (ft) by model and horizon:")
    print(pivot.to_string())
    if "mean_pinball_loss" in summary.columns:
        loss = summary.pivot(index="model", columns="h", values="mean_pinball_loss").round(3)
        loss.columns = [f"h={c}" for c in loss.columns]
        print("\nMean pinball loss (ft, five quantiles) by model and horizon:")
        print(loss.to_string())
    print("\nBest model at each horizon (MAE ratio to naive_last in parentheses):")
    for h in range(1, horizon + 1):
        best = summary[summary["h"] == h].sort_values("mae").iloc[0]
        print(f"  h={h:2d}: {best['model']:<20} MAE={best['mae']:.3f} ({best['mae_ratio']:.2f})")


def run_cross_validation(
    config_path: str | None = None,
    n_cutoffs: int | None = None,
    horizon: int | None = None,
    train_start: str | None = None,
    experiment_db: str | None = None,
    seed: int = 42,
    output_dir: str | None = None,
    forecasters: list[Forecaster] | None = None,
    make_plots: bool = True,
    results_dir: str | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """Walk-forward CV over a named, date-bounded development cohort.

    `results_dir` is opt-in and must name a new or empty directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(config_path)
    fc = config["forecasting"]
    split_name, evaluation_split, horizon = resolve_evaluation_split(config, split, horizon)
    train_start = train_start or fc["train_start"]
    experiment_db = experiment_db or fc["experiment_db"]
    output_dir = output_dir or fc["output_dir"]
    if results_dir:
        require_empty_snapshot_target(results_dir)
    os.makedirs(output_dir, exist_ok=True)

    data = load_monthly_data(config["database"]["path"])
    cutoffs = policy_cutoffs(
        data,
        evaluation_split["cutoff_start"],
        evaluation_split["cutoff_end"],
        horizon,
        n_cutoffs,
        seed,
    )
    logging.info(
        f"{len(cutoffs)} cutoffs from split {split_name!r}: "
        f"{cutoffs[0].date()} to {cutoffs[-1].date()}"
    )
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
    cutoff_desc = split_name if n_cutoffs is None else f"{n_cutoffs} sampled from {split_name}"
    tracker = ExperimentTracker(experiment_db)
    # The tracker records the commit, the tree state and the command line by itself. The tags
    # below are the things it cannot know: the data vintage and the cutoff policy.
    exp_id = tracker.experiment(
        f"GSL_CV_{stamp}",
        f"Walk-forward CV: {len(cutoffs)} cutoffs ({cutoff_desc}), {horizon}-month horizon, "
        f"training from {train_start or 'series start'}",
        tags={
            "data_min": str(data["month"].min().date()),
            "data_max": str(data["month"].max().date()),
            "n_months_available": len(data),
            "n_cutoffs": len(cutoffs),
            "cutoff_policy": cutoff_desc,
            "evaluation_policy_version": config["evaluation_policy"]["version"],
            "evaluation_split": split_name,
            "evaluation_split_status": evaluation_split["status"],
            "policy_cutoff_start": evaluation_split["cutoff_start"],
            "policy_cutoff_end": evaluation_split["cutoff_end"],
            "first_cutoff": str(cutoffs[0].date()),
            "last_cutoff": str(cutoffs[-1].date()),
            "horizon": horizon,
            "train_start": train_start or "",
            "headline_model": fc.get("headline_model") or "",
            "models": ",".join(sorted({f.name for f in forecasters})),
        },
    )
    log_to_tracker(tracker, exp_id, forecasters, cv_df, summary, headline_summary)

    per_cutoff_path = os.path.join(output_dir, f"cv_results_{stamp}.parquet")
    cv_df.to_parquet(per_cutoff_path, index=False)
    headline.to_parquet(os.path.join(output_dir, f"headline_{stamp}.parquet"), index=False)
    # The run records where the parquet went, so asking the tracker replaces passing a path
    # between commands in a text file. The path and not the bytes: the predictions table
    # already holds the same rows, keyed by cutoff and lead.
    tracker.log_tags("experiment", exp_id, {"cv_parquet": per_cutoff_path})
    logging.info(f"Saved per-cutoff results to {per_cutoff_path} (experiment {exp_id})")

    if results_dir:
        tracker.snapshot(exp_id, results_dir)
        logging.info(f"Wrote the committed results snapshot to {results_dir}")

    if make_plots:
        from src.forecasting.plots import plot_cv_mae, plot_cv_ratio

        subtitle = f"{len(cutoffs)} cutoffs from {cutoff_desc}"
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
        help="Seeded sample from the named split; omit to use its complete fixed cohort",
    )
    parser.add_argument("--horizon", type=int, help="Must equal the named split's fixed horizon")
    parser.add_argument("--split", help="Named evaluation split (default: configured development)")
    parser.add_argument("--train-start", help="Earliest training date, e.g. 1960-01-01")
    parser.add_argument("--experiment-db", help="Path to experiment database")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", help="Directory for parquet and PNG outputs")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--models", help="Comma-separated model names to evaluate (default: all registered)"
    )
    parser.add_argument(
        "--results-dir",
        help="Opt-in snapshot target; must be a new or empty directory",
    )
    args = parser.parse_args()
    forecasters = None
    if args.models:
        wanted = set(args.models.split(","))
        forecasters = [f for f in all_forecasters() if f.name in wanted]
        unknown = wanted - {f.name for f in forecasters}
        if unknown:
            parser.error(f"Unknown models: {sorted(unknown)}")
        if BASELINE not in wanted:
            forecasters.append(next(f for f in all_forecasters() if f.name == BASELINE))
    try:
        run_cross_validation(
            config_path=args.config,
            n_cutoffs=args.n_cutoffs,
            horizon=args.horizon,
            train_start=args.train_start,
            experiment_db=args.experiment_db,
            seed=args.seed,
            output_dir=args.output_dir,
            forecasters=forecasters,
            make_plots=not args.no_plots,
            results_dir=args.results_dir,
            split=args.split,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
