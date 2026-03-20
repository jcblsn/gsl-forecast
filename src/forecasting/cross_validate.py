import argparse
import logging
import random
from typing import Optional

import duckdb
import pandas as pd
from dateutil.relativedelta import relativedelta
from experiment_tracker import ExperimentTracker

from src.config import load_config
from plotnine import (
    aes,
    element_blank,
    element_text,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_x_continuous,
    theme,
    theme_bw,
)

from src.forecasting.run_forecast import get_forecasters
from src.forecasting.univariate.drift import DriftForecaster
from src.forecasting.univariate.moving_average import MovingAverageForecaster
from src.forecasting.univariate.naive import NaiveForecaster
from src.forecasting.univariate.exponential_smoothing import HoltWintersForecaster
from src.forecasting.univariate.theta import ThetaForecaster


def get_all_forecasters() -> list:
    """Full set of models for CV benchmarking (baselines + new candidates)."""
    return [
        # Baselines
        NaiveForecaster(method="last"),
        NaiveForecaster(method="seasonal", seasonal_period=12),
        MovingAverageForecaster(window=3),
        MovingAverageForecaster(window=6),
        MovingAverageForecaster(window=12),
        # Drift variants
        DriftForecaster(window=12),
        DriftForecaster(window=24),
        DriftForecaster(window=60),
        # ETS variants
        HoltWintersForecaster(trend="add", seasonal=None, damped_trend=False),
        HoltWintersForecaster(trend="add", seasonal=None, damped_trend=True),
        HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12, damped_trend=False),
        HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12, damped_trend=True),
        # Theta
        ThetaForecaster(),
    ]


def load_monthly_data(db_path: str) -> pd.DataFrame:
    with duckdb.connect(db_path, read_only=True) as conn:
        df = conn.execute(
            "SELECT month, avg_elevation FROM monthly_elevation ORDER BY month"
        ).fetchdf()
    df["month"] = pd.to_datetime(df["month"])
    return df


def sample_cutoffs(
    data: pd.DataFrame,
    n: int,
    history_years: int,
    horizon: int,
    seed: int,
) -> list[pd.Timestamp]:
    latest_month = data["month"].max()
    earliest_cutoff = latest_month - relativedelta(years=history_years)
    # Need at least horizon months of actuals after the cutoff
    latest_cutoff = latest_month - relativedelta(months=horizon)

    valid = data[(data["month"] >= earliest_cutoff) & (data["month"] <= latest_cutoff)]["month"]
    if len(valid) < n:
        raise ValueError(f"Only {len(valid)} valid cutoffs available, requested {n}")

    rng = random.Random(seed)
    return sorted(rng.sample(list(valid), n))


def evaluate_at_cutoff(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    forecasters: list,
    horizon: int,
    train_start: Optional[str] = None,
) -> pd.DataFrame:
    train = data[data["month"] <= cutoff].copy()
    if train_start:
        train = train[train["month"] >= pd.Timestamp(train_start)]
    actuals = data[data["month"] > cutoff].head(horizon).copy()
    actuals = actuals.reset_index(drop=True)
    actuals["h"] = range(1, len(actuals) + 1)

    records = []
    for forecaster in forecasters:
        try:
            forecaster.fit(train)
            preds = forecaster.predict(h=horizon)
            preds = preds.reset_index(drop=True)
            preds["h"] = range(1, len(preds) + 1)

            merged = preds.merge(actuals[["h", "avg_elevation"]], on="h")
            for _, row in merged.iterrows():
                records.append({
                    "model": forecaster.name,
                    "cutoff": cutoff,
                    "h": int(row["h"]),
                    "pred": row["pred"],
                    "actual": row["avg_elevation"],
                    "abs_error": abs(row["pred"] - row["avg_elevation"]),
                    "sq_error": (row["pred"] - row["avg_elevation"]) ** 2,
                })
        except Exception as e:
            logging.warning(f"Forecaster {forecaster.name} failed at cutoff {cutoff}: {e}")

    return pd.DataFrame(records)


def run_cross_validation(
    config_path: Optional[str] = None,
    n_cutoffs: int = 10,
    horizon: int = 12,
    history_years: int = 15,
    train_start: Optional[str] = None,
    experiment_db: str = "forecast_experiments.db",
    seed: int = 42,
    output_path: str = "gsl_cv.png",
) -> pd.DataFrame:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = load_config(config_path)
    data = load_monthly_data(config["database"]["path"])

    cutoffs = sample_cutoffs(data, n_cutoffs, history_years, horizon, seed)
    logging.info(f"Sampled {len(cutoffs)} cutoffs from {cutoffs[0].date()} to {cutoffs[-1].date()}")

    forecasters = get_all_forecasters()
    all_results = []

    if train_start:
        logging.info(f"Training data restricted to {train_start} onward")

    for i, cutoff in enumerate(cutoffs):
        logging.info(f"Cutoff {i+1}/{len(cutoffs)}: {cutoff.date()}")
        results = evaluate_at_cutoff(data, cutoff, forecasters, horizon, train_start=train_start)
        all_results.append(results)

    cv_df = pd.concat(all_results, ignore_index=True)

    # Aggregate by model and horizon
    summary = (
        cv_df.groupby(["model", "h"])
        .agg(mae=("abs_error", "mean"), rmse=("sq_error", lambda x: x.mean() ** 0.5))
        .reset_index()
    )

    # Log to experiment tracker
    tracker = ExperimentTracker(experiment_db)
    train_desc = f", training from {train_start}" if train_start else ""
    exp_id = tracker.create_experiment(
        f"GSL_CV_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
        f"Walk-forward CV: {n_cutoffs} cutoffs, {horizon}-month horizon, last {history_years} years{train_desc}",
    )

    # Log data provenance on the experiment
    data_min = str(data["month"].min().date())
    data_max = str(data["month"].max().date())
    tracker.log_tag("experiment", exp_id, "data_min", data_min)
    tracker.log_tag("experiment", exp_id, "data_max", data_max)
    tracker.log_tag("experiment", exp_id, "n_months_available", str(len(data)))
    if train_start:
        tracker.log_tag("experiment", exp_id, "train_start", train_start)

    models_in_results = set(cv_df["model"].unique())

    for forecaster in forecasters:
        run_id = tracker.start_run(exp_id)
        tracker.log_model(run_id, forecaster.name, forecaster.get_metrics())

        model_df = cv_df[cv_df["model"] == forecaster.name]
        if forecaster.name not in models_in_results or model_df.empty:
            # Forecaster failed at all cutoffs — log as failed run
            tracker.end_run(run_id, success=False, error="Failed at all cutoffs during CV")
            logging.warning(f"No results for {forecaster.name} — logged as failed run")
            continue

        # Log aggregate predictions (gives overall MAE/RMSE)
        tracker.log_predictions(
            run_id,
            predictions=model_df["pred"].tolist(),
            actual_values=model_df["actual"].tolist(),
        )

        # Log per-horizon MAE so `expt metrics` and `expt best` can query by horizon
        model_summary = summary[summary["model"] == forecaster.name].set_index("h")
        horizon_metrics = {f"mae_h{h}": float(row["mae"]) for h, row in model_summary.iterrows()}
        horizon_metrics.update({f"rmse_h{h}": float(row["rmse"]) for h, row in model_summary.iterrows()})
        tracker.log_metrics(run_id, horizon_metrics)

        tracker.end_run(run_id)

    logging.info("CV complete. Generating plot...")

    plot = (
        ggplot(summary, aes(x="h", y="mae", color="model", group="model"))
        + geom_line(size=0.8)
        + geom_point(size=1.5)
        + scale_color_manual(values=[
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a",
        ])
        + scale_x_continuous(breaks=list(range(1, horizon + 1)))
        + labs(
            title="Walk-Forward Cross-Validation — Mean Absolute Error by Horizon",
            subtitle=f"{n_cutoffs} random cutoffs within last {history_years} years",
            x="Forecast Horizon (months)",
            y="Mean Absolute Error (ft)",
            color="Model",
        )
        + theme_bw()
        + theme(
            figure_size=(10, 5),
            plot_title=element_text(size=13, face="bold"),
            plot_subtitle=element_text(size=10, color="#555555"),
            panel_grid_minor=element_blank(),
        )
    )

    plot.save(output_path, dpi=150, verbose=False)
    logging.info(f"Saved CV plot to {output_path}")

    # Print summary table
    pivot = summary.pivot(index="model", columns="h", values="mae").round(3)
    pivot.columns = [f"h={c}" for c in pivot.columns]
    print("\nMean Absolute Error by Model and Horizon:")
    print(pivot.to_string())
    print(f"\nBest model at each horizon:")
    for h in range(1, horizon + 1):
        h_df = summary[summary["h"] == h].sort_values("mae")
        best = h_df.iloc[0]
        print(f"  h={h:2d}: {best['model']:<20} MAE={best['mae']:.3f}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward CV for GSL forecasters")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--n-cutoffs", type=int, default=10, help="Number of random cutoffs")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon in months")
    parser.add_argument("--history-years", type=int, default=15, help="Years to sample cutoffs from")
    parser.add_argument("--experiment-db", default="forecast_experiments.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-start", default=None, help="Earliest date for training data (e.g. 1960-01-01)")
    parser.add_argument("--output", default="gsl_cv.png", help="Output PNG path")

    args = parser.parse_args()
    run_cross_validation(
        config_path=args.config,
        n_cutoffs=args.n_cutoffs,
        horizon=args.horizon,
        history_years=args.history_years,
        train_start=args.train_start,
        experiment_db=args.experiment_db,
        seed=args.seed,
        output_path=args.output,
    )
