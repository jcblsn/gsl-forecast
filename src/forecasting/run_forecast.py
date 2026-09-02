import argparse
import json
import logging
import os
from datetime import date

import duckdb
import pandas as pd
from experiment_tracker import ExperimentTracker

from src.config import load_config
from src.forecasting.base import Forecaster
from src.forecasting.data import load_monthly_data
from src.forecasting.quantiles import apply_intervals, error_quantiles
from src.forecasting.registry import production_forecasters

REQUIRED_COVARIATES = ("swe_eom_gsl", "prec_wy_eom_gsl")
MIN_OBS_DAYS = 28


def data_status(db_path: str) -> tuple[dict, list[str]]:
    """Describe the data vintage at the cutoff and list anything that would degrade the
    forecast silently: a stale series, a thin last month, or null covariates."""
    df = load_monthly_data(db_path)
    last = df.iloc[-1]
    with duckdb.connect(db_path, read_only=True) as conn:
        n_obs = conn.execute(
            "SELECT observation_count FROM monthly_elevation WHERE month = ?", [last["month"]]
        ).fetchone()[0]
    missing = [c for c in REQUIRED_COVARIATES if c not in df or pd.isna(last[c])]
    n_sites = last.get("n_snotel_sites")
    meta = {
        "data_max": str(last["month"].date()),
        "observation_count": int(n_obs),
        "n_snotel_sites": None if pd.isna(n_sites) else int(n_sites),
        "missing_covariates": missing,
    }
    problems = []
    if n_obs < MIN_OBS_DAYS:
        problems.append(f"only {n_obs} daily readings in {meta['data_max']}")
    if missing:
        problems.append(f"null at cutoff: {missing}")
    return meta, problems


def previous_month() -> date:
    first = date.today().replace(day=1)
    return (first - pd.DateOffset(months=1)).date()


def ensure_forecasts_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            month DATE,
            prediction FLOAT,
            model VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS run_id INTEGER")
    conn.execute("ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS experiment_id INTEGER")
    conn.execute("ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS data_max DATE")


def store_predictions(
    conn: duckdb.DuckDBPyConnection,
    predictions: pd.DataFrame,
    run_id: int,
    experiment_id: int,
    data_max: pd.Timestamp,
) -> None:
    rows = pd.DataFrame(
        {
            "month": pd.to_datetime(predictions["month"]),
            "prediction": predictions["pred"].astype(float),
            "model": predictions["model_name"],
            "created_at": pd.Timestamp.now(),
            "run_id": run_id,
            "experiment_id": experiment_id,
            "data_max": pd.Timestamp(data_max),
        }
    )
    conn.register("_new_forecasts", rows)
    conn.execute("""
        INSERT INTO forecasts
            (month, prediction, model, created_at, run_id, experiment_id, data_max)
        SELECT month, prediction, model, created_at, run_id, experiment_id, data_max
        FROM _new_forecasts
    """)
    conn.unregister("_new_forecasts")


def run_single_forecaster(
    forecaster: Forecaster,
    tracker: ExperimentTracker,
    exp_id: int,
    train_df: pd.DataFrame,
    horizon: int,
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame | None:
    run_id = tracker.start_run(exp_id)
    try:
        forecaster.fit(train_df)
        tracker.log_model(run_id, forecaster.name, forecaster.get_metrics())
        predictions = forecaster.predict(h=horizon)
        store_predictions(conn, predictions, run_id, exp_id, train_df["month"].max())
        tracker.end_run(run_id)
        return predictions
    except Exception as e:
        logging.error(f"Error running forecaster {forecaster.name}: {e}")
        tracker.end_run(run_id, success=False, error=str(e))
        return None


def run_forecasts(
    config_path: str | None = None,
    horizon: int | None = None,
    experiment_db: str | None = None,
    train_start: str | None = None,
    forecasters: list[Forecaster] | None = None,
) -> pd.DataFrame:
    """Fit each production model on all history from train_start and store h-step forecasts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(config_path)
    fc = config["forecasting"]
    horizon = horizon or fc["horizon"]
    experiment_db = experiment_db or fc["experiment_db"]
    train_start = train_start or fc["train_start"]
    forecasters = forecasters if forecasters is not None else production_forecasters()

    tracker = ExperimentTracker(experiment_db)
    exp_id = tracker.create_experiment(
        f"GSL_Forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
        f"Forward forecast, horizon={horizon}, training from {train_start or 'series start'}",
    )

    all_predictions = []
    with duckdb.connect(config["database"]["path"]) as conn:
        ensure_forecasts_table(conn)
        train_df = load_monthly_data(conn, train_start)
        data_min, data_max = train_df["month"].min().date(), train_df["month"].max().date()
        for k, v in {
            "data_min": str(data_min),
            "data_max": str(data_max),
            "n_months": str(len(train_df)),
            "train_start": train_start or "",
        }.items():
            tracker.log_tag("experiment", exp_id, k, v)
        logging.info(f"Training data: {data_min} to {data_max} ({len(train_df)} months)")

        for forecaster in forecasters:
            logging.info(f"Running forecaster: {forecaster.name}")
            preds = run_single_forecaster(forecaster, tracker, exp_id, train_df, horizon, conn)
            if preds is not None:
                all_predictions.append(preds)

    if not all_predictions:
        logging.warning("No predictions were generated")
        return pd.DataFrame()
    combined = pd.concat(all_predictions, ignore_index=True)
    logging.info(f"Stored {len(combined)} predictions under experiment {exp_id}")
    return combined


def export_forecasts(
    predictions: pd.DataFrame,
    path: str,
    cv_parquet: str | None = None,
    meta: dict | None = None,
) -> pd.DataFrame:
    """Write one dated forecast file: issue month, target month, lead, model, point, intervals.
    With `meta`, a sidecar <path stem>.meta.json records the data vintage behind it."""
    out = predictions.rename(columns={"model_name": "model"})[["month", "model", "pred"]].copy()
    out["month"] = pd.to_datetime(out["month"])
    origin = out["month"].min() - pd.DateOffset(months=1)
    out["issue"] = origin + pd.DateOffset(months=1)
    out["h"] = (out["month"].dt.year - origin.year) * 12 + out["month"].dt.month - origin.month
    if cv_parquet:
        eq = error_quantiles(pd.read_parquet(cv_parquet))
        out = pd.concat(
            [apply_intervals(g, eq, m) for m, g in out.groupby("model")], ignore_index=True
        )
    out = out.sort_values(["model", "h"]).reset_index(drop=True)
    out["issue"] = out["issue"].dt.date
    out["month"] = out["month"].dt.date
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False, float_format="%.3f")
    if meta:
        with open(os.path.splitext(path)[0] + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    logging.info(f"Exported {len(out)} rows to {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSL water level forecasts")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--horizon", type=int, help="Forecast horizon in months")
    parser.add_argument("--experiment-db", help="Path to experiment database")
    parser.add_argument("--train-start", help="Earliest training date, e.g. 1960-01-01")
    parser.add_argument(
        "--export", help="CSV path for the dated forecast, e.g. forecasts/2026-09.csv"
    )
    parser.add_argument("--intervals", help="cv_results parquet used for empirical intervals")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Issue even if the last month is thin or covariates are null at the cutoff",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    meta, problems = data_status(config["database"]["path"])
    if meta["data_max"] != str(previous_month()):
        raise SystemExit(f"Data end {meta['data_max']}, not last month; run gsl-pipeline first")
    if problems and not args.allow_incomplete:
        raise SystemExit("Refusing to issue: " + "; ".join(problems) + " (--allow-incomplete)")
    for p in problems:
        logging.warning(f"Issuing with incomplete data: {p}")
    preds = run_forecasts(
        config_path=args.config,
        horizon=args.horizon,
        experiment_db=args.experiment_db,
        train_start=args.train_start,
    )
    if args.export and not preds.empty:
        export_forecasts(preds, args.export, args.intervals, meta)


if __name__ == "__main__":
    main()
