import argparse
import json
import logging
import os
import tempfile
from datetime import date

import duckdb
import pandas as pd
from experiment_tracker import ExperimentTracker

from src.config import load_config
from src.forecasting.base import Forecaster
from src.forecasting.data import load_monthly_data
from src.forecasting.multivariate.blend import (
    BlendForecaster,
    issue_season,
)
from src.forecasting.quantiles import apply_intervals, error_quantiles
from src.forecasting.registry import production_forecasters

REQUIRED_COVARIATES = ("swe_eom_gsl", "prec_wy_eom_gsl", "head_diff_ft")
TABLE_LEADS = (3, 6, 12, 24)
SAMPLE_COLUMNS = (
    "avg_elevation",
    "swe_eom_gsl",
    "prec_wy_eom_gsl",
    "head_diff_ft",
    "inflow_kaf_total",
)
SAMPLE_MONTHS = 12
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
    run = tracker.run(exp_id, name=forecaster.name)
    try:
        with run:
            forecaster.fit(train_df)
            tracker.set_params(run.run_id, forecaster.get_metrics())
            predictions = forecaster.predict(h=horizon)
            store_predictions(conn, predictions, run.run_id, exp_id, train_df["month"].max())
        return predictions
    except Exception as e:
        logging.error(f"Error running forecaster {forecaster.name}: {e}")
        return None


def run_forecasts(
    config_path: str | None = None,
    horizon: int | None = None,
    experiment_db: str | None = None,
    train_start: str | None = None,
    forecasters: list[Forecaster] | None = None,
    headline_enabled: bool = True,
) -> pd.DataFrame:
    """Fit each production model on all history from train_start and store h-step forecasts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(config_path)
    fc = config["forecasting"]
    horizon = horizon or fc["horizon"]
    experiment_db = experiment_db or fc["experiment_db"]
    train_start = train_start or fc["train_start"]
    forecasters = forecasters if forecasters is not None else production_forecasters()
    configured_headline = fc.get("headline_model")

    tracker = ExperimentTracker(experiment_db)
    exp_id = tracker.experiment(
        f"GSL_Forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
        f"Forward forecast, horizon={horizon}, training from {train_start or 'series start'}",
    )

    all_predictions = []
    fitted = {}
    with duckdb.connect(config["database"]["path"]) as conn:
        ensure_forecasts_table(conn)
        train_df = load_monthly_data(conn, train_start)
        data_min, data_max = train_df["month"].min().date(), train_df["month"].max().date()
        tracker.log_tags(
            "experiment",
            exp_id,
            {
                "data_min": str(data_min),
                "data_max": str(data_max),
                "n_months": len(train_df),
                "train_start": train_start or "",
            },
        )
        logging.info(f"Training data: {data_min} to {data_max} ({len(train_df)} months)")

        for forecaster in forecasters:
            logging.info(f"Running forecaster: {forecaster.name}")
            preds = run_single_forecaster(forecaster, tracker, exp_id, train_df, horizon, conn)
            if preds is not None:
                all_predictions.append(preds)
                fitted[forecaster.name] = forecaster

    if not all_predictions:
        logging.warning("No predictions were generated")
        return pd.DataFrame()
    combined = pd.concat(all_predictions, ignore_index=True)
    headline = fitted.get(configured_headline) if headline_enabled else None
    if configured_headline and headline_enabled and headline is None:
        logging.error(f"Headline model {configured_headline} did not produce a forecast")
    headline, calibration = headline_or_none(headline, horizon)
    combined.attrs["headline_model"] = headline.name if headline else None
    combined.attrs["calibration"] = calibration
    combined.attrs["contributions"] = (
        headline.contributions(horizon) if headline else pd.DataFrame()
    )
    logging.info(f"Stored {len(combined)} predictions under experiment {exp_id}")
    return combined


def headline_or_none(
    model: Forecaster | None, horizon: int
) -> tuple[Forecaster | None, dict | None]:
    """The headline model and its calibration, or 2 None values when the model refuses.

    Warning: the refusal covers the headline number, not the issue. The model paths still go
    out, and `export_site_data` keeps the last complete bundle on the public page. A refusal
    that stopped the run left no forecast for that month, and the workflow retry took the
    same path and stopped again.
    """
    if model is None:
        return None, None
    try:
        return model, headline_calibration(model, horizon)
    except ValueError as e:
        logging.error(f"No headline this issue: {e}")
        return None, None


def headline_calibration(model: Forecaster, horizon: int) -> dict:
    """The calibration details behind one issue.

    Warning: a blend that finds too few walk-forward cutoffs holds a fixed ramp. The ramp is
    a placeholder, not a fitted result, so this function refuses to publish it.
    """
    if not isinstance(model, BlendForecaster):
        return {"model": model.name, "metrics": model.get_metrics()}
    season = issue_season(model.last_date)
    if season not in model.fitted_seasons:
        raise ValueError(
            f"{model.name} did not fit weights for the {season} season from "
            f"{model.n_weight_cutoffs} cutoffs; refusing to publish a headline"
        )
    return {
        "algorithm_version": 3,
        "model": model.name,
        "components": list(model.component_names),
        "objective": "walk-forward MAE inside the training data",
        "weight_step": model.weight_step,
        "constraint": "the share on every component except the last does not increase by lead",
        "issue_season": season,
        "n_weight_cutoffs": model.n_weight_cutoffs,
        # One curve per component, keyed by season. `covariate_share` is 1 minus the weight
        # on the last component, which is the part of the forecast the covariates carry.
        "weights": {
            name: {
                component: [round(float(v), 2) for v in curve[:horizon, i]]
                for i, component in enumerate(model.component_names)
            }
            for name, curve in model.weights.items()
        },
        "covariate_share": {
            name: [round(float(1.0 - v), 2) for v in curve[:horizon, -1]]
            for name, curve in model.weights.items()
        },
    }


def latest_cv_parquet(experiment_db: str) -> str:
    """The per-cutoff file of the newest cross-validation run, from the tracker.

    Selection by modification time can pick up a file from an earlier run, and a model
    missing from it has no interval. The run records the file it wrote, so ask the run.
    """
    with ExperimentTracker(experiment_db) as tracker:
        for experiment in tracker.experiments():
            path = tracker.tags("experiment", experiment["experiment_id"]).get("cv_parquet")
            if path and os.path.exists(path):
                return path
    raise SystemExit(f"No cross-validation run recorded in {experiment_db}; run gsl-cv first")


def require_intervals(out: pd.DataFrame, cv_parquet: str) -> None:
    """Stop the export when a model has no interval.

    Warning: `apply_intervals` gives NaN for a model that the cross-validation file does not
    hold, and it raises nothing. The forecast is a range, not one number, so a missing
    interval is not publishable. Re-run `gsl-cv` and pass the file it writes.
    """
    quantile_cols = [c for c in out.columns if c.startswith("q") and c[1:].isdigit()]
    if not quantile_cols:
        raise SystemExit(f"{cv_parquet} produced no quantile columns")
    incomplete = out.groupby("model")[quantile_cols].apply(lambda g: g.isna().any().any())
    missing = sorted(incomplete[incomplete].index)
    if missing:
        raise SystemExit(
            f"No interval for {missing} in {cv_parquet}; re-run gsl-cv and take the path "
            "from `expt artifacts`"
        )


def export_forecasts(
    predictions: pd.DataFrame,
    path: str,
    cv_parquet: str | None = None,
    meta: dict | None = None,
) -> pd.DataFrame:
    """Write one dated forecast file: issue month, target month, lead, model, point,
    intervals. With `meta`, the sidecar <path stem>.meta.json records the data vintage. A
    complete headline issue also writes the sidecar <path stem>.explain.json."""
    calibration = predictions.attrs.get("calibration")
    headline_model = predictions.attrs.get("headline_model")
    out = predictions.rename(columns={"model_name": "model"})[["month", "model", "pred"]].copy()
    out.attrs = {}
    out["month"] = pd.to_datetime(out["month"])
    origin = out["month"].min() - pd.DateOffset(months=1)
    out["issue"] = origin + pd.DateOffset(months=1)
    out["h"] = (out["month"].dt.year - origin.year) * 12 + out["month"].dt.month - origin.month
    if cv_parquet:
        eq = error_quantiles(pd.read_parquet(cv_parquet))
        out = pd.concat(
            [apply_intervals(g, eq, m) for m, g in out.groupby("model")], ignore_index=True
        )
        require_intervals(out, cv_parquet)
    out = out.sort_values(["model", "h"]).reset_index(drop=True)
    out["issue"] = out["issue"].dt.date
    out["month"] = out["month"].dt.date
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False, float_format="%.3f")
    stem = os.path.splitext(path)[0]
    if meta:
        meta = dict(meta)
        meta["headline"] = {
            "available": headline_model is not None,
            "model": headline_model,
            "calibration": calibration,
        }
        write_json(stem + ".meta.json", meta)
    if headline_model:
        write_json(stem + ".explain.json", explanation_payload(out, predictions, headline_model))
    logging.info(f"Exported {len(out)} rows to {path}")
    return out


def _json_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, date)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (float, int)):
        return float(value)
    return value


def write_json(path: str, payload: dict) -> None:
    """Write one JSON file in a single operation, so that a reader cannot read a part file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def explanation_payload(
    exported: pd.DataFrame, predictions: pd.DataFrame, headline_model: str
) -> dict:
    """The dated explanation of the headline forecast, for a program to read."""
    contributions = predictions.attrs["contributions"]
    forecast = exported[exported["model"] == headline_model].sort_values("h")
    targets = []
    for _, row in forecast.iterrows():
        terms = contributions[contributions["h"] == row["h"]]
        targets.append(
            {
                "month": _json_value(row["month"]),
                "h": int(row["h"]),
                "pred": float(row["pred"]),
                "q05": _json_value(row.get("q05")),
                "q95": _json_value(row.get("q95")),
                "covariate_weight": float(terms["covariate_weight"].iloc[0])
                if "covariate_weight" in terms
                else None,
                "snow_weight": float(terms["covariate_weight"].iloc[0])
                if "covariate_weight" in terms
                else None,
                "contributions": [
                    {
                        "input": term["input"],
                        "value": _json_value(term["value"]),
                        "reference": _json_value(term["reference"]),
                        "contribution_ft": float(term["contribution_ft"]),
                    }
                    for _, term in terms.iterrows()
                ],
            }
        )
    return {
        "schema_version": 2,
        "issue": _json_value(forecast["issue"].iloc[0]),
        "headline_model": headline_model,
        "calibration": predictions.attrs["calibration"],
        "targets": targets,
    }


def export_site_data(
    data_dir: str,
    exported: pd.DataFrame,
    observed: pd.DataFrame,
    meta: dict,
    explanation: dict | None,
) -> None:
    """Write the site status on each run, and the headline after a complete issue.

    Warning: an incomplete run must not replace the published headline. Such a run writes
    the status file, which gives the reason, and keeps the last complete bundle.
    """
    status = {
        "attempted_issue": str(exported["issue"].iloc[0]),
        "available": explanation is not None,
        "problems": meta.get("problems", []),
        "data_max": meta["data_max"],
    }
    write_json(os.path.join(data_dir, "status.json"), status)
    if explanation is None:
        return
    bundle = dict(explanation)
    history = observed.sort_values("month").tail(120)
    bundle["observations"] = [
        {"month": _json_value(row["month"]), "elevation": float(row["avg_elevation"])}
        for _, row in history.iterrows()
    ]
    table = exported[exported["h"].isin(TABLE_LEADS)]
    bundle["models"] = [
        {"model": row["model"], "h": int(row["h"]), "pred": float(row["pred"])}
        for _, row in table.iterrows()
    ]
    bundle["inputs"] = input_sample(observed)
    bundle["vintage"] = {
        "data_max": meta.get("data_max"),
        "observation_count": meta.get("observation_count"),
        "n_snotel_sites": meta.get("n_snotel_sites"),
        "missing_covariates": meta.get("missing_covariates", []),
    }
    write_json(os.path.join(data_dir, "latest.json"), bundle)


def input_sample(observed: pd.DataFrame) -> dict:
    """The last months of the table the models fit on, for the method page to show."""
    columns = [c for c in SAMPLE_COLUMNS if c in observed.columns]
    rows = observed.sort_values("month").tail(SAMPLE_MONTHS)
    return {
        "columns": ["month", *columns],
        "rows": [
            {"month": _json_value(row["month"]), **{c: _json_value(row[c]) for c in columns}}
            for _, row in rows.iterrows()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSL water level forecasts")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--horizon", type=int, help="Forecast horizon in months")
    parser.add_argument("--experiment-db", help="Path to experiment database")
    parser.add_argument("--train-start", help="Earliest training date, e.g. 1960-01-01")
    parser.add_argument(
        "--export", help="CSV path for the dated forecast, e.g. forecasts/2026-09-01.csv"
    )
    parser.add_argument(
        "--intervals",
        nargs="?",
        const="latest",
        help="cv_results parquet for empirical intervals; bare flag takes the newest run's",
    )
    parser.add_argument("--site-data-dir", help="Directory for the public site data files")
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
    meta["problems"] = problems
    preds = run_forecasts(
        config_path=args.config,
        horizon=args.horizon,
        experiment_db=args.experiment_db,
        train_start=args.train_start,
        headline_enabled=not problems,
    )
    if args.export and not preds.empty:
        intervals = args.intervals
        if intervals == "latest":
            intervals = latest_cv_parquet(
                args.experiment_db or config["forecasting"]["experiment_db"]
            )
            logging.info(f"Intervals from {intervals}")
        exported = export_forecasts(preds, args.export, intervals, meta)
        if args.site_data_dir:
            headline_model = preds.attrs.get("headline_model")
            explanation = (
                explanation_payload(exported, preds, headline_model) if headline_model else None
            )
            export_site_data(
                args.site_data_dir,
                exported,
                load_monthly_data(config["database"]["path"]),
                meta,
                explanation,
            )


if __name__ == "__main__":
    main()
