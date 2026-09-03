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
from src.forecasting.blend import (
    BLEND_MODEL,
    SNOW_MODEL,
    blend_contributions,
    blend_forward,
    fit_weights,
    weight_metadata,
)
from src.forecasting.data import load_monthly_data
from src.forecasting.quantiles import apply_intervals, error_quantiles
from src.forecasting.registry import production_forecasters

REQUIRED_COVARIATES = ("swe_eom_gsl", "prec_wy_eom_gsl", "head_diff_ft")
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
    cv_results: str | None = None,
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
    exp_id = tracker.create_experiment(
        f"GSL_Forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
        f"Forward forecast, horizon={horizon}, training from {train_start or 'series start'}",
    )

    all_predictions = []
    explanation = pd.DataFrame()
    calibration = None
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

        component_predictions = (
            pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
        )
        if cv_results and headline_enabled and not component_predictions.empty:
            try:
                cv_df = pd.read_parquet(cv_results)
                weights = fit_weights(cv_df)
                blended = blend_forward(component_predictions, weights)
                snow_model = next(f for f in forecasters if f.name == SNOW_MODEL)
                snow_terms = snow_model.contributions(horizon)
                explanation = blend_contributions(snow_terms, component_predictions, weights)
                calibration = weight_metadata(weights, cv_df)
                run_id = tracker.start_run(exp_id)
                tracker.log_model(
                    run_id,
                    BLEND_MODEL,
                    {
                        "components": ",".join(calibration["components"]),
                        "weight_fit": "seasonal_monotone_mae",
                        "cutoff_max": calibration["cutoff_max"],
                    },
                )
                store_predictions(conn, blended, run_id, exp_id, train_df["month"].max())
                tracker.end_run(run_id)
                all_predictions.append(blended)
            except (ValueError, StopIteration) as error:
                logging.error(f"Could not create {BLEND_MODEL}: {error}")

    if not all_predictions:
        logging.warning("No predictions were generated")
        return pd.DataFrame()
    combined = pd.concat(all_predictions, ignore_index=True)
    combined.attrs["headline_model"] = (
        configured_headline if calibration and configured_headline == BLEND_MODEL else None
    )
    combined.attrs["calibration"] = calibration
    combined.attrs["contributions"] = explanation
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
    calibration = predictions.attrs.get("calibration")
    headline_model = predictions.attrs.get("headline_model")
    contributions = predictions.attrs.get("contributions")
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
    out = out.sort_values(["model", "h"]).reset_index(drop=True)
    out["issue"] = out["issue"].dt.date
    out["month"] = out["month"].dt.date
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False, float_format="%.3f")
    if meta:
        meta = dict(meta)
        meta["headline"] = {
            "available": headline_model is not None,
            "model": headline_model,
            "calibration": calibration,
        }
        _write_json(os.path.splitext(path)[0] + ".meta.json", meta)
    if headline_model:
        explanation = explanation_payload(out, contributions, calibration, headline_model)
        _write_json(os.path.splitext(path)[0] + ".explain.json", explanation)
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


def _write_json(path: str, payload: dict) -> None:
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
    exported: pd.DataFrame,
    contributions: pd.DataFrame,
    calibration: dict,
    headline_model: str,
) -> dict:
    """Build the dated machine-readable explanation for the headline model."""
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
                "swe_weight": float(terms["swe_weight"].iloc[0]),
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
        "schema_version": 1,
        "issue": _json_value(forecast["issue"].iloc[0]),
        "headline_model": headline_model,
        "calibration": calibration,
        "targets": targets,
    }


def export_site_data(
    data_dir: str,
    exported: pd.DataFrame,
    observed: pd.DataFrame,
    meta: dict,
    explanation: dict | None,
) -> None:
    """Update site status and replace the headline data only after a complete issue."""
    issue = str(exported["issue"].iloc[0])
    status = {
        "attempted_issue": issue,
        "available": explanation is not None,
        "problems": meta.get("problems", []),
        "data_max": meta["data_max"],
    }
    _write_json(os.path.join(data_dir, "status.json"), status)
    if explanation is None:
        return
    bundle = dict(explanation)
    history = observed.sort_values("month").tail(120)
    bundle["observations"] = [
        {"month": _json_value(row["month"]), "elevation": float(row["avg_elevation"])}
        for _, row in history.iterrows()
    ]
    _write_json(os.path.join(data_dir, "latest.json"), bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSL water level forecasts")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--horizon", type=int, help="Forecast horizon in months")
    parser.add_argument("--experiment-db", help="Path to experiment database")
    parser.add_argument("--train-start", help="Earliest training date, e.g. 1960-01-01")
    parser.add_argument(
        "--export", help="CSV path for the dated forecast, e.g. forecasts/2026-09.csv"
    )
    parser.add_argument(
        "--cv-results",
        "--intervals",
        dest="cv_results",
        help="CV parquet used for blend calibration and empirical intervals",
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
        cv_results=args.cv_results,
        headline_enabled=not problems,
    )
    if args.export and not preds.empty:
        exported = export_forecasts(preds, args.export, args.cv_results, meta)
        if args.site_data_dir:
            explanation = None
            headline_model = preds.attrs.get("headline_model")
            if headline_model:
                explanation = explanation_payload(
                    exported,
                    preds.attrs["contributions"],
                    preds.attrs["calibration"],
                    headline_model,
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
