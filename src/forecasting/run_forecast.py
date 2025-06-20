import argparse
import logging
from typing import Any, Dict, List

import pandas as pd
from experiment_tracker import ExperimentTracker

from forecasting.univariate.moving_average import MovingAverageForecaster
from forecasting.univariate.naive import NaiveForecaster
from utils.connect_db import get_db_connection
from utils.load_config import load_configuration


def get_forecasters() -> List:
    return [
        NaiveForecaster(method="last"),
        NaiveForecaster(method="seasonal", seasonal_period=12),
        MovingAverageForecaster(window=3),
        MovingAverageForecaster(window=6),
        MovingAverageForecaster(window=12),
    ]


def setup_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = "config/config.json"
    return load_configuration(config_path)


def prepare_data(conn, validation_months: int):
    full_df = conn.execute("SELECT * FROM monthly_elevation ORDER BY month").fetchdf()

    if validation_months > 0:
        train_df = full_df.iloc[:-validation_months].copy()
        val_df = full_df.iloc[-validation_months:].copy()
        logging.info(
            f"Split data: {len(train_df)} training rows, {len(val_df)} validation rows"
        )
    else:
        train_df = full_df.copy()
        val_df = pd.DataFrame()
        logging.info(f"Using all {len(train_df)} rows for training")

    return full_df, train_df, val_df


def run_validation(forecaster, tracker, run_id, train_df, val_df):
    if val_df.empty:
        return

    forecaster.fit(train_df)
    val_predictions = forecaster.predict(h=len(val_df), start_date=train_df["month"].max())

    val_merged = pd.merge(
        val_predictions[["month", "pred"]],
        val_df[["month", "avg_elevation"]],
        on="month",
        how="inner",
    )

    if not val_merged.empty:
        valid_preds = val_merged["pred"].tolist()
        valid_actuals = val_merged["avg_elevation"].tolist()
        tracker.log_predictions(run_id, valid_preds, valid_actuals)


def store_predictions(conn, predictions):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            month DATE,
            prediction FLOAT,
            model VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for _, row in predictions.iterrows():
        conn.execute(
            "INSERT INTO forecasts (month, prediction, model) VALUES (?, ?, ?)",
            [row["month"], row["pred"], row["model_name"]],
        )


def run_single_forecaster(
    forecaster, tracker, exp_id, full_df, train_df, val_df, horizon, conn
):
    run_id = tracker.start_run(exp_id)
    logging.info(f"Started run {run_id} for model {forecaster.name}")

    try:
        tracker.log_model(run_id, forecaster.name, forecaster.get_metrics())
        run_validation(forecaster, tracker, run_id, train_df, val_df)

        forecaster.fit(full_df)
        future_predictions = forecaster.predict(h=horizon)
        store_predictions(conn, future_predictions)

        tracker.end_run(run_id)
        logging.info(f"Completed run {run_id} for model {forecaster.name}")
        return future_predictions

    except Exception as e:
        logging.error(f"Error running forecaster {forecaster.name}: {str(e)}")
        tracker.end_run(run_id, success=False, error=str(e))
        return None


def run_forecasts(
    config_path=None,
    horizon=12,
    experiment_db="forecast_experiments.db",
    validation_months=6,
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = setup_config(config_path)
    tracker = ExperimentTracker(experiment_db)
    experiment_name = f"GSL_Forecast_{pd.Timestamp.now().strftime('%Y%m%d')}"
    exp_id = tracker.create_experiment(
        experiment_name,
        f"Great Salt Lake forecast run with horizon={horizon}, validation_months={validation_months}",
    )

    forecasters = get_forecasters()
    all_predictions = []

    with get_db_connection(config["database"]["path"]) as conn:
        full_df, train_df, val_df = prepare_data(conn, validation_months)

        for forecaster in forecasters:
            logging.info(f"Running forecaster: {forecaster.name}")
            predictions = run_single_forecaster(
                forecaster, tracker, exp_id, full_df, train_df, val_df, horizon, conn
            )
            if predictions is not None:
                all_predictions.append(predictions)

    if all_predictions:
        combined = pd.concat(all_predictions)
        logging.info(f"Generated {len(combined)} total predictions")
    else:
        logging.warning("No predictions were generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSL water level forecasts")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument(
        "--horizon", type=int, default=12, help="Forecast horizon in months"
    )
    parser.add_argument(
        "--experiment-db",
        default="forecast_experiments.db",
        help="Path to experiment database",
    )
    parser.add_argument(
        "--validation-months",
        type=int,
        default=6,
        help="Number of months to use for validation (0 to use all data for training)",
    )

    args = parser.parse_args()
    run_forecasts(
        config_path=args.config,
        horizon=args.horizon,
        experiment_db=args.experiment_db,
        validation_months=args.validation_months,
    )
