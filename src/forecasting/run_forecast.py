import argparse
import json
import logging
import os
from typing import Any, Dict, List

import duckdb
import pandas as pd
from experiment_tracker import ExperimentTracker

from src.forecasting.univariate.drift import DriftForecaster
from src.forecasting.univariate.exponential_smoothing import HoltWintersForecaster
from src.forecasting.univariate.moving_average import MovingAverageForecaster
from src.forecasting.univariate.naive import NaiveForecaster
from src.forecasting.univariate.theta import ThetaForecaster


def _load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = "config/config.json"
    with open(config_path) as f:
        return json.load(f)


def get_forecasters() -> List:
    return [
        NaiveForecaster(method="last"),
        NaiveForecaster(method="seasonal", seasonal_period=12),
        MovingAverageForecaster(window=3),
        MovingAverageForecaster(window=6),
        MovingAverageForecaster(window=12),
        DriftForecaster(window=24),
        HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12, damped_trend=True),
        HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12, damped_trend=False),
        HoltWintersForecaster(trend="add", seasonal=None, damped_trend=True),
        ThetaForecaster(),
    ]


def prepare_data(conn, validation_months: int):
    full_df = conn.execute("SELECT * FROM monthly_elevation ORDER BY month").fetchdf()

    if validation_months > 0:
        train_df = full_df.iloc[:-validation_months].copy()
        val_df = full_df.iloc[-validation_months:].copy()
    else:
        train_df = full_df.copy()
        val_df = pd.DataFrame()

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
        tracker.log_predictions(run_id, val_merged["pred"].tolist(), val_merged["avg_elevation"].tolist())


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


def run_single_forecaster(forecaster, tracker, exp_id, full_df, train_df, val_df, horizon, conn):
    run_id = tracker.start_run(exp_id)

    try:
        tracker.log_model(run_id, forecaster.name, forecaster.get_metrics())
        run_validation(forecaster, tracker, run_id, train_df, val_df)

        forecaster.fit(full_df)
        future_predictions = forecaster.predict(h=horizon)
        store_predictions(conn, future_predictions)

        tracker.end_run(run_id)
        return future_predictions

    except Exception as e:
        logging.error(f"Error running forecaster {forecaster.name}: {e}")
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

    config = _load_config(config_path)
    tracker = ExperimentTracker(experiment_db)
    exp_id = tracker.create_experiment(
        f"GSL_Forecast_{pd.Timestamp.now().strftime('%Y%m%d')}",
        f"Great Salt Lake forecast run with horizon={horizon}, validation_months={validation_months}",
    )

    forecasters = get_forecasters()
    all_predictions = []

    with duckdb.connect(config["database"]["path"]) as conn:
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
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon in months")
    parser.add_argument("--experiment-db", default="forecast_experiments.db", help="Path to experiment database")
    parser.add_argument("--validation-months", type=int, default=6, help="Number of months to use for validation")

    args = parser.parse_args()
    run_forecasts(
        config_path=args.config,
        horizon=args.horizon,
        experiment_db=args.experiment_db,
        validation_months=args.validation_months,
    )
