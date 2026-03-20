import argparse
import json
import os
from typing import Optional

import duckdb
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    geom_line,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_x_datetime,
    theme,
    theme_bw,
)


def _load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(base_dir, "config", "config.json")
    with open(config_path) as f:
        return json.load(f)


def load_plot_data(db_path: str, history_years: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(db_path, read_only=True) as conn:
        actuals = conn.execute(f"""
            SELECT month, avg_elevation
            FROM monthly_elevation
            WHERE month >= CURRENT_DATE - INTERVAL '{history_years} years'
            ORDER BY month
        """).fetchdf()

        forecasts = conn.execute("""
            SELECT month, prediction, model,
                   ROW_NUMBER() OVER (PARTITION BY month, model ORDER BY created_at DESC) AS rn
            FROM forecasts
        """).fetchdf()

    forecasts = forecasts[forecasts["rn"] == 1].drop(columns="rn")

    actuals["month"] = pd.to_datetime(actuals["month"])
    forecasts["month"] = pd.to_datetime(forecasts["month"])

    return actuals, forecasts


def build_plot(actuals: pd.DataFrame, forecasts: pd.DataFrame):
    cutoff = actuals["month"].max()

    actuals_line = actuals.copy()
    actuals_line["model"] = "observed"

    forecast_lines = forecasts.rename(columns={"prediction": "avg_elevation"})

    combined = pd.concat([actuals_line, forecast_lines], ignore_index=True)

    is_observed = combined["model"] == "observed"
    combined["linewidth"] = combined["model"].apply(lambda m: 0.8 if m == "observed" else 0.6)
    combined["linetype"] = combined["model"].apply(lambda m: "solid" if m == "observed" else "dashed")

    n_models = forecasts["model"].nunique()

    plot = (
        ggplot(combined, aes(x="month", y="avg_elevation", color="model", group="model"))
        + geom_line(aes(size="linewidth", linetype="linetype"))
        + geom_vline(xintercept=cutoff, linetype="dotted", color="#888888", size=0.5)
        + scale_color_manual(values=[
            "#888888",  # observed
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78",
        ])
        + scale_x_datetime(date_labels="%Y", date_breaks="2 years")
        + labs(
            title="Great Salt Lake — Monthly Average Elevation",
            subtitle="Historical observations with out-of-sample model forecasts",
            x=None,
            y="Elevation (ft)",
            color="Model",
        )
        + theme_bw()
        + theme(
            figure_size=(12, 5),
            plot_title=element_text(size=13, face="bold"),
            plot_subtitle=element_text(size=10, color="#555555"),
            legend_position="right",
            legend_title=element_text(size=9),
            legend_text=element_text(size=8),
            panel_grid_minor=element_blank(),
            axis_text_x=element_text(angle=45, hjust=1),
        )
    )

    return plot


def plot_forecasts(
    config_path: Optional[str] = None,
    output_path: str = "gsl_forecast.png",
    history_years: int = 10,
) -> str:
    config = _load_config(config_path)
    actuals, forecasts = load_plot_data(config["database"]["path"], history_years)

    if forecasts.empty:
        raise RuntimeError("No forecasts found in database — run run_forecast.py first.")

    plot = build_plot(actuals, forecasts)
    plot.save(output_path, dpi=150, verbose=False)
    print(f"Saved plot to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GSL forecast results")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output", default="gsl_forecast.png", help="Output PNG path")
    parser.add_argument("--history-years", type=int, default=10, help="Years of history to show")

    args = parser.parse_args()
    plot_forecasts(config_path=args.config, output_path=args.output, history_years=args.history_years)
