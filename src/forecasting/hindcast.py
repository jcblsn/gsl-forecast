"""Fit models at a past cutoff and chart their forecasts against what the gauge recorded."""

import argparse
import glob
import logging
import os
from datetime import date

import pandas as pd

from src.config import load_config
from src.forecasting.base import Forecaster
from src.forecasting.data import load_monthly_data
from src.forecasting.quantiles import apply_intervals, error_quantiles
from src.forecasting.registry import all_forecasters

DEFAULT_MODELS = ("blend", "swe_head", "ets_damped_s12", "naive_last")

# One fixed colour for each model, so a chart that drops a series does not recolour the
# others. The 6 values are the categorical palette subset that passes the all-pairs colour
# separation checks; the excluded orange collides with the red under deuteranopia. A model
# outside this table stops the run, because a shared colour makes 2 series look like 1.
COLORS = {
    "blend": "#e34948",
    "swe_head": "#008300",
    "swe_regression": "#2a78d6",
    "ets_damped_s12": "#4a3aa7",
    "naive_last": "#eda100",
    "inflow_chain": "#1baf7a",
}

# A variant takes the colour of the closest comparison and its own line type. A family is
# then 1 hue with 1 line type for each member, which needs no further colour separation.
VARIANT_OF = {
    "blend_swe": ("blend", "dashed"),
    "inflow_chain_area": ("inflow_chain", "dashed"),
    "state_space": ("ets_damped_s12", "dotted"),
}


def model_style(models: list[str]) -> tuple[dict, dict]:
    """The colour and the line type for each model. Unknown models stop the run."""
    unknown = [m for m in models if m not in COLORS and m not in VARIANT_OF]
    if unknown:
        raise SystemExit(
            f"No colour for {', '.join(unknown)}. The chart gives each model its own colour, "
            f"so add one to COLORS in {__name__}, or select from: "
            f"{', '.join(sorted(set(COLORS) | set(VARIANT_OF)))}"
        )
    colors = {m: COLORS[VARIANT_OF.get(m, (m,))[0]] for m in models}
    linetypes = {m: VARIANT_OF.get(m, (m, "solid"))[1] for m in models}
    return colors, linetypes


def hindcast_frame(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    forecasters: list[Forecaster],
    horizon: int,
    cv_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecasts from `cutoff` with intervals from other years' CV errors when cv_df is given."""
    train = data[data["month"] <= cutoff]
    eq = None
    if cv_df is not None and not cv_df.empty:
        other = cv_df[pd.to_datetime(cv_df["cutoff"]).dt.year != cutoff.year]
        eq = error_quantiles(other)
    frames = []
    for f in forecasters:
        preds = f.fit(train).predict(horizon).rename(columns={"model_name": "model"})
        preds["h"] = range(1, horizon + 1)
        if eq is not None and f.name in set(eq["model"]):
            preds = apply_intervals(preds, eq, f.name)
        elif eq is not None:
            logging.warning(
                f"{f.name} has no interval: the cross-validation file does not hold it. "
                "Re-run gsl-cv with this model to get one."
            )
        frames.append(preds)
    out = pd.concat(frames, ignore_index=True)
    out["month"] = pd.to_datetime(out["month"])
    return out.merge(
        data[["month", "avg_elevation"]].rename(columns={"avg_elevation": "actual"}),
        on="month",
        how="left",
    )


def score(fc: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    m = fc.dropna(subset=["actual"]).copy()
    m["ae"] = (m["pred"] - m["actual"]).abs()
    spring = m["month"].dt.month.isin([4, 5, 6]) & (m["month"].dt.year == cutoff.year)
    rows = []
    for model, g in m.groupby("model"):
        row = {
            "model": model,
            "mae_h1_6": g[g["h"] <= 6]["ae"].mean(),
            "mae_h1_12": g[g["h"] <= 12]["ae"].mean(),
            "mae_h13_24": g[g["h"] > 12]["ae"].mean(),
        }
        s = g[spring.loc[g.index]]
        if not s.empty:
            row["apr_jun_monthly_mean_max_pred"] = s["pred"].max()
            row["apr_jun_monthly_mean_max_obs"] = s["actual"].max()
        # After the concat every row holds a q05 column, and it is NULL for a model that the
        # cross-validation file does not cover. A comparison against NULL is false, so an
        # absent interval used to read as 0.00 coverage rather than as no interval.
        if "q05" in g.columns and g["q05"].notna().all():
            row["cov90"] = ((g["actual"] >= g["q05"]) & (g["actual"] <= g["q95"])).mean()
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def plot(fc: pd.DataFrame, data: pd.DataFrame, cutoff: pd.Timestamp, path: str) -> None:
    from plotnine import (
        aes,
        annotate,
        element_blank,
        element_text,
        geom_line,
        geom_ribbon,
        geom_vline,
        ggplot,
        labs,
        scale_color_manual,
        scale_fill_manual,
        scale_linetype_manual,
        theme,
        theme_minimal,
    )

    horizon = int(fc["h"].max())
    obs = data[
        (data["month"] >= cutoff - pd.DateOffset(years=5))
        & (data["month"] <= cutoff + pd.DateOffset(months=horizon))
    ]
    models = list(fc["model"].unique())
    colors, linetypes = model_style(models)
    top = max(obs["avg_elevation"].max(), fc["q95"].max() if "q95" in fc else 0)
    note = f"cutoff: data through {cutoff:%b %Y}"
    swe = data.loc[data["month"] == cutoff, "swe_eom_gsl"] if "swe_eom_gsl" in data else []
    if len(swe) and pd.notna(swe.iloc[0]):
        note += f"; basin SWE {swe.iloc[0]:.1f} in"
    p = ggplot()
    if "q05" in fc.columns:
        p = p + geom_ribbon(fc, aes("month", ymin="q05", ymax="q95", fill="model"), alpha=0.15)
    p = (
        p
        + geom_vline(xintercept=cutoff, linetype="dotted", color="#8a887f")
        + annotate("text", x=cutoff, y=top, label=note, ha="right", size=9, color="#8a887f")
        + geom_line(obs, aes("month", "avg_elevation"), color="#151512", size=1)
        + geom_line(fc, aes("month", "pred", color="model", linetype="model"), size=1)
        + scale_color_manual(values=colors)
        + scale_fill_manual(values=colors)
        + scale_linetype_manual(values=linetypes)
        + labs(
            title=f"Hindcast from {cutoff:%B %Y}: {horizon}-month forecasts against observations",
            x="",
            y="elevation (ft)",
        )
        + theme_minimal(base_size=12)
        + theme(
            panel_grid_minor=element_blank(),
            legend_position="top",
            legend_title=element_blank(),
            plot_title=element_text(size=14, weight="bold"),
            figure_size=(10, 5),
        )
    )
    p.save(path, dpi=150, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chart a hindcast from a past cutoff")
    parser.add_argument("cutoff", help="Last month of data to use, YYYY-MM")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--cv", help="cv_results parquet for intervals (default: latest)")
    parser.add_argument("--output-dir", help="Default: <config output_dir>/<today>")
    parser.add_argument("--config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config = load_config(args.config)
    fc_cfg = config["forecasting"]
    cutoff = pd.Timestamp(args.cutoff + "-01")
    horizon = args.horizon or fc_cfg["horizon"]
    data = load_monthly_data(config["database"]["path"], fc_cfg["train_start"])
    wanted = args.models.split(",")
    forecasters = [f for f in all_forecasters() if f.name in wanted]
    cv_path = args.cv or next(
        iter(
            sorted(
                glob.glob(
                    os.path.join(fc_cfg["output_dir"], "**", "cv_results_*.parquet"), recursive=True
                ),
                key=os.path.getmtime,
                reverse=True,
            )
        ),
        None,
    )
    cv_df = pd.read_parquet(cv_path) if cv_path else None
    fc = hindcast_frame(data, cutoff, forecasters, horizon, cv_df)
    out_dir = args.output_dir or os.path.join(fc_cfg["output_dir"], date.today().isoformat())
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{cutoff:%Y%m%d}_gsl_hindcast")
    plot(fc, data, cutoff, stem + ".png")
    fc.to_csv(stem + ".csv", index=False, float_format="%.3f")
    print(f"Wrote {stem}.png and .csv (intervals from {cv_path or 'none'})")
    print(score(fc, cutoff).to_string(index=False))


if __name__ == "__main__":
    main()
