"""Score every dated forecast in forecasts/ against what the gauge has since recorded."""

import argparse
import glob
import os

import pandas as pd

from src.config import load_config
from src.forecasting.data import load_monthly_data
from src.forecasting.quantiles import qcol


def load_issued(forecast_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(forecast_dir, "*.csv")))
    paths = [p for p in paths if os.path.basename(p)[0].isdigit()]
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["issue"] = pd.to_datetime(df["issue"])
    df["month"] = pd.to_datetime(df["month"])
    return df


def verify(issued: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Join issued forecasts to observed monthly means; keep rows that can be scored."""
    obs = observed[["month", "avg_elevation"]].rename(columns={"avg_elevation": "actual"})
    scored = issued.merge(obs, on="month", how="inner")
    scored["error"] = scored["pred"] - scored["actual"]
    scored["abs_error"] = scored["error"].abs()
    lo, hi = qcol(0.05), qcol(0.95)
    if lo in scored.columns:
        scored["in90"] = (scored["actual"] >= scored[lo]) & (scored["actual"] <= scored[hi])
    return scored


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    agg = {"mae": ("abs_error", "mean"), "bias": ("error", "mean"), "n": ("error", "size")}
    if "in90" in scored.columns:
        agg["cov90"] = ("in90", "mean")
    return scored.groupby(["model", "h"]).agg(**agg).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dated forecasts against observations")
    parser.add_argument("--config")
    parser.add_argument("--forecast-dir", default="forecasts")
    parser.add_argument("--out", default=os.path.join("forecasts", "verification.csv"))
    args = parser.parse_args()
    config = load_config(args.config)
    issued = load_issued(args.forecast_dir)
    if issued.empty:
        print("No dated forecasts found")
        return
    scored = verify(issued, load_monthly_data(config["database"]["path"]))
    if scored.empty:
        print(f"{len(issued)} issued rows, none observed yet")
        return
    summary = summarize(scored)
    summary.to_csv(args.out, index=False, float_format="%.3f")
    print(f"Scored {len(scored)} forecast rows from {issued['issue'].nunique()} issues")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
