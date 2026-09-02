"""Place our spring-peak forecasts next to the published NRCS outlook record."""

import argparse
import glob
import os

import pandas as pd

from src.config import load_config
from src.forecasting.headline import ISSUE_LABELS

BENCHMARK_CSV = os.path.join("data", "benchmarks", "nrcs_outlooks.csv")


def latest_headline_parquet(output_dir: str) -> str:
    paths = sorted(glob.glob(os.path.join(output_dir, "headline_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No headline_*.parquet under {output_dir}; run gsl-cv first")
    return paths[-1]


def compare(headline: pd.DataFrame, nrcs: pd.DataFrame) -> pd.DataFrame:
    """Per issue date: NRCS implied peak and error, our best model's peak and error.

    NRCS actuals are daily-reading peaks; ours are the peak of the monthly mean. Both errors
    are reported against their own definition, and the NRCS implied peak is also scored
    against the monthly-mean actual so the two are comparable in the last column.
    """
    nrcs = nrcs.copy()
    nrcs["issue_date"] = pd.to_datetime(nrcs["issue_date"])
    nrcs["issue"] = nrcs["issue_date"].dt.month.map(ISSUE_LABELS)
    nrcs["water_year"] = nrcs["issue_date"].dt.year
    peaks = headline[headline["target"] == "peak"]
    rows = []
    for _, r in nrcs.iterrows():
        ours = peaks[(peaks["issue"] == r["issue"]) & (peaks["water_year"] == r["water_year"])]
        row = {
            "issue_date": r["issue_date"].date(),
            "nrcs_implied_peak": r["implied_peak_ft"],
            "actual_daily_peak": r["actual_peak_ft"],
            "nrcs_error": (
                r["implied_peak_ft"] - r["actual_peak_ft"]
                if pd.notna(r["implied_peak_ft"])
                else float("nan")
            ),
        }
        if not ours.empty:
            best = ours.sort_values("abs_error").iloc[0]
            row.update(
                {
                    "our_model": best["model"],
                    "our_peak": round(best["pred"], 2),
                    "actual_monthly_peak": round(best["actual"], 2),
                    "our_error": round(best["pred"] - best["actual"], 2),
                    "nrcs_error_vs_monthly": (
                        round(r["implied_peak_ft"] - best["actual"], 2)
                        if pd.notna(r["implied_peak_ft"])
                        else float("nan")
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CV spring-peak errors with NRCS")
    parser.add_argument("--headline", help="headline_*.parquet from gsl-cv (default: latest)")
    parser.add_argument("--benchmark", default=BENCHMARK_CSV)
    parser.add_argument("--config")
    args = parser.parse_args()
    output_dir = load_config(args.config)["forecasting"]["output_dir"]
    path = args.headline or latest_headline_parquet(output_dir)
    headline = pd.read_parquet(path)
    nrcs = pd.read_csv(args.benchmark)
    print(f"Headline scores: {path}")
    print(compare(headline, nrcs).to_string(index=False))
    print("\nNRCS actuals are daily peaks; our actuals are peaks of the monthly mean.")


if __name__ == "__main__":
    main()
