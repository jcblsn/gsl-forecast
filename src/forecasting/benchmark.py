"""Place our spring-peak forecasts next to the published NRCS outlook record."""

import argparse
import glob
import os

import pandas as pd

from src.config import load_config
from src.forecasting.cross_validate import evaluate_at_cutoff
from src.forecasting.data import load_monthly_data
from src.forecasting.headline import ISSUE_LABELS, headline_scores
from src.forecasting.registry import all_forecasters

BENCHMARK_CSV = os.path.join("data", "benchmarks", "nrcs_outlooks.csv")
MODEL = "ets_damped_s12"


def latest_headline_parquet(output_dir: str) -> str:
    paths = sorted(glob.glob(os.path.join(output_dir, "headline_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No headline_*.parquet under {output_dir}; run gsl-cv first")
    return paths[-1]


def compare(headline: pd.DataFrame, nrcs: pd.DataFrame, model: str = MODEL) -> pd.DataFrame:
    """Per issue date: NRCS implied peak and error, and one named model's peak and error.

    The model is fixed in advance rather than chosen per row so the comparison is not a
    best-of-thirteen selection made after seeing the answer.

    NRCS actuals are daily-reading peaks; ours are the peak of the monthly mean. Both errors
    are reported against their own definition, and the NRCS implied peak is also scored
    against the monthly-mean actual so the two are comparable in the last column.
    """
    nrcs = nrcs.copy()
    nrcs["issue_date"] = pd.to_datetime(nrcs["issue_date"])
    nrcs["issue"] = nrcs["issue_date"].dt.month.map(ISSUE_LABELS)
    nrcs["water_year"] = nrcs["issue_date"].dt.year
    peaks = headline[(headline["target"] == "peak") & (headline["model"] == model)]
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
            best = ours.iloc[0]
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


def refit_at_issues(nrcs: pd.DataFrame, db_path: str, train_start: str | None, model: str):
    """Fit the model at each NRCS issue date and score its spring peak, no CV run needed.

    Uses the data as it stands today rather than the vintage available at the time, so this
    is a hindcast of the method, not a record of what was issued.
    """
    data = load_monthly_data(db_path, train_start)
    forecaster = next(f for f in all_forecasters() if f.name == model)
    frames = []
    for issue in pd.to_datetime(nrcs["issue_date"]).unique():
        cutoff = pd.Timestamp(issue) - pd.DateOffset(months=1)
        if cutoff not in set(data["month"]):
            continue
        frames.append(evaluate_at_cutoff(data, cutoff, [forecaster], 8, train_start))
    cv = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return headline_scores(cv, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CV spring-peak errors with NRCS")
    parser.add_argument("--headline", help="headline_*.parquet from gsl-cv (default: latest)")
    parser.add_argument("--benchmark", default=BENCHMARK_CSV)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Refit the model at each NRCS issue date instead of reading CV output",
    )
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    nrcs = pd.read_csv(args.benchmark)
    if args.refit:
        headline = refit_at_issues(
            nrcs, config["database"]["path"], config["forecasting"]["train_start"], args.model
        )
        print(f"Refit {args.model} at each issue date")
    else:
        path = args.headline or latest_headline_parquet(config["forecasting"]["output_dir"])
        headline = pd.read_parquet(path)
        print(f"Headline scores: {path}")
    print(compare(headline, nrcs, args.model).to_string(index=False))
    print("\nNRCS actuals are daily peaks; our actuals are peaks of the monthly mean.")


if __name__ == "__main__":
    main()
