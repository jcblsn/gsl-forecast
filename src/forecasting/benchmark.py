"""Compare maximum April-June monthly-mean forecasts with the published NRCS record."""

import argparse
import os

import duckdb
import pandas as pd

from src.config import load_config
from src.forecasting.cross_validate import evaluate_at_cutoff
from src.forecasting.data import load_monthly_data
from src.forecasting.headline import APR_JUN_MONTHLY_MEAN_MAX, ISSUE_LABELS, headline_scores
from src.forecasting.multivariate.inflow_chain import InflowChainForecaster
from src.forecasting.registry import all_forecasters

BENCHMARK_CSV = os.path.join("data", "benchmarks", "nrcs_outlooks.csv")
MODEL = "ets_damped_s12"


def compare(
    headline: pd.DataFrame,
    nrcs: pd.DataFrame,
    model: str = MODEL,
    inflow: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Align NRCS outlooks and one fixed project model by issue date.

    NRCS errors use daily peaks; project errors use the maximum April-June monthly mean.
    The final error column evaluates the NRCS estimate against the monthly-mean target.
    """
    nrcs = nrcs.copy()
    nrcs["issue_date"] = pd.to_datetime(nrcs["issue_date"])
    nrcs["issue"] = nrcs["issue_date"].dt.month.map(ISSUE_LABELS)
    nrcs["water_year"] = nrcs["issue_date"].dt.year
    peaks = headline[
        (headline["target"] == APR_JUN_MONTHLY_MEAN_MAX) & (headline["model"] == model)
    ]
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
                    "our_apr_jun_monthly_mean_max": round(best["pred"], 2),
                    "actual_apr_jun_monthly_mean_max": round(best["actual"], 2),
                    "our_error": round(best["pred"] - best["actual"], 2),
                    "nrcs_error_vs_apr_jun_monthly_mean_max": (
                        round(r["implied_peak_ft"] - best["actual"], 2)
                        if pd.notna(r["implied_peak_ft"])
                        else float("nan")
                    ),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    if inflow is not None and not inflow.empty:
        out = out.merge(inflow, on="issue_date", how="left")
    return out


def refit_at_issues(nrcs: pd.DataFrame, data: pd.DataFrame, train_start: str | None, model: str):
    """Fit the model at each NRCS issue date and score its April-June monthly-mean maximum.

    Uses the data as it stands today rather than the vintage available at the time, so this
    is a hindcast of the method, not a record of what was issued.
    """
    forecaster = next(f for f in all_forecasters() if f.name == model)
    frames = []
    for issue in pd.to_datetime(nrcs["issue_date"]).unique():
        cutoff = pd.Timestamp(issue) - pd.DateOffset(months=1)
        if cutoff not in set(data["month"]):
            continue
        frames.append(evaluate_at_cutoff(data, cutoff, [forecaster], 8, train_start))
    cv = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return headline_scores(cv, data)


def issued_inflow(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The published median Great Salt Lake inflow forecast and its period, per issue date."""
    out = conn.execute(
        """
        SELECT publication_date AS issue_date, period_start, period_end,
               kaf AS nrcs_inflow_p50_kaf, ROUND(100 * kaf / normal_kaf) AS nrcs_pct_normal
        FROM nrcs_inflow_forecasts WHERE exceedance = 50 ORDER BY 1
        """
    ).fetchdf()
    out["issue_date"] = pd.to_datetime(out["issue_date"]).dt.date
    return out


def seasonal_inflow(data: pd.DataFrame, issued: pd.DataFrame, train_start: str | None):
    """inflow_chain's stage-one volume over each issued forecast's period, and the gauged
    volume for the same months, so the two inflow forecasts share an actual."""
    rows = []
    for _, r in issued.iterrows():
        issue = pd.Timestamp(r["issue_date"])
        cutoff = issue - pd.DateOffset(months=1)
        first, last = (int(r[c][:2]) for c in ("period_start", "period_end"))
        months = [pd.Timestamp(year=issue.year, month=m, day=1) for m in range(first, last + 1)]
        train = data[
            (data["month"] <= cutoff) & (data["month"] >= pd.Timestamp(train_start or "1800"))
        ]
        if train.empty or train["month"].iloc[-1] != cutoff:
            continue
        model = InflowChainForecaster().fit(train)
        leads = [(m.year - cutoff.year) * 12 + m.month - cutoff.month for m in months]
        obs = data.set_index("month").loc[[m for m in months if m in set(data["month"])]]
        rows.append(
            {
                "issue_date": r["issue_date"],
                "our_inflow_kaf": round(sum(model.inflow_forecast(h) for h in leads if h >= 1)),
                "actual_inflow_kaf": (
                    round(obs["inflow_kaf_total"].sum()) if len(obs) == len(months) else None
                ),
            }
        )
    return issued.drop(columns=["period_start", "period_end"]).merge(
        pd.DataFrame(rows), on="issue_date", how="left"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare spring-maximum refits with published NRCS outlooks"
    )
    parser.add_argument("--benchmark", default=BENCHMARK_CSV)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    db_path = config["database"]["path"]
    train_start = config["forecasting"]["train_start"]
    nrcs = pd.read_csv(args.benchmark)
    data = load_monthly_data(db_path, train_start)
    with duckdb.connect(db_path, read_only=True) as conn:
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        issued = issued_inflow(conn) if "nrcs_inflow_forecasts" in tables else pd.DataFrame()
    inflow = seasonal_inflow(data, issued, train_start) if not issued.empty else None
    headline = refit_at_issues(nrcs, data, train_start, args.model)
    print(
        f"Refit {args.model} at each issue date. NRCS errors use daily peaks; "
        "project errors use the maximum April–June monthly mean."
    )
    pd.set_option("display.width", 200)
    print(compare(headline, nrcs, args.model, inflow).to_string(index=False))


if __name__ == "__main__":
    main()
