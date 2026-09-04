"""Score every dated forecast in forecasts/ against what the gauge has since recorded."""

import argparse
import glob
import json
import os

import pandas as pd

from src.config import load_config
from src.forecasting.data import load_monthly_data
from src.forecasting.quantiles import qcol
from src.forecasting.run_forecast import ISSUE_METADATA_SCHEMA_VERSION, ISSUE_STATUSES


def read_issue_metadata(path: str) -> dict:
    """Read and validate the required provenance sidecar for one dated CSV."""
    metadata_path = os.path.splitext(path)[0] + ".meta.json"
    try:
        with open(metadata_path) as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing or malformed issue metadata: {metadata_path}") from error
    required = {
        "schema_version",
        "issue",
        "issue_status",
        "forecast_version",
        "code_commit",
        "code_dirty",
        "evaluation_policy_version",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Issue metadata {metadata_path} is missing {missing}")
    if metadata["schema_version"] != ISSUE_METADATA_SCHEMA_VERSION:
        raise ValueError(f"Unsupported issue metadata schema in {metadata_path}")
    if metadata["issue_status"] not in ISSUE_STATUSES:
        raise ValueError(f"Invalid issue_status in {metadata_path}")
    for field in ("issue", "forecast_version", "code_commit", "evaluation_policy_version"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f"Invalid {field} in {metadata_path}")
    if not isinstance(metadata["code_dirty"], bool) and metadata["code_dirty"] != "unknown":
        raise ValueError(f"Invalid code_dirty in {metadata_path}")
    return metadata


def load_issued(forecast_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(forecast_dir, "*.csv")))
    paths = [p for p in paths if os.path.basename(p)[0].isdigit()]
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        metadata = read_issue_metadata(path)
        frame = pd.read_csv(path)
        issue_values = pd.to_datetime(frame["issue"]).dt.strftime("%Y-%m-%d").unique()
        if len(issue_values) != 1 or issue_values[0] != metadata["issue"]:
            raise ValueError(f"Issue date in {path} does not match its metadata")
        for field in (
            "issue_status",
            "forecast_version",
            "evaluation_policy_version",
            "code_commit",
            "code_dirty",
        ):
            frame[field] = metadata[field]
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
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
    dimensions = ["issue_status", "forecast_version", "model", "h"]
    return scored.groupby(dimensions).agg(**agg).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score dated forecasts against observed monthly means"
    )
    parser.add_argument("--config")
    parser.add_argument("--forecast-dir", default="forecasts")
    parser.add_argument("--out", default=os.path.join("forecasts", "verification.csv"))
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        issued = load_issued(args.forecast_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error
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
