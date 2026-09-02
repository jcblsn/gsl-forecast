"""Headline scalars scored from walk-forward CV output.

The two numbers people act on are the spring peak (April-June maximum of the monthly mean)
and the water-year-end level (the September mean). This module extracts them from per-cutoff
CV predictions for cutoffs that correspond to the operational issue dates (data through
December-April, i.e. outlooks issued January 1 through May 1).
"""

import pandas as pd

PEAK_MONTHS = (4, 5, 6)
WY_END_MONTH = 9
ISSUE_LABELS = {1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may"}


def issue_month(cutoff: pd.Timestamp) -> int:
    """The outlook issued on the first of the month after the cutoff's month."""
    return cutoff.month % 12 + 1


def headline_scores(cv_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """One row per model, cutoff and target with the predicted and actual scalar.

    `data` is the full monthly series (columns month, avg_elevation); it supplies months that
    were already observed at the cutoff, so an April-cutoff peak includes the observed April.
    """
    if cv_df.empty:
        return pd.DataFrame(
            columns=["model", "cutoff", "issue", "water_year", "target", "pred", "actual"]
        )
    obs = data.set_index(pd.to_datetime(data["month"]))["avg_elevation"]
    rows = []
    for (model, cutoff), grp in cv_df.groupby(["model", "cutoff"]):
        cutoff = pd.Timestamp(cutoff)
        issue = issue_month(cutoff)
        if issue not in ISSUE_LABELS:
            continue
        year = cutoff.year if cutoff.month < 12 else cutoff.year + 1
        months = cutoff + pd.DateOffset(months=1) * grp["h"].values
        pred = pd.Series(grp["pred"].values, index=pd.DatetimeIndex(months))
        actual = pd.Series(grp["actual"].values, index=pd.DatetimeIndex(months))
        known = obs[(obs.index <= cutoff) & (obs.index.year == year)]
        for target, wanted in (("peak", PEAK_MONTHS), ("wy_end", (WY_END_MONTH,))):
            sel = [pd.Timestamp(year=year, month=m, day=1) for m in wanted]
            sel = [pd.Timestamp(s).to_period("M") for s in sel]
            p_future = pred[pred.index.to_period("M").isin(sel)]
            a_future = actual[actual.index.to_period("M").isin(sel)]
            k = known[known.index.to_period("M").isin(sel)]
            if len(p_future) + len(k) < len(wanted):
                continue
            rows.append(
                {
                    "model": model,
                    "cutoff": cutoff,
                    "issue": ISSUE_LABELS[issue],
                    "water_year": year,
                    "target": target,
                    "pred": max(p_future.max(), k.max()) if len(k) else p_future.max(),
                    "actual": max(a_future.max(), k.max()) if len(k) else a_future.max(),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_error"] = (out["pred"] - out["actual"]).abs()
    return out


def summarize_headline(scores: pd.DataFrame) -> pd.DataFrame:
    """MAE and count per model, issue month and target."""
    if scores.empty:
        return pd.DataFrame(columns=["model", "issue", "target", "mae", "n"])
    return (
        scores.groupby(["model", "issue", "target"])
        .agg(mae=("abs_error", "mean"), n=("abs_error", "size"))
        .reset_index()
    )


def headline_metrics(summary: pd.DataFrame, model: str) -> dict[str, float]:
    """Tracker metric names like peak_mae_apr and wyend_mae_jan."""
    out = {}
    for _, r in summary[summary["model"] == model].iterrows():
        key = "peak" if r["target"] == "peak" else "wyend"
        out[f"{key}_mae_{r['issue']}"] = float(r["mae"])
    return out


def print_headline(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("\nNo headline scalars scored (no January-May issue cutoffs with a full target).")
        return
    order = list(ISSUE_LABELS.values())
    for target, title in (("peak", "Spring peak"), ("wy_end", "Water-year-end")):
        sub = summary[summary["target"] == target]
        if sub.empty:
            continue
        pivot = sub.pivot(index="model", columns="issue", values="mae").round(3)
        pivot = pivot[[c for c in order if c in pivot.columns]]
        pivot.columns = [f"{c} 1" for c in pivot.columns]
        print(f"\n{title} MAE (ft) by issue date:")
        print(pivot.to_string())
