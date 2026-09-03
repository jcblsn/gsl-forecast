"""Empirical prediction intervals and probabilistic scores.

Point forecasters get intervals from their own walk-forward errors: for each horizon the
quantiles of past (actual - pred) are added to the point forecast. Scoring uses pinball loss
averaged over the quantile set and the empirical coverage of the nominal central 90% interval.
"""

import numpy as np
import pandas as pd

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def qcol(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def error_quantiles(cv_df: pd.DataFrame, quantiles=QUANTILES) -> pd.DataFrame:
    """Per model and horizon, quantiles of (actual - pred) from CV errors."""
    err = cv_df.assign(err=cv_df["actual"] - cv_df["pred"])
    out = err.groupby(["model", "h"])["err"].quantile(list(quantiles)).unstack()
    out.columns = [qcol(q) for q in quantiles]
    return out.reset_index()


def apply_intervals(preds: pd.DataFrame, eq: pd.DataFrame, model: str) -> pd.DataFrame:
    """Attach quantile columns to a point forecast frame with an `h` column."""
    rows = eq[eq["model"] == model].set_index("h")
    cols = [c for c in rows.columns if c.startswith("q")]
    out = preds.copy()
    for c in cols:
        out[c] = out["pred"] + out["h"].map(rows[c]).to_numpy()
    return out


def pinball(actual: np.ndarray, pred_q: np.ndarray, q: float) -> np.ndarray:
    diff = actual - pred_q
    return np.maximum(q * diff, (q - 1) * diff)


def probabilistic_scores(scored: pd.DataFrame, quantiles=QUANTILES) -> pd.DataFrame:
    """Mean pinball loss across quantiles and central-90% coverage, per model and lead."""
    losses = np.column_stack(
        [pinball(scored["actual"].to_numpy(), scored[qcol(q)].to_numpy(), q) for q in quantiles]
    )
    frame = scored[["model", "h"]].copy()
    frame["mean_pinball_loss"] = losses.mean(axis=1)
    frame["in90"] = (scored["actual"] >= scored[qcol(0.05)]) & (
        scored["actual"] <= scored[qcol(0.95)]
    )
    return (
        frame.groupby(["model", "h"])
        .agg(mean_pinball_loss=("mean_pinball_loss", "mean"), cov90=("in90", "mean"))
        .reset_index()
    )


def leave_one_year_out_scores(cv_df: pd.DataFrame, quantiles=QUANTILES) -> pd.DataFrame:
    """Score intervals honestly: each cutoff's interval comes from errors of other years."""
    df = cv_df.copy()
    df["year"] = pd.to_datetime(df["cutoff"]).dt.year
    frames = []
    for year in sorted(df["year"].unique()):
        held = df[df["year"] == year]
        eq = error_quantiles(df[df["year"] != year], quantiles)
        scored = pd.concat(
            [apply_intervals(g, eq, m) for m, g in held.groupby("model")], ignore_index=True
        )
        frames.append(scored)
    return probabilistic_scores(pd.concat(frames, ignore_index=True), quantiles)
