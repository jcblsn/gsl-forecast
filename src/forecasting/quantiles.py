"""Empirical prediction intervals and probabilistic scores.

Point forecasters get intervals from their own walk-forward errors: for each horizon the
quantiles of past (actual - pred) are added to the point forecast. Scoring uses pinball loss
averaged over the quantile set and the empirical coverage of the nominal central 90% interval.

The errors are strongly heteroskedastic by issue season, so a band pooled over every issue
month is too wide in one season and too narrow in another. The band is therefore scaled by
the issue season. Coverage is reported by season as well as in aggregate, because an
aggregate coverage near 0.90 hides a season at 0.83 and a season at 0.98.
"""

import numpy as np
import pandas as pd

from src.forecasting.cutoffs import SEASON_MONTHS, issue_season

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# Pseudo-observations that hold a season's width toward the pooled width. A season cell at
# one lead holds about 13 to 22 errors, which is too few to trust its own spread outright.
SEASON_SCALE_PRIOR = 10.0
SEASON_COL = "issue_season"


def qcol(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def with_season(cv_df: pd.DataFrame) -> pd.DataFrame:
    """The CV frame with the issue season of each cutoff attached."""
    if "cutoff" not in cv_df.columns:
        raise ValueError("Interval calibration needs a cutoff column to read the issue season")
    out = cv_df.copy()
    out[SEASON_COL] = pd.to_datetime(out["cutoff"]).map(issue_season)
    return out


def season_shape(err: pd.DataFrame, prior: float = SEASON_SCALE_PRIOR) -> pd.DataFrame:
    """Per model, horizon and season, the centre and the width of the band.

    A season cell holds about 13 to 22 errors at 1 lead, which is too few to read a 5% or a
    95% quantile from. The centre and the width are estimable from that many errors, but the
    shape of the tail is not. Therefore the centre and the width are conditional on the
    season and the shape is pooled.

    The centre is the median of the season's errors and the width is their mean absolute
    deviation from that median. Each is a convex combination of the season's value and the
    pooled value, with weight `n / (n + prior)`. A season with few errors keeps close to the
    pooled value; a season with many moves toward its own.
    """
    pooled = err.groupby(["model", "h"])["err"].median().rename("m0")
    joined = err.join(pooled, on=["model", "h"])
    spread = (
        joined.assign(dev=lambda d: (d["err"] - d["m0"]).abs())
        .groupby(["model", "h"])["dev"]
        .mean()
        .rename("s0")
    )
    cell = err.groupby(["model", "h", SEASON_COL])["err"].agg(["median", "size"])
    cell = cell.rename(columns={"median": "m_season", "size": "n"})
    cell["d_season"] = (
        err.join(cell["m_season"], on=["model", "h", SEASON_COL])
        .assign(dev=lambda d: (d["err"] - d["m_season"]).abs())
        .groupby(["model", "h", SEASON_COL])["dev"]
        .mean()
    )
    cell = cell.join(pooled, on=["model", "h"]).join(spread, on=["model", "h"])
    w = cell["n"] / (cell["n"] + prior)
    cell["center"] = cell["m0"] + w * (cell["m_season"] - cell["m0"])
    cell["width"] = cell["s0"] + w * (cell["d_season"] - cell["s0"])
    return cell[["center", "width"]]


def error_quantiles(
    cv_df: pd.DataFrame, quantiles=QUANTILES, prior: float = SEASON_SCALE_PRIOR
) -> pd.DataFrame:
    """Per model, issue season and horizon, quantiles of (actual - pred) from CV errors.

    `season_shape` gives the centre and the width of each season's band. Every error is then
    standardised by the centre and the width of its own season, and the quantiles of those
    standardised errors are pooled over the seasons. A season's band is its centre plus its
    width multiplied by that pooled shape.

    Standardising before pooling matters. Pooling the raw errors would give every season the
    tail of whichever season has the widest errors, so a band scaled down for a narrow season
    would keep a skew the narrow season does not have.

    The result carries every season in `SEASON_MONTHS`, so a forecast issued in a season the
    cross-validation never covered still gets a band.
    """
    err = with_season(cv_df).assign(err=lambda d: d["actual"] - d["pred"])
    shape = season_shape(err, prior)
    keys = ["model", "h", SEASON_COL]
    z = err.join(shape, on=keys)
    # A season whose errors are all the same has 0 width. Its standardised error is then 0
    # and its band collapses onto its centre, which is what a constant error deserves.
    z["z"] = (z["err"] - z["center"]) / z["width"].where(z["width"] > 0, 1.0)
    pooled = z.groupby(["model", "h"])["z"].quantile(list(quantiles)).unstack()
    pooled.columns = [qcol(q) for q in quantiles]
    frames = []
    for season in SEASON_MONTHS:
        cell = (
            shape.xs(season, level=SEASON_COL)
            if season in shape.index.get_level_values(SEASON_COL)
            else None
        )
        block = pooled.copy()
        center = cell["center"].reindex(block.index) if cell is not None else 0.0
        width = cell["width"].reindex(block.index) if cell is not None else 1.0
        if cell is not None:
            fallback = shape.groupby(["model", "h"])[["center", "width"]].mean()
            center = center.fillna(fallback["center"].reindex(block.index))
            width = width.fillna(fallback["width"].reindex(block.index))
        for q in quantiles:
            block[qcol(q)] = center + width * block[qcol(q)]
        block[SEASON_COL] = season
        frames.append(block.reset_index())
    return pd.concat(frames, ignore_index=True)


def apply_intervals(
    preds: pd.DataFrame, eq: pd.DataFrame, model: str, season: str | None = None
) -> pd.DataFrame:
    """Attach quantile columns to a point forecast frame with an `h` column.

    `season` names the issue season the band is calibrated for. Without it the frame must
    carry a `cutoff` column, and the season comes from the first cutoff in it.
    """
    if season is None:
        season = issue_season(pd.to_datetime(preds["cutoff"]).iloc[0])
    rows = eq[(eq["model"] == model) & (eq[SEASON_COL] == season)].set_index("h")
    cols = [c for c in rows.columns if c.startswith("q") and c[1:].isdigit()]
    out = preds.copy()
    for c in cols:
        out[c] = out["pred"] + out["h"].map(rows[c]).to_numpy()
    return out


def pinball(actual: np.ndarray, pred_q: np.ndarray, q: float) -> np.ndarray:
    diff = actual - pred_q
    return np.maximum(q * diff, (q - 1) * diff)


def probabilistic_scores(
    scored: pd.DataFrame, quantiles=QUANTILES, by: tuple[str, ...] = ("model", "h")
) -> pd.DataFrame:
    """Mean pinball loss across quantiles, central-90% coverage and band width, per group."""
    losses = np.column_stack(
        [pinball(scored["actual"].to_numpy(), scored[qcol(q)].to_numpy(), q) for q in quantiles]
    )
    frame = scored[list(by)].copy()
    frame["mean_pinball_loss"] = losses.mean(axis=1)
    frame["in90"] = (scored["actual"] >= scored[qcol(0.05)]) & (
        scored["actual"] <= scored[qcol(0.95)]
    )
    frame["width90"] = scored[qcol(0.95)] - scored[qcol(0.05)]
    return (
        frame.groupby(list(by))
        .agg(
            mean_pinball_loss=("mean_pinball_loss", "mean"),
            cov90=("in90", "mean"),
            width90=("width90", "mean"),
            n_scored=("in90", "size"),
        )
        .reset_index()
    )


def leave_one_year_out_intervals(cv_df: pd.DataFrame, quantiles=QUANTILES) -> pd.DataFrame:
    """Intervals for every CV row, each calibrated from the errors of the other years."""
    df = with_season(cv_df)
    df["year"] = pd.to_datetime(df["cutoff"]).dt.year
    frames = []
    for year in sorted(df["year"].unique()):
        held = df[df["year"] == year]
        eq = error_quantiles(df[df["year"] != year], quantiles)
        frames += [
            apply_intervals(g, eq, model, season)
            for (model, season), g in held.groupby(["model", SEASON_COL])
        ]
    return pd.concat(frames, ignore_index=True)


def leave_one_year_out_scores(cv_df: pd.DataFrame, quantiles=QUANTILES) -> pd.DataFrame:
    """Aggregate interval scores per model and lead, from held-out-year intervals."""
    scored = leave_one_year_out_intervals(cv_df, quantiles)
    return probabilistic_scores(scored, quantiles).drop(columns="n_scored")


def season_scores(cv_df: pd.DataFrame, quantiles=QUANTILES) -> pd.DataFrame:
    """Interval scores per model, issue season and lead, from held-out-year intervals.

    An aggregate coverage near 0.90 hides a season at 0.83 and a season at 0.98, so this is
    the table that says whether the band is calibrated where a decision is made.
    """
    scored = leave_one_year_out_intervals(cv_df, quantiles)
    return probabilistic_scores(scored, quantiles, by=("model", SEASON_COL, "h"))
