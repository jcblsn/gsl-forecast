"""Calibration and prediction for the versioned SWE and ETS blend."""

from __future__ import annotations

import numpy as np
import pandas as pd

BLEND_MODEL = "swe_ets_blend_v1"
SNOW_MODEL = "swe_head"
ETS_MODEL = "ets_damped_s12"
WEIGHT_GRID = np.linspace(0.0, 1.0, 101)
MIN_CALIBRATION_ROWS = 20

SEASON_MONTHS = {
    "accumulation": {11, 12, 1, 2, 3},
    "melt": {4, 5, 6},
    "recession": {7, 8, 9, 10},
}


def issue_season(cutoff: pd.Timestamp) -> str:
    """Return the water-year stage for the issue after a cutoff."""
    issue_month = cutoff.month % 12 + 1
    return next(name for name, months in SEASON_MONTHS.items() if issue_month in months)


def paired_predictions(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Align the 2 component forecasts on cutoff and lead."""
    wanted = cv_df[cv_df["model"].isin([SNOW_MODEL, ETS_MODEL])]
    if wanted.empty:
        return pd.DataFrame()
    paired = wanted.pivot_table(
        index=["cutoff", "h", "actual"], columns="model", values="pred", aggfunc="first"
    ).reset_index()
    if SNOW_MODEL not in paired or ETS_MODEL not in paired:
        return pd.DataFrame()
    paired = paired.dropna(subset=[SNOW_MODEL, ETS_MODEL]).copy()
    paired["cutoff"] = pd.to_datetime(paired["cutoff"])
    paired["season"] = paired["cutoff"].map(issue_season)
    return paired


def _monotone_weights(loss: np.ndarray) -> np.ndarray:
    """Find the minimum-loss nonincreasing path through the weight grid."""
    n_h, n_weights = loss.shape
    cost = np.full((n_h, n_weights), np.inf)
    previous = np.zeros((n_h, n_weights), dtype=int)
    cost[0] = loss[0]
    for i in range(1, n_h):
        for current in range(n_weights):
            allowed = cost[i - 1, current:]
            prior = current + int(np.argmin(allowed))
            cost[i, current] = loss[i, current] + cost[i - 1, prior]
            previous[i, current] = prior
    selected = np.zeros(n_h, dtype=int)
    selected[-1] = int(np.argmin(cost[-1]))
    for i in range(n_h - 1, 0, -1):
        selected[i - 1] = previous[i, selected[i]]
    return WEIGHT_GRID[selected]


def fit_weights(cv_df: pd.DataFrame, min_rows: int = MIN_CALIBRATION_ROWS) -> pd.DataFrame:
    """Fit 3 seasonal weight curves by walk-forward absolute error."""
    paired = paired_predictions(cv_df)
    if paired.empty:
        raise ValueError(f"CV results must contain {SNOW_MODEL} and {ETS_MODEL}")
    horizons = sorted(int(h) for h in paired["h"].unique())
    if horizons != list(range(1, max(horizons) + 1)):
        raise ValueError("CV results must contain consecutive forecast leads")

    rows = []
    for season in SEASON_MONTHS:
        losses = []
        counts = []
        for h in horizons:
            sample = paired[(paired["season"] == season) & (paired["h"] == h)]
            if len(sample) < min_rows:
                raise ValueError(f"{season} lead {h} has {len(sample)} rows; {min_rows} required")
            snow = sample[SNOW_MODEL].to_numpy(dtype=float)
            ets = sample[ETS_MODEL].to_numpy(dtype=float)
            actual = sample["actual"].to_numpy(dtype=float)
            prediction = ets[:, None] + WEIGHT_GRID[None, :] * (snow - ets)[:, None]
            losses.append(np.abs(actual[:, None] - prediction).sum(axis=0))
            counts.append(len(sample))
        weights = _monotone_weights(np.asarray(losses))
        rows.extend(
            {
                "season": season,
                "h": h,
                "swe_weight": float(weight),
                "n": count,
            }
            for h, weight, count in zip(horizons, weights, counts, strict=True)
        )
    return pd.DataFrame(rows)


def blend_pairs(paired: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted weights to aligned component predictions."""
    joined = paired.merge(weights[["season", "h", "swe_weight"]], on=["season", "h"])
    pred = joined[ETS_MODEL] + joined["swe_weight"] * (joined[SNOW_MODEL] - joined[ETS_MODEL])
    out = pd.DataFrame(
        {
            "model": BLEND_MODEL,
            "cutoff": joined["cutoff"],
            "h": joined["h"].astype(int),
            "pred": pred,
            "actual": joined["actual"],
        }
    )
    error = out["pred"] - out["actual"]
    out["abs_error"] = error.abs()
    out["sq_error"] = error**2
    return out


def cross_fitted_predictions(
    cv_df: pd.DataFrame, min_rows: int = MIN_CALIBRATION_ROWS
) -> pd.DataFrame:
    """Fit weights without the cutoff year that receives each prediction."""
    paired = paired_predictions(cv_df)
    if paired.empty:
        return pd.DataFrame()
    frames = []
    years = paired["cutoff"].dt.year
    for year in sorted(years.unique()):
        training = cv_df[pd.to_datetime(cv_df["cutoff"]).dt.year != year]
        weights = fit_weights(training, min_rows=min_rows)
        frames.append(blend_pairs(paired[years == year], weights))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def blend_forward(predictions: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Blend the 2 current component paths with the issue season weights."""
    wide = predictions[predictions["model_name"].isin([SNOW_MODEL, ETS_MODEL])].pivot(
        index="month", columns="model_name", values="pred"
    )
    if SNOW_MODEL not in wide or ETS_MODEL not in wide:
        raise ValueError(f"Predictions must contain {SNOW_MODEL} and {ETS_MODEL}")
    wide = wide.dropna().sort_index().reset_index()
    origin = pd.Timestamp(wide["month"].min()) - pd.DateOffset(months=1)
    season = issue_season(origin)
    wide["h"] = range(1, len(wide) + 1)
    selected = weights[weights["season"] == season][["h", "swe_weight"]]
    wide = wide.merge(selected, on="h", validate="one_to_one")
    pred = wide[ETS_MODEL] + wide["swe_weight"] * (wide[SNOW_MODEL] - wide[ETS_MODEL])
    return pd.DataFrame(
        {
            "month": pd.to_datetime(wide["month"]),
            "target": "avg_elevation",
            "pred": pred,
            "model_name": BLEND_MODEL,
        }
    )


def blend_contributions(
    snow_terms: pd.DataFrame, predictions: pd.DataFrame, weights: pd.DataFrame
) -> pd.DataFrame:
    """Scale snow-model terms and put the ETS share in the reference path."""
    snow = predictions[predictions["model_name"] == SNOW_MODEL].sort_values("month")
    ets = predictions[predictions["model_name"] == ETS_MODEL].sort_values("month")
    if len(snow) != len(ets):
        raise ValueError("The component forecast paths must have equal lengths")
    origin = pd.Timestamp(snow["month"].min()) - pd.DateOffset(months=1)
    season = issue_season(origin)
    selected = weights[weights["season"] == season].set_index("h")["swe_weight"]
    ets_by_h = pd.Series(ets["pred"].to_numpy(dtype=float), index=range(1, len(ets) + 1))

    out = snow_terms.copy()
    out["swe_weight"] = out["h"].map(selected)
    out["contribution_ft"] = out["contribution_ft"] * out["swe_weight"]
    reference = out["input"] == "reference_path"
    out.loc[reference, "contribution_ft"] += (1.0 - out.loc[reference, "swe_weight"]) * out.loc[
        reference, "h"
    ].map(ets_by_h)
    return out


def weight_metadata(weights: pd.DataFrame, cv_df: pd.DataFrame) -> dict:
    """Return the calibration details required to reproduce an issue."""
    cutoffs = pd.to_datetime(cv_df["cutoff"])
    scored = cv_df[cv_df["model"] == BLEND_MODEL].copy()
    if not scored.empty:
        if "abs_error" not in scored:
            scored["abs_error"] = (scored["pred"] - scored["actual"]).abs()
        held_out_mae = [
            {
                "h": int(h),
                "mae_ft": round(float(group["abs_error"].mean()), 3),
                "n": int(len(group)),
            }
            for h, group in scored.groupby("h", sort=True)
        ]
    else:
        held_out_mae = []
    curves = {
        season: [round(float(v), 2) for v in group.sort_values("h")["swe_weight"]]
        for season, group in weights.groupby("season", sort=False)
    }
    return {
        "algorithm_version": 1,
        "model": BLEND_MODEL,
        "components": [SNOW_MODEL, ETS_MODEL],
        "objective": "walk-forward MAE",
        "weight_step": 0.01,
        "constraint": "nonincreasing by lead",
        "cutoff_min": str(cutoffs.min().date()),
        "cutoff_max": str(cutoffs.max().date()),
        "n_cutoffs": int(cutoffs.nunique()),
        "held_out_mae": held_out_mae,
        "weights": curves,
    }
