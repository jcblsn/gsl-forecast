"""Shared estimator for the multivariate models.

The design matrix mixes columns on very different scales: the lake level is about 4192 feet
and the snowpack is about 10 inches. A single penalty cannot act on both columns, so the
estimator standardises the design, solves the penalised system, and maps the coefficients
back. Each fit has 32 to 47 rows and 4 parameters, so the penalty is chosen per fit by
generalised cross-validation over the rows of that fit.
"""

import logging

import numpy as np
import pandas as pd

TIME_COL = "month"
TARGET_COL = "avg_elevation"
ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0)
# One fit reads the rows of a single calendar month, so the record supplies about 1 row per
# year. 10 rows against 4 or 5 parameters is thin, and the review calls the bar too
# permissive. Raising it was measured and reverted: the bar never binds on an outer fit
# (swe_head scores identically at 10, 15 and 20), and where it does bind, on the early inner
# cutoffs of the blend's weight pass, it degrades the weights instead of protecting a fit.
# The blend's lead-6 MAE rises from 0.578 to 0.595 to 0.626 ft at 10, 15 and 20. The real
# repair is a pooled model across issue months and leads, not a higher bar on 288 fits.
MIN_OBS = 10


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and scale the columns after the leading intercept column."""
    features = X[:, 1:]
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (features - mean) / scale, mean, scale


def gcv_alpha(Z: np.ndarray, y_centered: np.ndarray, grid=ALPHA_GRID) -> float:
    """The penalty in `grid` with the lowest generalised cross-validation score.

    The score is n * RSS / (n - df)^2, where df counts the intercept plus the ridge trace.
    Only the rows of this fit enter, so the choice uses no data after the cutoff.
    """
    n = len(y_centered)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    uy = U.T @ y_centered
    best, best_score = grid[0], np.inf
    for alpha in grid:
        shrink = s**2 / (s**2 + alpha)
        residual = y_centered - U @ (shrink * uy)
        df = shrink.sum() + 1.0
        if n - df <= 1e-6:
            continue
        score = n * float(residual @ residual) / (n - df) ** 2
        if score < best_score:
            best, best_score = alpha, score
    return best


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float | None = None) -> np.ndarray:
    """Least squares with a ridge penalty on the standardised non-intercept columns.

    `X` carries a leading column of ones. The returned coefficients are on the original
    scale, so a caller still evaluates `x @ beta` with the same leading 1. With `alpha` None
    the penalty comes from generalised cross-validation; a float switches the search off.
    """
    y = np.asarray(y, dtype=float)
    if X.shape[1] == 1:
        return np.array([y.mean()])
    Z, mean, scale = _standardize(X)
    y_mean = y.mean()
    y_centered = y - y_mean
    penalty = gcv_alpha(Z, y_centered) if alpha is None else float(alpha)
    gram = Z.T @ Z + penalty * np.eye(Z.shape[1])
    coef = np.linalg.solve(gram, Z.T @ y_centered) / scale
    return np.concatenate([[y_mean - coef @ mean], coef])


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Data lacks covariate columns {missing}; run gsl-pipeline first")


def design(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame[features].to_numpy(dtype=float)])


# A feature whose standard deviation at one issue month is this small a share of its
# standard deviation over the whole training frame is structurally absent at that month.
NEAR_ZERO_SD_RATIO = 0.01


def select_features(
    rows: pd.DataFrame,
    features: list[str],
    now: pd.Series,
    min_obs: int,
    scale_reference: pd.DataFrame,
) -> tuple[list[str], dict[str, str]]:
    """The features one fit may use, and the reason it drops each of the others.

    A fit drops a feature that is NULL at the cutoff, that too few training rows carry, or
    that barely varies among those rows. Dropping one feature no longer drops the others.

    The variation rule matters because features depend on the issue season. Snow water
    equivalent is structurally 0 at an August cutoff. The standardised ridge divides by that
    near-zero standard deviation and returns a coefficient of hundreds of feet per inch. The
    forecast contribution stays small because the input is near 0, but the coefficient is a
    diagnostic failure. `scale_reference` gives the same column over every month, so the
    rule compares a season with the whole record and needs no unit-specific threshold.
    """
    kept, dropped = [], {}
    for feature in features:
        values = rows[feature]
        n = int(values.notna().sum())
        sd = float(values.std()) if n > 1 else 0.0
        reference = float(scale_reference[feature].std())
        if pd.isna(now[feature]):
            dropped[feature] = "it is NULL at the cutoff"
        elif n < min_obs:
            dropped[feature] = f"only {n} training rows carry it, below min_obs={min_obs}"
        elif not np.isfinite(sd) or sd <= NEAR_ZERO_SD_RATIO * reference:
            dropped[feature] = "it barely varies at this issue month"
        else:
            kept.append(feature)
    return kept, dropped


def log_fallback(model: str, lead: int, dropped: dict[str, str]) -> None:
    """Record every dropped feature, so a degraded fit is visible in the log."""
    for feature, reason in dropped.items():
        logging.debug(f"{model} lead {lead}: dropped {feature} because {reason}")
