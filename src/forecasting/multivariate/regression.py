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


def fallback_reason(n_complete: int, min_obs: int, features_now: bool) -> str | None:
    """Why a fit must drop its covariates, or None when the full fit is available.

    The rule is declared here so a degraded fit is visible instead of being a side effect of
    a NULL check. Percent-of-median snowpack, for example, is NULL from June to September.
    """
    if not features_now:
        return "a feature is NULL at the cutoff"
    if n_complete < min_obs:
        return f"only {n_complete} training rows carry every feature, below min_obs={min_obs}"
    return None


def log_fallback(model: str, lead: int, reason: str) -> None:
    logging.debug(f"{model} lead {lead}: covariates dropped because {reason}")
