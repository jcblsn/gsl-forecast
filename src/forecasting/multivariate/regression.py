"""Shared ridge estimator for the multivariate models.

The estimator standardizes predictors before applying 1 penalty, then restores their original
scales. With `alpha=None`, it selects the penalty within each fit by generalized
cross-validation.
"""

import logging

import numpy as np
import pandas as pd

TIME_COL = "month"
TARGET_COL = "avg_elevation"
ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0)
# Same-month fits contain about 1 row per year. This floor prevents fits on fewer than 10 rows.
MIN_OBS = 10


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale the columns after the leading intercept column."""
    features = X[:, 1:]
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (features - mean) / scale, mean, scale


def gcv_alpha(Z: np.ndarray, y_centered: np.ndarray, grid=ALPHA_GRID) -> float:
    """Return the penalty with the lowest generalized cross-validation score.

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
    """Fit least squares with a ridge penalty on standardized predictors.

    `X` carries a leading column of ones. The returned coefficients are on the original
    scale, so a caller still evaluates `x @ beta` with the same leading 1. With `alpha` None
    the penalty comes from generalized cross-validation; a float disables the search.
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
    """Return usable predictors and the reason for each exclusion.

    The fit excludes a predictor that is missing at the cutoff, too sparse, or nearly constant.
    The full training frame supplies a unit-free reference scale for the variation check.
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
