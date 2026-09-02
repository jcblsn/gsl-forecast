import numpy as np
import pandas as pd

TIME_COL = "month"
TARGET_COL = "avg_elevation"


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Least squares with a small ridge penalty on the non-intercept columns."""
    penalty = alpha * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Data lacks covariate columns {missing}; run gsl-pipeline first")


def design(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame[features].to_numpy(dtype=float)])
