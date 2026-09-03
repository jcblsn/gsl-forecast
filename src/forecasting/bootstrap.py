"""Block-bootstrap uncertainty for the cross-validation numbers.

The 157 development cutoffs overlap heavily. A 24-month forecast from one cutoff shares 23
of its target months with the next cutoff, so 157 errors are nowhere near 157 independent
cases: the cohort spans about 13 hydrologic years. Treating the errors as independent would
make a 0.01 ft difference between 2 models look decisive.

A circular moving-block bootstrap over the cutoff sequence keeps that dependence. It
resamples runs of consecutive cutoffs rather than single cutoffs, so a resample carries the
serial structure of the lake and of the weather that drives it. The block length is 24
months, which is the forecast horizon and therefore the span over which 2 cutoffs can share
a target month.

These are descriptive sensitivity estimates. They are not formal sampling intervals under a
fully specified data-generating process, and they do not prove stationarity. They are a
better statement of evidential precision than a 3-decimal rank table.
"""

import numpy as np
import pandas as pd

BLOCK_MONTHS = 24
DRAWS = 5000
LEVEL = 0.95
MAE_COLUMNS = ("model", "h", "mae", "lo", "hi", "n_cutoffs")
IMPROVEMENT_COLUMNS = (
    "model",
    "baseline",
    "h",
    "improvement",
    "lo",
    "hi",
    "excludes_zero",
    "n_cutoffs",
)


def circular_block_index(n: int, block: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """A `draws` by `n` matrix of positions from a circular moving-block resample.

    Each row is built from `ceil(n / block)` runs of `block` consecutive positions. A run
    starts anywhere in the series and wraps past the end, so every position starts a run
    equally often and no position is under-represented at the edges.
    """
    block = min(block, n)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(draws, n_blocks, 1))
    offsets = np.arange(block).reshape(1, 1, block)
    return ((starts + offsets) % n).reshape(draws, n_blocks * block)[:, :n]


def _error_matrix(cv_df: pd.DataFrame, models: list[str], h: int) -> tuple[list, np.ndarray]:
    """Absolute errors as a cutoff-by-model matrix, over the cutoffs every model scored."""
    lead = cv_df[(cv_df["h"] == h) & cv_df["model"].isin(models)]
    wide = lead.pivot_table(index="cutoff", columns="model", values="abs_error")
    wide = wide.reindex(columns=models).dropna()
    return list(wide.index), wide.to_numpy(dtype=float)


def mae_intervals(
    cv_df: pd.DataFrame,
    models: list[str] | None = None,
    leads: list[int] | None = None,
    block: int = BLOCK_MONTHS,
    draws: int = DRAWS,
    level: float = LEVEL,
    seed: int = 0,
) -> pd.DataFrame:
    """Per model and lead: MAE with a circular moving-block bootstrap interval."""
    models = models or sorted(cv_df["model"].unique())
    leads = leads or sorted(cv_df["h"].unique())
    tail = (1.0 - level) / 2.0
    rng = np.random.default_rng(seed)
    rows = []
    for h in leads:
        cutoffs, errors = _error_matrix(cv_df, models, h)
        if not cutoffs:
            continue
        index = circular_block_index(len(cutoffs), block, draws, rng)
        for j, model in enumerate(models):
            draws_mae = errors[index, j].mean(axis=1)
            lo, hi = np.quantile(draws_mae, [tail, 1.0 - tail])
            rows.append(
                {
                    "model": model,
                    "h": int(h),
                    "mae": float(errors[:, j].mean()),
                    "lo": float(lo),
                    "hi": float(hi),
                    "n_cutoffs": len(cutoffs),
                }
            )
    return pd.DataFrame(rows, columns=list(MAE_COLUMNS))


def paired_improvements(
    cv_df: pd.DataFrame,
    baseline: str,
    models: list[str] | None = None,
    leads: list[int] | None = None,
    block: int = BLOCK_MONTHS,
    draws: int = DRAWS,
    level: float = LEVEL,
    seed: int = 0,
) -> pd.DataFrame:
    """Per model and lead: `baseline MAE - model MAE`, with a paired bootstrap interval.

    A positive improvement means the model beats the baseline. Both models are scored on the
    same resampled cutoffs, so the interval measures the difference and not the 2 levels.
    An interval that contains 0 does not support a claim of improvement.
    """
    models = [m for m in (models or sorted(cv_df["model"].unique())) if m != baseline]
    leads = leads or sorted(cv_df["h"].unique())
    tail = (1.0 - level) / 2.0
    rng = np.random.default_rng(seed)
    rows = []
    for h in leads:
        for model in models:
            cutoffs, errors = _error_matrix(cv_df, [baseline, model], h)
            if not cutoffs:
                continue
            index = circular_block_index(len(cutoffs), block, draws, rng)
            gap = errors[:, 0] - errors[:, 1]
            draws_gap = gap[index].mean(axis=1)
            lo, hi = np.quantile(draws_gap, [tail, 1.0 - tail])
            rows.append(
                {
                    "model": model,
                    "baseline": baseline,
                    "h": int(h),
                    "improvement": float(gap.mean()),
                    "lo": float(lo),
                    "hi": float(hi),
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "n_cutoffs": len(cutoffs),
                }
            )
    return pd.DataFrame(rows, columns=list(IMPROVEMENT_COLUMNS))


def format_improvements(improvements: pd.DataFrame) -> str:
    """The paired table as text, 1 line per model and lead."""
    lines = []
    for _, row in improvements.iterrows():
        verdict = "excludes 0" if row["excludes_zero"] else "includes 0"
        lines.append(
            f"  h={int(row['h']):2d} {row['model']:<20} "
            f"{row['improvement']:+.3f} ft [{row['lo']:+.3f}, {row['hi']:+.3f}] {verdict}"
        )
    return "\n".join(lines)
