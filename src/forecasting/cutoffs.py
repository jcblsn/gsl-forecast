"""The rule that selects the cutoff months for a walk-forward pass.

The harness uses this rule. The blend also uses it, because the blend runs an inner pass in
its own training data. The rule is in this module so that a model does not import the
harness.
"""

import random

import pandas as pd
from dateutil.relativedelta import relativedelta

# The water-year stage of an issue. The blend fits one weight curve per stage, and the
# interval layer scales its band per stage, because the errors are strongly heteroskedastic
# by stage.
SEASON_MONTHS = {
    "accumulation": {11, 12, 1, 2, 3},
    "melt": {4, 5, 6},
    "recession": {7, 8, 9, 10},
}


def issue_season(cutoff: pd.Timestamp) -> str:
    """The water-year stage of the issue that follows a cutoff."""
    issue_month = pd.Timestamp(cutoff).month % 12 + 1
    return next(name for name, months in SEASON_MONTHS.items() if issue_month in months)


def valid_cutoffs(data: pd.DataFrame, history_years: int, horizon: int) -> list[pd.Timestamp]:
    """Every month in the last `history_years` that has `horizon` months of actuals after it."""
    latest_month = data["month"].max()
    earliest_cutoff = latest_month - relativedelta(years=history_years)
    latest_cutoff = latest_month - relativedelta(months=horizon)
    mask = (data["month"] >= earliest_cutoff) & (data["month"] <= latest_cutoff)
    return list(data.loc[mask, "month"])


def sample_cutoffs(
    data: pd.DataFrame,
    n: int | None,
    history_years: int,
    horizon: int,
    seed: int = 42,
) -> list[pd.Timestamp]:
    """All valid cutoffs when n is None, otherwise a seeded random sample of n."""
    valid = valid_cutoffs(data, history_years, horizon)
    if n is None:
        return valid
    if len(valid) < n:
        raise ValueError(f"Only {len(valid)} valid cutoffs available, requested {n}")
    return sorted(random.Random(seed).sample(valid, n))


def policy_cutoffs(
    data: pd.DataFrame,
    cutoff_start: str,
    cutoff_end: str,
    horizon: int,
    n: int | None = None,
    seed: int = 42,
) -> list[pd.Timestamp]:
    """The fixed monthly cohort declared by an evaluation policy."""
    start = pd.Timestamp(cutoff_start)
    end = pd.Timestamp(cutoff_end)
    if start > end:
        raise ValueError(f"Evaluation cutoff start {start.date()} is after {end.date()}")
    if horizon < 1:
        raise ValueError("Evaluation horizon must be positive")

    expected_cutoffs = pd.date_range(start, end, freq="MS")
    required = pd.date_range(start, end + pd.DateOffset(months=horizon), freq="MS")
    available = pd.DatetimeIndex(pd.to_datetime(data["month"]).unique())
    missing = required.difference(available)
    if len(missing):
        shown = ", ".join(str(value.date()) for value in missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise ValueError(f"Evaluation policy months are absent from the data: {shown}{suffix}")

    cutoffs = list(expected_cutoffs)
    if n is None:
        return cutoffs
    if len(cutoffs) < n:
        raise ValueError(f"Only {len(cutoffs)} policy cutoffs available, requested {n}")
    return sorted(random.Random(seed).sample(cutoffs, n))
