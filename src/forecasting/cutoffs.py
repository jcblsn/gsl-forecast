"""The rule that selects the cutoff months for a walk-forward pass.

The harness uses this rule. The blend also uses it, because the blend runs an inner pass in
its own training data. The rule is in this module so that a model does not import the
harness.
"""

import random

import pandas as pd
from dateutil.relativedelta import relativedelta


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
