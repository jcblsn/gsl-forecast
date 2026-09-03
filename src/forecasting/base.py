from abc import ABC, abstractmethod
from datetime import date
from typing import Self

import pandas as pd


class Forecaster(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> Self:
        pass

    @abstractmethod
    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        """Return a DataFrame with columns [time_col, target, pred, model_name] for h steps
        after start_date (defaults to the last fitted date)."""

    def get_metrics(self) -> dict[str, object]:
        return {}

    def feature_columns(self) -> list[str]:
        """The covariate columns the model reads, for the availability test.

        A model that uses the lake record alone returns an empty list. A model that reads
        `monthly_covariates` must name every column it reads, because
        `tests/test_leakage.py` checks that list against the columns that exist at issue
        time.
        """
        return []
