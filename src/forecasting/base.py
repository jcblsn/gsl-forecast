from abc import ABC, abstractmethod
from typing import Dict, Self

import pandas as pd


class Forecaster(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> Self:
        pass

    @abstractmethod
    def predict(self, h: int) -> pd.DataFrame:
        pass

    def get_metrics(self) -> Dict[str, float]:
        return {}
