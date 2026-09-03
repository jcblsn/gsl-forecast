from datetime import date
from typing import Self

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ..base import Forecaster
from ..multivariate.regression import TARGET_COL, TIME_COL, require_columns

ENDPOINT_CANDIDATES = (
    "endpoint_7d_median",
    "endpoint_3d_median",
    "last_elevation",
    TARGET_COL,
)


class EndpointSeasonalForecaster(Forecaster):
    """Latest endpoint plus the historical median change for the issue month and lead."""

    def __init__(
        self,
        candidates: tuple[str, ...] = ENDPOINT_CANDIDATES,
        min_obs: int = 5,
        min_selection_errors: int = 12,
        name: str = "endpoint_seasonal",
    ):
        super().__init__(name=name)
        self.candidates = candidates
        self.min_obs = min_obs
        self.min_selection_errors = min_selection_errors
        self._data: pd.DataFrame | None = None
        self.anchor_column = TARGET_COL
        self.anchor_selection_mae: float | None = None
        self.anchor_selection_n = 0

    def _candidate_columns(self, data: pd.DataFrame) -> list[str]:
        last = data.iloc[-1]
        return [column for column in self.candidates if column in data and pd.notna(last[column])]

    def _seasonal_deltas(
        self, column: str, h: int, origin_month: int, stop: int | None = None
    ) -> np.ndarray:
        frame = self._data.iloc[:stop] if stop is not None else self._data
        anchors = frame[column].to_numpy(dtype=float)
        targets = frame[TARGET_COL].to_numpy(dtype=float)
        months = frame[TIME_COL]
        values = []
        for i in range(len(frame) - h):
            if months.iloc[i].month != origin_month:
                continue
            if months.iloc[i + h] != months.iloc[i] + relativedelta(months=h):
                continue
            if np.isfinite(anchors[i]) and np.isfinite(targets[i + h]):
                values.append(targets[i + h] - anchors[i])
        return np.asarray(values, dtype=float)

    def _expanding_errors(self, column: str) -> dict[int, float]:
        errors = {}
        anchors = self._data[column].to_numpy(dtype=float)
        targets = self._data[TARGET_COL].to_numpy(dtype=float)
        months = self._data[TIME_COL]
        for i in range(len(self._data) - 1):
            if not np.isfinite(anchors[i]) or not np.isfinite(targets[i + 1]):
                continue
            if months.iloc[i + 1] != months.iloc[i] + relativedelta(months=1):
                continue
            changes = self._seasonal_deltas(column, 1, months.iloc[i].month, stop=i)
            if len(changes) >= self.min_obs:
                prediction = anchors[i] + float(np.median(changes))
                errors[i] = abs(prediction - targets[i + 1])
        return errors

    def _select_anchor(self) -> None:
        candidates = self._candidate_columns(self._data)
        errors = {column: self._expanding_errors(column) for column in candidates}
        common = set.intersection(*(set(values) for values in errors.values())) if errors else set()
        if len(common) < self.min_selection_errors:
            self.anchor_column = candidates[0]
            return
        indices = sorted(common)
        scores = {
            column: float(np.mean([values[i] for i in indices]))
            for column, values in errors.items()
        }
        self.anchor_column = min(candidates, key=scores.__getitem__)
        self.anchor_selection_mae = scores[self.anchor_column]
        self.anchor_selection_n = len(indices)

    def fit(self, data: pd.DataFrame) -> Self:
        require_columns(data, [TIME_COL, TARGET_COL])
        self._data = data.sort_values(TIME_COL).reset_index(drop=True)
        if len(self._data) < 2:
            raise ValueError("endpoint_seasonal needs at least 2 monthly observations")
        self.last_date = pd.Timestamp(self._data[TIME_COL].iloc[-1])
        self._select_anchor()
        self.is_fitted = True
        return self

    def _prediction(self, h: int) -> float:
        anchor = float(self._data[self.anchor_column].iloc[-1])
        changes = self._seasonal_deltas(self.anchor_column, h, self.last_date.month)
        if len(changes) < self.min_obs and self.anchor_column != TARGET_COL:
            anchor = float(self._data[TARGET_COL].iloc[-1])
            changes = self._seasonal_deltas(TARGET_COL, h, self.last_date.month)
        if not len(changes):
            return anchor
        return anchor + float(np.median(changes))

    def predict(self, h: int, start_date: date | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        if h < 1:
            raise ValueError("Forecast horizon must be positive")
        origin = pd.Timestamp(start_date or self.last_date)
        return pd.DataFrame(
            {
                TIME_COL: [origin + relativedelta(months=lead) for lead in range(1, h + 1)],
                "target": TARGET_COL,
                "pred": [self._prediction(lead) for lead in range(1, h + 1)],
                "model_name": self.name,
            }
        )

    def get_metrics(self) -> dict[str, object]:
        return {
            "anchor_column": self.anchor_column,
            "anchor_selection_mae": self.anchor_selection_mae,
            "anchor_selection_n": self.anchor_selection_n,
            "min_obs": self.min_obs,
        }
