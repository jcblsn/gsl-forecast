"""The 157 development cutoffs overlap, so a difference needs an interval around it."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.bootstrap import (
    circular_block_index,
    mae_intervals,
    paired_improvements,
)


def cv_frame(n_cutoffs=120, seed=0):
    """Errors that drift over a 4-year cycle, as real cutoffs do, plus a constant gap."""
    rng = np.random.default_rng(seed)
    cutoffs = pd.date_range("2010-01-01", periods=n_cutoffs, freq="MS")
    drift = 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(n_cutoffs) / 48)
    rows = []
    for i, cutoff in enumerate(cutoffs):
        for model, error in (
            ("weak", 1.0 + drift[i]),
            ("strong", 0.2 + drift[i]),
            ("tie", 1.0 + drift[i] + rng.normal(0, 0.01)),
        ):
            rows.append({"model": model, "cutoff": cutoff, "h": 6, "abs_error": abs(error)})
    return pd.DataFrame(rows)


def test_every_position_can_start_a_block():
    rng = np.random.default_rng(0)
    index = circular_block_index(10, 4, 500, rng)
    assert index.shape == (500, 10)
    assert index.min() >= 0 and index.max() <= 9
    assert np.all(np.diff(index[:, :4], axis=1) % 10 == 1)


def test_a_short_series_takes_one_block():
    rng = np.random.default_rng(0)
    assert circular_block_index(3, 24, 5, rng).shape == (5, 3)


def test_the_interval_brackets_the_point_estimate():
    result = mae_intervals(cv_frame(), ["weak", "strong"], [6], draws=500).set_index("model")
    for model in ("weak", "strong"):
        row = result.loc[model]
        assert row["lo"] <= row["mae"] <= row["hi"]
        assert row["n_cutoffs"] == 120


def test_a_real_gap_excludes_zero_and_a_tie_does_not():
    result = paired_improvements(cv_frame(), "weak", ["strong", "tie"], [6], draws=2000).set_index(
        "model"
    )
    assert result.loc["strong", "improvement"] == pytest.approx(0.8, abs=0.05)
    assert result.loc["strong", "excludes_zero"]
    assert abs(result.loc["tie", "improvement"]) < 0.01
    assert not result.loc["tie", "excludes_zero"]


def test_a_model_absent_from_the_frame_gives_no_row():
    result = paired_improvements(cv_frame(), "weak", ["strong", "missing"], [6], draws=200)
    assert list(result["model"]) == ["strong"]


def test_blocks_widen_the_interval_when_errors_are_serially_correlated():
    """A single-cutoff resample would call a dependent series more precise than it is."""
    frame = cv_frame()
    wide = mae_intervals(frame, ["weak"], [6], block=24, draws=2000).iloc[0]
    narrow = mae_intervals(frame, ["weak"], [6], block=1, draws=2000).iloc[0]
    assert wide["hi"] - wide["lo"] > narrow["hi"] - narrow["lo"]
