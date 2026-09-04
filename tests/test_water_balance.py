"""The storage balance, its observation operator, and the rules that keep it honest.

Every other model here maps elevation to elevation. This one conserves volume, so the tests
protect the things conservation depends on: the state is storage, the area and the salinity
come from the month before, and the published monthly mean is derived from the end-of-month
state rather than confused with it.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.forecasting import hypsometry
from src.forecasting.multivariate.water_balance import (
    ACRES_PER_KM2,
    KAF_GL_TO_MT,
    WaterBalanceForecaster,
    extraterrestrial_radiation,
    hargreaves_depth_ft,
    salinity_factor,
)


def frame(n: int = 120, start: str = "2000-01-01") -> pd.DataFrame:
    """A synthetic lake that rises in spring and falls in summer, with matching weather."""
    months = pd.date_range(start, periods=n, freq="MS")
    rng = np.random.default_rng(7)
    season = np.sin(2 * np.pi * (months.month - 4) / 12)
    elevation = 4194.0 + np.cumsum(0.05 * season + rng.normal(0, 0.02, n))
    return pd.DataFrame(
        {
            "month": months,
            "avg_elevation": elevation,
            "elevation_eom_ft": elevation + 0.01,
            "inflow_kaf_total": np.clip(120 + 90 * season + rng.normal(0, 12, n), 5, None),
            "tmax_f_kslc": 62 + 28 * np.sin(2 * np.pi * (months.month - 4) / 12),
            "tmin_f_kslc": 40 + 22 * np.sin(2 * np.pi * (months.month - 4) / 12),
            "prcp_in_kslc": np.clip(1.3 - 0.7 * season + rng.normal(0, 0.2, n), 0, None),
            "salt_mass_mt": np.linspace(1200, 900, n),
            "swe_eom_gsl": np.clip(8 * np.sin(2 * np.pi * (months.month - 1) / 12), 0, None),
            "prec_wy_eom_gsl": np.tile(np.arange(1, 13) * 1.4, n // 12 + 1)[:n],
        }
    )


def test_the_balance_conserves_volume_over_a_closed_step():
    """The step is an accounting identity. If storage in does not equal storage out plus the
    fluxes, the model is not a balance whatever it is called."""
    model = WaterBalanceForecaster().fit(frame())
    volume = hypsometry.volume_kaf(4194.0)
    stepped = model._step(volume, 1000.0, month=7, days=31, inflow=100.0)
    area = hypsometry.area_km2(4194.0) * ACRES_PER_KM2
    weather = model._climatology.loc[7]
    evaporation = hargreaves_depth_ft(
        np.array([weather["tmax_f_kslc"]]),
        np.array([weather["tmin_f_kslc"]]),
        np.array([7]),
        np.array([31]),
    )[0]
    salinity = 1000.0 / volume / KAF_GL_TO_MT
    factor = salinity_factor(np.array([salinity]), model._coefficients["salt_coefficient"])[0]
    expected = volume + (
        model._coefficients["intercept_kaf"]
        + model._beta[model.INFLOW] * 100.0
        + model._beta[model.PRECIP] * (weather["prcp_in_kslc"] / 12.0) * area / 1000.0
        - model._beta[model.EVAP] * evaporation * factor * area / 1000.0
        + model._coefficients["seasonal_ft"][6] * area / 1000.0
        + model._residual[7] * area / 1000.0
    )
    assert stepped == pytest.approx(expected, rel=1e-9)


def test_the_published_month_is_the_mean_of_the_two_bracketing_states():
    """The target is a monthly mean and the state is an instant at month end. Reporting the
    end-of-month state as if it were the mean puts the forecast half a month out of phase."""
    data = frame()
    model = WaterBalanceForecaster(salinity=False).fit(data)
    predictions = model.predict(3)
    start = hypsometry.volume_kaf(float(data["elevation_eom_ft"].iloc[-1]))
    first = model._step(
        start,
        0.0,
        month=predictions["month"].iloc[0].month,
        days=predictions["month"].iloc[0].days_in_month,
        inflow=model._inflow_model.inflow_forecast(1),
    )
    assert predictions["pred"].iloc[0] == pytest.approx(
        float(hypsometry.elevation_ft((start + first) / 2.0)), rel=1e-9
    )


def test_salinity_comes_from_the_carried_state_and_not_from_the_input():
    """Salinity is salt mass over volume and elevation is a function of volume, so a
    contemporaneous salinity partly is the answer. The model must read its own state."""
    data = frame()
    model = WaterBalanceForecaster().fit(data)
    before = model.predict(6)["pred"].to_numpy()
    tampered = data.copy()
    tampered.loc[tampered.index[-1], "salt_mass_mt"] = 5000.0
    # Refitting on tampered history would change the fit; only the carried state is at issue
    # here, so the check is that predict reads salt mass from the last fitted row.
    model._data.loc[model._data.index[-1], "salt_mass_mt"] = 5000.0
    after = model.predict(6)["pred"].to_numpy()
    assert not np.allclose(before, after), "salinity must enter the step"
    assert (after < before).all(), "more salt suppresses evaporation and holds the lake up"


def test_the_area_that_scales_the_fluxes_is_the_area_before_the_step():
    """Evaporation over a month happens on the lake that was there, not on the lake the step
    produces. Using this month's area would use the answer to compute the answer."""
    model = WaterBalanceForecaster().fit(frame())
    rows = model._frame(frame())
    assert rows["area_prev"].iloc[3] == pytest.approx(rows["area_acres"].iloc[2])
    assert np.isnan(rows["area_prev"].iloc[0])


def test_a_gap_in_the_record_is_not_treated_as_a_monthly_step():
    """A step between March and September is not one month of fluxes. Fitting it as one
    would put 6 months of evaporation into a single monthly coefficient."""
    data = frame(n=60)
    broken = pd.concat([data.iloc[:20], data.iloc[26:]]).reset_index(drop=True)
    model = WaterBalanceForecaster(salinity=False)
    rows = model._frame(broken)
    assert not bool(rows["consecutive"].iloc[20])
    assert bool(rows["consecutive"].iloc[21])


def test_the_hypsometry_domain_guard_refuses_a_clamp():
    """`np.interp` returns the end of the table for an input past it and says nothing. A
    storage model that drifts out of the table would step on a constant area."""
    low, high = hypsometry.elevation_domain()
    with pytest.raises(ValueError, match="outside the hypsometry table"):
        hypsometry.area_km2(high + 5.0)
    with pytest.raises(ValueError, match="outside the hypsometry table"):
        hypsometry.volume_kaf(low - 5.0)
    assert hypsometry.area_km2(high + 5.0, strict=False) == pytest.approx(hypsometry.area_km2(high))


def test_the_forecast_covers_the_requested_months_exactly():
    """The harness matches predictions to actuals by target month and raises otherwise."""
    data = frame()
    model = WaterBalanceForecaster().fit(data)
    predictions = model.predict(24)
    expected = pd.date_range(
        data["month"].iloc[-1] + pd.DateOffset(months=1), periods=24, freq="MS"
    )
    assert pd.DatetimeIndex(predictions["month"]).equals(expected)
    assert np.isfinite(predictions["pred"]).all()
    assert predictions["model_name"].unique().tolist() == ["water_balance"]


def test_the_model_names_every_covariate_it_reads():
    """The leakage test and the blend cache both key on this list."""
    model = WaterBalanceForecaster()
    named = set(model.feature_columns())
    assert {"elevation_eom_ft", "inflow_kaf_total", "salt_mass_mt"} <= named
    assert {"tmax_f_kslc", "tmin_f_kslc", "prcp_in_kslc"} <= named
    assert "salt_mass_mt" not in WaterBalanceForecaster(salinity=False).feature_columns()


def test_extraterrestrial_radiation_peaks_in_summer_at_this_latitude():
    """The seasonal shape of available energy is astronomy. If it were wrong the fitted
    temperature coefficient would silently absorb the error."""
    monthly = extraterrestrial_radiation(np.arange(1, 13))
    assert monthly.argmax() == 5
    assert monthly.argmin() == 11
    assert monthly.max() > 2 * monthly.min()


def test_salinity_suppression_is_bounded_below():
    """A fitted coefficient must not be able to drive evaporation negative, which would make
    the lake gain water by being salty."""
    assert salinity_factor(np.array([0.0]), 2.0) == pytest.approx(1.0)
    assert salinity_factor(np.array([130.0]), 2.0) == pytest.approx(0.74)
    assert salinity_factor(np.array([900.0]), 5.0)[0] >= 0.05


def test_too_few_closed_steps_is_refused_rather_than_fitted():
    """A balance fitted on a handful of steps would report a closure it has not measured."""
    with pytest.raises(ValueError, match="closed monthly steps"):
        WaterBalanceForecaster(min_obs=200).fit(frame(n=60))


def test_predict_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="must be fitted"):
        WaterBalanceForecaster().predict(3, start_date=date(2020, 1, 1))
