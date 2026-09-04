"""The UGS brine reader and the salt-mass trajectory it produces.

Salt mass is the one part of the brine record this project depends on. It is not conserved:
mineral extraction removes salt and the causeway exports it. A reader that held it constant
would put the recent lake at the wrong salinity, and the evaporation term with it.
"""

from datetime import datetime
from io import BytesIO

import numpy as np
import openpyxl
import pandas as pd
import pytest

from src.pipeline import brine


def workbook(rows) -> bytes:
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "AS2"
    sheet.append(
        ["SITE", "DATE", "DEPTH-FT", "Salinity EOS (g/L)", "LAB-DEN\n(g/cm3)", "LK-ELEV (feet)"]
    )
    for row in rows:
        sheet.append(row)
    buffer = BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def test_the_reader_recovers_a_campaign_and_its_depths():
    """Each campaign samples a depth profile. The reader must keep the depths, because the
    upper brine and the deep brine are different water."""
    content = workbook(
        [
            ["AS2", datetime(2020, 1, 15), 3, 120.0, 1.09, 4193.0],
            ["AS2", datetime(2020, 1, 15), 25, 180.0, 1.14, 4193.0],
        ]
    )
    samples = brine.read_workbook(content, ["AS2"])
    assert len(samples) == 2
    assert sorted(samples["depth_ft"]) == [3.0, 25.0]
    assert samples["site"].unique().tolist() == ["AS2"]


def test_the_deep_brine_layer_is_left_out_of_the_upper_brine_salinity():
    """The lake evaporates at the salinity of the water on top. Averaging the deep brine in
    would raise the salinity, and would suppress the evaporation term too far."""
    content = workbook(
        [
            ["AS2", datetime(2020, 1, 15), 3, 120.0, 1.09, 4193.0],
            ["AS2", datetime(2020, 1, 15), 10, 130.0, 1.10, 4193.0],
            ["AS2", datetime(2020, 1, 15), 25, 300.0, 1.20, 4193.0],
        ]
    )
    campaigns = brine.upper_brine_campaigns(brine.read_workbook(content, ["AS2"]), "AS2")
    assert len(campaigns) == 1
    assert campaigns["salinity_gl"].iloc[0] == pytest.approx(125.0)


def test_salt_mass_falls_as_the_record_runs():
    """Extraction and causeway export remove salt. A model that holds the mass constant gets
    the salinity of the modern lake wrong, and its evaporation with it."""
    content = workbook(
        [
            ["AS2", datetime(1993, 6, 1), 3, 142.0, 1.11, 4200.0],
            ["AS2", datetime(2025, 6, 1), 3, 110.0, 1.08, 4193.5],
        ]
    )
    campaigns = brine.upper_brine_campaigns(brine.read_workbook(content, ["AS2"]), "AS2")
    gauge = pd.DataFrame(
        {
            "d": pd.to_datetime(["1993-06-01", "2025-06-01"]),
            "elevation": [4200.2, 4193.5],
        }
    )
    salt = brine.salt_mass_series(campaigns, gauge)
    assert salt["salt_mass_mt"].iloc[0] > salt["salt_mass_mt"].iloc[1]
    assert salt["salt_mass_mt"].iloc[0] == pytest.approx(1350, rel=0.25)


def test_a_sparse_year_interpolates_without_inventing_a_seasonal_cycle():
    """UGS samples about twice a year. A monthly salinity read straight off that sampling
    would carry a seasonal cycle the measurements cannot support."""
    salt = pd.DataFrame(
        {
            "sample_date": pd.to_datetime(["2020-01-15", "2021-01-15"]),
            "salt_mass_mt": [1200.0, 1000.0],
        }
    )
    months = pd.date_range("2020-01-01", "2021-01-01", freq="MS")
    monthly = brine.monthly_salt_mass(salt, months)
    # A straight line in time, so the change per day is constant. The change per month is
    # not, because months have different lengths, and that is the point: the shape comes
    # from the calendar and not from a cycle the sampling claims to have seen.
    per_day = monthly["salt_mass_mt"].diff() / monthly["month"].diff().dt.days
    assert per_day.dropna().std() == pytest.approx(0.0, abs=1e-9)
    # The first month falls before the first campaign, so it stays null rather than
    # extrapolating backwards into a record nobody measured.
    assert pd.isna(monthly["salt_mass_mt"].iloc[0])
    assert monthly["salt_mass_mt"].dropna().is_monotonic_decreasing


def test_a_value_after_the_last_campaign_is_carried_forward_and_declares_its_age():
    """The forecast runs every month and a campaign can be 6 months old at issue. A carried
    value is an assumption, so its age must travel with it."""
    salt = pd.DataFrame({"sample_date": pd.to_datetime(["2020-01-15"]), "salt_mass_mt": [1000.0]})
    months = pd.date_range("2019-11-01", "2020-06-01", freq="MS")
    monthly = brine.monthly_salt_mass(salt, months).set_index("month")
    assert np.isnan(monthly.loc["2019-11-01", "salt_mass_mt"])
    assert np.isnan(monthly.loc["2019-11-01", "salt_mass_age_days"])
    assert monthly.loc["2020-06-01", "salt_mass_mt"] == pytest.approx(1000.0)
    assert monthly.loc["2020-06-01", "salt_mass_age_days"] == pytest.approx(138.0)


def test_no_campaigns_gives_null_columns_rather_than_an_error():
    """A source outage must leave the columns null, so `data_status` can refuse the issue,
    rather than stopping the whole pipeline."""
    months = pd.date_range("2020-01-01", "2020-03-01", freq="MS")
    monthly = brine.monthly_salt_mass(pd.DataFrame(), months)
    assert monthly["salt_mass_mt"].isna().all()
    assert len(monthly) == 3


def test_the_density_to_salinity_conversion_is_fitted_and_not_assumed():
    """The independent record reports density. Converting it with a borrowed constant would
    make the cross-check a test of that constant. The UGS record carries both, so the
    relation is fitted on the pairs it already holds."""
    from src.forecasting.validate_salinity import density_to_salinity

    rng = np.random.default_rng(3)
    density = rng.uniform(1.05, 1.20, 200)
    samples = pd.DataFrame(
        {
            "density_g_cm3": density,
            "salinity_gl": 1245.0 * density - 1237.0 + rng.normal(0, 0.5, 200),
        }
    )
    slope, intercept, rmse = density_to_salinity(samples)
    assert slope == pytest.approx(1245.0, rel=0.02)
    assert intercept == pytest.approx(-1237.0, rel=0.02)
    assert rmse < 2.0


def test_too_few_paired_samples_refuse_the_conversion():
    """A conversion fitted on a handful of pairs would carry the cross-check on its own."""
    from src.forecasting.validate_salinity import density_to_salinity

    samples = pd.DataFrame({"density_g_cm3": [1.1, 1.15], "salinity_gl": [130.0, 190.0]})
    with pytest.raises(ValueError, match="Too few paired"):
        density_to_salinity(samples)
