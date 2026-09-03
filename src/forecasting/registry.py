from src.forecasting.base import Forecaster
from src.forecasting.multivariate.blend import BlendForecaster
from src.forecasting.multivariate.inflow_chain import InflowChainForecaster
from src.forecasting.multivariate.state_space import StateSpaceForecaster
from src.forecasting.multivariate.swe_regression import SweRegressionForecaster
from src.forecasting.univariate.drift import DriftForecaster
from src.forecasting.univariate.exponential_smoothing import HoltWintersForecaster
from src.forecasting.univariate.moving_average import MovingAverageForecaster
from src.forecasting.univariate.naive import NaiveForecaster
from src.forecasting.univariate.theta import ThetaForecaster

BASELINE = "naive_last"

SWE_HEAD_FEATURES = ["swe_eom_gsl", "prec_wy_eom_gsl", "head_diff_ft"]


def _anchor() -> tuple[str, object]:
    """The last component of every blend: the best model that uses the lake record alone."""
    return (
        "ets_damped_s12",
        lambda: HoltWintersForecaster(
            trend="add", seasonal="add", seasonal_periods=12, damped_trend=True
        ),
    )


def _swe_head() -> tuple[str, object]:
    return (
        "swe_head",
        lambda: SweRegressionForecaster(features=SWE_HEAD_FEATURES, name="swe_head"),
    )


def three_component_blends() -> list[Forecaster]:
    """One blend for each candidate second covariate component.

    `blend` is strong to lead 12 and weak past lead 18, because `swe_head` is. Each model
    below adds a component that is strong where `swe_head` is weak, and the fitted weight
    decides where to use it. The experimental state-space baseline is not a blend candidate.
    """
    second = {
        "blend3_swe": (
            "swe_regression",
            lambda: SweRegressionForecaster(name="swe_regression"),
        ),
        "blend3_chain": ("inflow_chain", InflowChainForecaster),
    }
    return [
        BlendForecaster(components=[_swe_head(), pair, _anchor()], name=name)
        for name, pair in second.items()
    ]


def all_forecasters() -> list[Forecaster]:
    """Every model benchmarked in cross-validation. Fresh instances on each call."""
    return [
        NaiveForecaster(method="last"),
        NaiveForecaster(method="seasonal", seasonal_period=12),
        MovingAverageForecaster(window=3),
        MovingAverageForecaster(window=6),
        MovingAverageForecaster(window=12),
        DriftForecaster(window=12),
        DriftForecaster(window=24),
        DriftForecaster(window=60),
        HoltWintersForecaster(trend="add", seasonal=None, damped_trend=False),
        HoltWintersForecaster(trend="add", seasonal=None, damped_trend=True),
        HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12, damped_trend=False),
        HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12, damped_trend=True),
        ThetaForecaster(),
        SweRegressionForecaster(),
        SweRegressionForecaster(
            features=["swe_eom_gsl", "prec_wy_eom_gsl", "head_diff_ft"], name="swe_head"
        ),
        InflowChainForecaster(),
        InflowChainForecaster(level_term="area", name="inflow_chain_area"),
        StateSpaceForecaster(),
        BlendForecaster(),
        BlendForecaster(
            snow_features=["swe_eom_gsl", "prec_wy_eom_gsl"],
            snow_name="swe_regression",
            name="blend_swe",
        ),
        *three_component_blends(),
    ]


PRODUCTION_MODELS = {
    "naive_last",
    "naive_seasonal",
    "drift_24m",
    "ets_damped_s12",
    "ets_add_s12",
    "ets_damped_noseas",
    "theta",
    "swe_regression",
    "swe_head",
    "inflow_chain",
    "blend",
}

EXPERIMENTAL_MODELS = {"state_space"}


def production_forecasters() -> list[Forecaster]:
    """Subset written to the forecasts table by gsl-forecast."""
    return [f for f in all_forecasters() if f.name in PRODUCTION_MODELS]
