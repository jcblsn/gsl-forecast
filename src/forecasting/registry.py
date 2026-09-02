from src.forecasting.base import Forecaster
from src.forecasting.multivariate.inflow_chain import InflowChainForecaster
from src.forecasting.multivariate.swe_regression import SweRegressionForecaster
from src.forecasting.univariate.drift import DriftForecaster
from src.forecasting.univariate.exponential_smoothing import HoltWintersForecaster
from src.forecasting.univariate.moving_average import MovingAverageForecaster
from src.forecasting.univariate.naive import NaiveForecaster
from src.forecasting.univariate.theta import ThetaForecaster

BASELINE = "naive_last"


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
    "inflow_chain",
}


def production_forecasters() -> list[Forecaster]:
    """Subset written to the forecasts table by gsl-forecast."""
    return [f for f in all_forecasters() if f.name in PRODUCTION_MODELS]
