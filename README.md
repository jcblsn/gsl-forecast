# GSL Forecast

Forecasting the Great Salt Lake's water surface elevation using historical USGS gauge data.

## Overview

The pipeline fetches daily elevation readings from two USGS gauges, aggregates them to monthly averages, runs a suite of time-series forecasters, and tracks experiments with [experiment-tracker](https://github.com/jcblsn/experiment-tracker). Results can be visualized and compared across models and forecast horizons.

**Data source:** USGS National Water Information System
**Storage:** DuckDB
**Best model (walk-forward CV, 20 cutoffs, 1960–present):** Holt-Winters with damped additive trend + additive 12-month seasonality (`ets_damped_s12`)

## Setup

```bash
uv sync
```

## CLI Commands

### Run the ELT pipeline

Fetches USGS data (continuous 15-min + daily) and populates the local DuckDB. Incremental — checks the max date in the DB and only fetches new records.

```bash
uv run gsl-pipeline
# or
uv run python -m src.pipeline.elt
```

### Run forecasts

Fits all models on the full historical series and writes 12-month forward predictions to the `forecasts` table. Logs each run to the experiment tracker.

```bash
uv run gsl-forecast [--horizon 12] [--validation-months 6] [--experiment-db forecast_experiments.db]
```

### Walk-forward cross-validation

Samples random cutoffs from recent history, fits every model at each cutoff, evaluates at each forecast horizon h=1..12, and logs per-horizon MAE/RMSE to the experiment tracker.

```bash
uv run gsl-cv [--n-cutoffs 20] [--horizon 12] [--history-years 15] [--train-start 1960-01-01] [--output gsl_cv.png]
```

Key findings:
- Training on data from 1960 onward outperforms the full 1847–present series — the modern lake regime is more predictive than 19th-century levels
- `ets_damped_s12` wins at all 12 horizons in both configurations

### Plot forecasts

Generates a plotnine chart of historical elevation + all model forecasts.

```bash
uv run gsl-plot [--history-years 10] [--output gsl_forecast.png]
```

## Querying Results

The experiment tracker CLI (`expt`) can query any logged metric:

```bash
# List all experiments
expt --db forecast_experiments.db list

# Show all runs for an experiment (includes per-horizon MAE/RMSE)
expt --db forecast_experiments.db runs <experiment_id>

# Find the best model at a specific horizon
expt --db forecast_experiments.db best <experiment_id> --metric mae_h6 --minimize

# Aggregate metrics across all runs in an experiment
expt --db forecast_experiments.db aggregate <experiment_id> --metric mae_h1
```

CV runs log `mae_h1`…`mae_h12` and `rmse_h1`…`rmse_h12` per model, so any horizon is directly queryable.

## Models

| Model | Description |
|-------|-------------|
| `naive_last` | Repeat last observed value |
| `naive_seasonal` | Repeat same month from prior year |
| `ma_simple_{3,6,12}` | Simple moving average over N months |
| `drift_{12,24,60}m` | Project average slope over last N months |
| `ets_add_s12` | Holt-Winters: additive trend + additive seasonal |
| `ets_damped_s12` | Holt-Winters: damped additive trend + additive seasonal (**best**) |
| `ets_add_noseas` | Holt linear trend, no seasonal component |
| `ets_damped_noseas` | Holt damped trend, no seasonal component |
| `theta` | Theta method: average of SES + linear trend forecasts |

All models implement the `Forecaster` ABC (`src/forecasting/base.py`) with `fit(df)`, `predict(h)`, and `get_metrics()`.

## Project Structure

```
src/
  pipeline/
    elt.py              # Extract (USGS HTTP), load (DuckDB), transform (monthly agg)
  forecasting/
    base.py             # Forecaster ABC
    run_forecast.py     # Fit all models, store predictions, log to experiment tracker
    cross_validate.py   # Walk-forward CV with per-horizon metric logging
    plot_forecasts.py   # Plotnine visualization of actuals + forecasts
    view_results.py     # Print experiment metrics via experiment tracker
    univariate/
      naive.py
      moving_average.py
      drift.py
      exponential_smoothing.py
      theta.py
config/
  config.json           # USGS URLs, site IDs, DB path
tests/
  conftest.py
  test_elt.py           # Integration tests with in-memory DuckDB
  test_forecasting.py   # Forecaster correctness tests
```

## Data

Two USGS sources are fetched and stored in DuckDB (`./data/gsl.db`):

| Table | Source | Granularity |
|-------|--------|-------------|
| `usgs_water_surface_elevation_continuous` | Site 10010100 | 15-min intervals |
| `usgs_water_surface_elevation_daily` | Site 10010000 | Daily (1847–present) |
| `monthly_elevation` | Derived | Monthly avg/min/max |
| `forecasts` | Model output | Monthly predictions |

## Tests

```bash
uv run pytest
```
