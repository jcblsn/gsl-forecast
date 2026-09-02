# GSL Forecast

Forecasting the Great Salt Lake's water surface elevation using historical USGS gauge data.

## Overview

The pipeline fetches daily elevation readings from two USGS gauges, aggregates them to monthly averages, runs a suite of time-series forecasters, and tracks experiments with [experiment-tracker](https://github.com/jcblsn/experiment-tracker). Results can be visualized and compared across models and forecast horizons.

Data source: USGS National Water Information System. Storage: DuckDB.

Best univariate model (walk-forward CV, every month-end cutoff in the last 15 years, training from 1960): Holt-Winters with damped additive trend and additive 12-month seasonality (`ets_damped_s12`). See [Current results](#current-results).

## Setup

```bash
uv sync
uv run --frozen pytest
```

Modelling choices live in `config/config.json` under `forecasting`: `train_start`, `horizon`, `experiment_db`, `output_dir`, and the CV cutoff policy. CLI flags override config; anything not passed falls back to config.

## CLI Commands

### Run the ELT pipeline

Fetches USGS data (continuous 15-min + daily) and populates the local DuckDB. Incremental — checks the max date in the DB and only fetches new records.

```bash
uv run gsl-pipeline
```

The current calendar month is excluded from `monthly_elevation` so a partial month is never treated as a full-month average.

### Run forecasts

Fits the production subset of models (see `src/forecasting/registry.py`) on history from `train_start` and writes forward predictions to the `forecasts` table, tagged with `run_id`, `experiment_id`, and `data_max` so every prediction is traceable to a run and a data vintage.

```bash
uv run gsl-forecast [--horizon 12] [--train-start 1960-01-01] [--experiment-db forecast_experiments.db]
```

### Walk-forward cross-validation

Uses every month-end cutoff in the last `history_years` (about 170) by default, fits every registered model at each cutoff, evaluates at h=1..12, and logs per-horizon MAE, RMSE, and MAE relative to `naive_last` to the experiment tracker. Per-cutoff results are saved as parquet under `outputs/` so errors can be sliced by season of cutoff.

```bash
uv run gsl-cv [--n-cutoffs 20] [--horizon 12] [--history-years 15] [--train-start 1960-01-01] [--output-dir outputs] [--no-plots]
```

Pass `--n-cutoffs N` for a seeded random sample instead of all cutoffs.

### Plot forecasts

Generates a plotnine chart of historical elevation + all model forecasts.

```bash
uv run gsl-plot [--history-years 10] [--output outputs/gsl_forecast.png]
```

## Current results

Walk-forward CV, 169 month-end cutoffs (September 2011 to August 2025), 12-month horizon, training from 1960, data through August 2026. MAE in feet; ratio is MAE divided by `naive_last` MAE at the same horizon.

| Horizon | ets_damped_s12 | naive_last | Ratio |
|---|---|---|---|
| 1 | 0.14 | 0.34 | 0.42 |
| 3 | 0.44 | 0.90 | 0.49 |
| 6 | 0.81 | 1.32 | 0.62 |
| 9 | 1.06 | 1.24 | 0.86 |
| 12 | 1.21 | 1.28 | 0.95 |

`ets_damped_s12` is best at every horizon, but by 12 months it is within 5% of repeating the last value. Training from 1980 instead of 1960 changes nothing material.

Six-month MAE by month of cutoff shows where the univariate ceiling is:

| Cutoff month | ets_damped_s12 | naive_last |
|---|---|---|
| Dec | 1.24 | 1.29 |
| Jan | 1.22 | 1.04 |
| Feb | 1.13 | 1.19 |
| Apr | 0.78 | 1.92 |
| May | 0.60 | 2.02 |
| Aug | 0.34 | 0.30 |

From a winter cutoff the model has to guess the size of the coming spring rise and does no better than naive. By April the rise is underway and seasonality carries it. Snowpack and streamflow data are what resolve the winter case, which is the motivation for the multivariate work in `autoresearch-memo.md`.

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

CV runs log `mae_h1`…`mae_h12`, `rmse_h1`…`rmse_h12`, and `mae_ratio_h1`…`mae_ratio_h12` (MAE divided by `naive_last` MAE at the same horizon) per model, so any horizon is directly queryable. `uv run gsl-results <experiment_id>` prints a ranked table.

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
| `theta` | Theta method: SES plus half the linear trend slope |

All models implement the `Forecaster` ABC (`src/forecasting/base.py`) with `fit(df)`, `predict(h)`, and `get_metrics()`. The single list of models is `all_forecasters()` in `src/forecasting/registry.py`; `production_forecasters()` is the subset written by `gsl-forecast`.

## Project Structure

```
src/
  pipeline/
    elt.py              # Extract (USGS HTTP), load (DuckDB), transform (monthly agg)
  forecasting/
    base.py             # Forecaster ABC
    registry.py         # The one list of models (all / production subset)
    run_forecast.py     # Fit production models, store predictions with run identity
    cross_validate.py   # Walk-forward CV, per-cutoff parquet, per-horizon metrics
    plots.py            # CV plots
    plot_forecasts.py   # Plotnine visualization of actuals + forecasts
    view_results.py     # Print experiment metrics via experiment tracker
    univariate/
      naive.py
      moving_average.py
      drift.py
      exponential_smoothing.py
      theta.py
config/
  config.json           # USGS URLs, site IDs, DB path, modelling defaults
tests/
  test_elt.py           # ELT tests with in-memory DuckDB
  test_forecasting.py   # Forecaster correctness tests
  test_cross_validate.py
  test_end_to_end.py    # CV and forecast runs against a temp DB; CLI --help checks
outputs/                # gitignored: CV parquet and PNGs
```

## Data

Two USGS sources are fetched and stored in DuckDB (`./data/gsl.db`):

| Table | Source | Granularity |
|-------|--------|-------------|
| `usgs_water_surface_elevation_continuous` | Site 10010100 | 15-min intervals |
| `usgs_water_surface_elevation_daily` | Site 10010000 | Daily (1847–present) |
| `monthly_elevation` | Derived | Monthly avg/min/max/count, complete months only |
| `forecasts` | Model output | Monthly predictions with run_id, experiment_id, data_max |

Before 1960 the daily table holds about one reading per month, and before 1980 about two, so `monthly_elevation` rows from that era are single readings rather than averages. That is the main reason training from 1960 onward beats the full 1847-present series, and why `train_start` is a config setting rather than a flag to remember.

## Tests and lint

```bash
uv run --frozen pytest
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
```

CI runs the same three commands on every push.
