# GSL Forecast

A dated, verified, year-round forecast of the Great Salt Lake's water surface elevation.

## Goal

Build the best live, continuously operationalized forecast of Great Salt Lake south-arm elevation (USGS gauge 10010000 at Saltair, feet): a dated, versioned, probabilistic monthly forecast for months 1-24 from the present that combines a univariate baseline with snowpack and streamflow covariates and an explicit water balance, runs every month of the year, and is scored publicly against the gauge as observations arrive.

Two scalars carry most of the decision weight and are the headline targets:

- Spring peak elevation (the April-June maximum of the monthly mean)
- Water-year-end elevation (the September mean, the annual low)

Horizons: 1-6 months is the operational window where snowpack makes the lake predictable; 6-24 months is the gap no one currently fills.

## Why

A survey of existing forecasts is in `operational-forecasts-survey.md`. In short:

- The only routine, dated product that targets lake elevation is the NRCS Utah Snow Survey's advisory rise-to-peak outlook, issued January through May since 2024. Its April-issue peak error was 0.1-0.4 ft in 2024-2026 against a stated band of about plus or minus half a foot. It stops in May, so nothing operational forecasts the autumn low or anything beyond six months.
- CBRFC issues ensemble streamflow forecasts for the tributaries (about 16-18% April error on April-July volume) but no lake product.
- Long-range models (USU Climate Center's climate-oscillation regression, the Strike Team's 30-year Monte Carlo, the state's GSLIM planning model) are scenario tools or multi-year statistical forecasts with roughly 3 ft RMSE at 8 years, and none is verified as a dated forecast.

The benchmark to beat in season is the NRCS outlook; out of season the benchmark is our own univariate model, which from a winter cutoff does no better than repeating the last value (see [Current results](#current-results)).

## Roadmap

- [x] Univariate baselines with walk-forward CV and experiment tracking
- [x] Survey of operational and gray-literature forecasts
- [x] Score the headline scalars (spring peak, water-year-end low) by issue month and place them next to the NRCS record in `data/benchmarks/`
- [x] Ingest covariates: SNOTEL basin snow water equivalent and precipitation, USGS inflow gauges (Bear 10126000, Weber 10141000, Jordan 10170490), issued NRCS/CBRFC inflow forecasts
- [x] Multivariate models: SWE regression (the NRCS method) and a reduced-form inflow-chain water balance
- [x] Probabilistic output (q05-q95) from walk-forward errors, scored with CRPS and 90% coverage in CV
- [x] Monthly GitHub Actions run that commits dated forecasts to `forecasts/`, plus `gsl-verify` for a live skill record
- [x] Autoresearch program for the multivariate models (`program.md`); loop not yet run
- [ ] Bathymetry (USGS elevation-area-volume table) in the water balance
- [ ] Percent-of-median snowpack features and issued inflow forecasts as covariates

## Overview

The pipeline fetches daily elevation readings from two USGS gauges, aggregates them to monthly averages, runs a suite of time-series forecasters, and tracks experiments with [experiment-tracker](https://github.com/jcblsn/experiment-tracker). Results can be visualized and compared across models and forecast horizons.

Data source: USGS National Water Information System. Storage: DuckDB.

Best univariate model (walk-forward CV, every month-end cutoff in the last 15 years, training from 1960): Holt-Winters with damped additive trend and additive 12-month seasonality (`ets_damped_s12`). See [Current results](#current-results).

## Setup

```bash
uv sync
uv run --frozen pytest
```

Modelling choices live in `config/config.json` under `forecasting`: `train_start`, `horizon` (24 months), `experiment_db`, `output_dir`, and the CV cutoff policy. CLI flags override config; anything not passed falls back to config.

## CLI Commands

### Run the ELT pipeline

Fetches USGS elevation data (continuous 15-min + daily), then the covariates (SNOTEL daily SWE and precipitation for every site in the Bear, Weber and Provo-Jordan hydrologic units; USGS daily discharge for the three inflow gauges), and populates the local DuckDB. Incremental — checks the max date per table (per station for SNOTEL) and only fetches new records. The first run pulls about 50 years of daily data and takes a few minutes.

```bash
uv run gsl-pipeline [--skip-covariates]
```

The current calendar month is excluded from `monthly_elevation` so a partial month is never treated as a full-month average.

### Run forecasts

Fits the production subset of models (see `src/forecasting/registry.py`) on history from `train_start` and writes forward predictions to the `forecasts` table, tagged with `run_id`, `experiment_id`, and `data_max` so every prediction is traceable to a run and a data vintage.

```bash
uv run gsl-forecast [--horizon 24] [--train-start 1960-01-01] [--experiment-db forecast_experiments.db] [--export forecasts/2026-09.csv --intervals outputs/cv_results_<stamp>.parquet]
```

`--export` writes a dated forecast file (issue month, target month, lead, model, point forecast, and q05-q95 when `--intervals` names a CV results file to take empirical error quantiles from). The monthly GitHub Actions workflow (`.github/workflows/forecast.yml`) runs pipeline, CV, forecast and verification and commits the file under `forecasts/`.

### Walk-forward cross-validation

Uses every month-end cutoff in the last `history_years` (about 170) by default, fits every registered model at each cutoff, evaluates at h=1..24, and logs per-horizon MAE, RMSE, and MAE relative to `naive_last` to the experiment tracker. Per-cutoff results are saved as parquet under `outputs/` so errors can be sliced by season of cutoff.

```bash
uv run gsl-cv [--n-cutoffs 20] [--horizon 24] [--history-years 15] [--train-start 1960-01-01] [--output-dir outputs] [--no-plots]
```

Pass `--n-cutoffs N` for a seeded random sample instead of all cutoffs, and `--models a,b` to evaluate a subset (the `naive_last` baseline is always included).

Besides per-horizon MAE, CV logs the two headline scalars by issue date (`peak_mae_jan` … `peak_mae_may`, `wyend_mae_jan` … `wyend_mae_may`: the spring peak and September level as forecast from data ending the previous month) and probabilistic scores (`crps_h1` … and `cov90_h1` …) from leave-one-year-out empirical intervals. A `headline_<stamp>.parquet` sits next to the per-cutoff parquet.

### Benchmark against NRCS

```bash
uv run gsl-benchmark [--headline outputs/headline_<stamp>.parquet] [--model ets_damped_s12]
```

Prints the NRCS outlook record from `data/benchmarks/nrcs_outlooks.csv` (issue date, implied peak, actual peak) next to one fixed model's spring-peak forecast from the same issue date. NRCS actuals are daily peaks; ours are peaks of the monthly mean, so the last column rescores NRCS against the monthly-mean actual.

### Verify issued forecasts

```bash
uv run gsl-verify [--forecast-dir forecasts]
```

Joins every dated forecast in `forecasts/` to the observed monthly means and writes MAE, bias and 90% coverage by model and lead to `forecasts/verification.csv`. This is the live skill record.

### Plot forecasts

Generates a plotnine chart of historical elevation + all model forecasts.

```bash
uv run gsl-plot [--history-years 10] [--output outputs/gsl_forecast.png]
```

## Current results

These results predate the move to a 24-month horizon and will be refreshed on the next CV run.

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
| `swe_regression` | For the cutoff's calendar month and each lead, regresses the change in elevation on current level, basin month-end SWE and water-year precipitation across past years (the NRCS outlook generalised to every month and lead) |
| `inflow_chain` | Snowpack predicts each future month's tributary inflow; a fitted monthly bucket step (change as a function of that month's inflow and the starting level) rolls the elevation forward |

All models implement the `Forecaster` ABC (`src/forecasting/base.py`) with `fit(df)`, `predict(h)`, and `get_metrics()`. The single list of models is `all_forecasters()` in `src/forecasting/registry.py`; `production_forecasters()` is the subset written by `gsl-forecast`.

## Project Structure

```
src/
  pipeline/
    elt.py              # Extract (USGS HTTP), load (DuckDB), transform (monthly agg)
    covariates.py       # SNOTEL and discharge ingestion, monthly_covariates
  forecasting/
    base.py             # Forecaster ABC
    registry.py         # The one list of models (all / production subset)
    run_forecast.py     # Fit production models, store predictions with run identity
    cross_validate.py   # Walk-forward CV, per-cutoff parquet, per-horizon metrics
    plots.py            # CV plots
    plot_forecasts.py   # Plotnine visualization of actuals + forecasts
    view_results.py     # Print experiment metrics via experiment tracker
    data.py             # monthly_elevation joined to monthly_covariates
    headline.py         # Spring-peak and water-year-end scoring by issue month
    quantiles.py        # Empirical intervals, pinball/CRPS, coverage
    benchmark.py        # gsl-benchmark: our peaks next to the NRCS record
    verify.py           # gsl-verify: score dated forecasts in forecasts/
    multivariate/
      regression.py     # Ridge helper and column checks
      swe_regression.py
      inflow_chain.py
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
data/
  benchmarks/nrcs_outlooks.csv          # Published NRCS outlooks vs actual peaks, 2024-2026
  external/nrcs_gsl_inflow_forecasts.csv # Issued GSL inflow exceedance forecasts
forecasts/              # Dated forecast CSVs committed by the monthly workflow
program.md              # Autoresearch strategy for the multivariate models
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
| `snotel_sites` | NRCS AWDB | Active SNOTEL sites in HUC 1601 (Bear), 160201 (Weber), 160202 (Provo-Jordan) |
| `snotel_daily` | NRCS AWDB | Daily SWE and water-year precipitation per site, 1978-present |
| `usgs_discharge_daily` | Sites 10126000, 10141000, 10170490 | Daily mean discharge, cfs |
| `monthly_covariates` | Derived | Month-end basin-mean SWE and precipitation (per basin and pooled), site count, monthly inflow in kaf per river and total |

Basin SWE is a plain mean over the sites reporting at month end, so it drifts as the site roster grows; a percent-of-median version is on the roadmap.

Before 1960 the daily table holds about one reading per month, and before 1980 about two, so `monthly_elevation` rows from that era are single readings rather than averages. That is the main reason training from 1960 onward beats the full 1847-present series, and why `train_start` is a config setting rather than a flag to remember.

## Tests and lint

```bash
uv run --frozen pytest
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
```

CI runs the same three commands on every push.
