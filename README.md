# GSL Forecast

A dated, verified, year-round forecast of the Great Salt Lake's water surface elevation.

## Goal

Build the best live, continuously operationalized forecast of Great Salt Lake south-arm elevation (USGS gauge 10010000 at Saltair, feet): a dated, versioned, probabilistic monthly forecast for months 1-24 from the present that combines a univariate baseline with snowpack and streamflow covariates and an explicit water balance, runs every month of the year, and is scored publicly against the gauge as observations arrive.

Two scalars carry most of the decision weight and are the headline targets:

- Spring peak elevation (the April-June maximum of the monthly mean)
- Water-year-end elevation (the September mean, the annual low)

Horizons: 1-6 months is the operational window where snowpack makes the lake predictable; 6-24 months is the gap no one currently fills.

## Why

A survey of existing forecasts is in `docs/operational-forecasts-survey.md` and the literature review in `docs/literature-review.md`. In short:

- The only routine, dated product that targets lake elevation is the NRCS Utah Snow Survey's advisory rise-to-peak outlook, issued January through May since 2024. Its April-issue peak error was 0.1-0.4 ft in 2024-2026 against a stated band of about plus or minus half a foot. It stops in May, so nothing operational forecasts the autumn low or anything beyond six months.
- CBRFC issues ensemble streamflow forecasts for the tributaries (about 16-18% April error on April-July volume) but no lake product.
- Long-range models (USU Climate Center's climate-oscillation regression, the Strike Team's 30-year Monte Carlo, the state's GSLIM planning model) are scenario tools or multi-year statistical forecasts with roughly 3 ft RMSE at 8 years, and none is verified as a dated forecast.

The benchmark to beat in season is the NRCS outlook; out of season the benchmark is our own univariate model, which from a winter cutoff does no better than repeating the last value (see [Current results](#current-results)).

## Roadmap

- [x] Univariate baselines with walk-forward CV and experiment tracking
- [x] Survey of operational and gray-literature forecasts
- [x] Score the headline scalars (spring peak, water-year-end low) by issue month and place them next to the NRCS record in `data/benchmarks/`
- [x] Ingest covariates: SNOTEL basin snow water equivalent and precipitation, USGS inflow gauges (Bear 10126000, Weber 10141000, Jordan 10170490)
- [x] Multivariate models: SWE regression (the NRCS method) and a reduced-form inflow-chain water balance
- [x] Probabilistic output (q05-q95) from walk-forward errors, scored with CRPS and 90% coverage in CV
- [x] Monthly GitHub Actions run that commits dated forecasts to `forecasts/`, plus `gsl-verify` for a live skill record
- [x] Feature store: percent-of-median snowpack, soil moisture, reservoir storage, north-arm level and breach flow, the issued NRCS inflow forecast (all from live APIs)
- [x] Autoresearch program for the multivariate models (`docs/program.md`); loop not yet run
- [x] Bathymetry (USGS elevation-area-volume table) as `inflow_chain_area`; climate-division temperature and precipitation ingested
- [ ] Reservoir storage and percent-of-median snowpack in the production models (via the program loop)
- [ ] One blended official model for the 24-month product

## Overview

The pipeline fetches the daily south-arm elevation from USGS, aggregates it to monthly means, joins the monthly covariates, runs a suite of time-series forecasters, and tracks experiments with [experiment-tracker](https://github.com/jcblsn/experiment-tracker). Results can be visualized and compared across models and forecast horizons.

Data sources: the USGS Water Data API and the NRCS Air and Water Database (AWDB) API. Storage: DuckDB.

Best univariate model (walk-forward CV, every month-end cutoff in the last 15 years, training from 1960): Holt-Winters with damped additive trend and additive 12-month seasonality (`ets_damped_s12`). See [Current results](#current-results).

## Setup

```bash
uv sync
uv run --frozen pytest
```

Modelling choices live in `config/config.json` under `forecasting`: `train_start`, `horizon` (24 months), `experiment_db`, `output_dir`, and the CV cutoff policy. CLI flags override config; anything not passed falls back to config.

## CLI Commands

### Run the ELT pipeline

Fetches the daily south-arm elevation, then the covariates (see the [Data](#data) section), and populates the local DuckDB. Incremental: each table is fetched from its max date, and USGS series re-fetch the trailing 45 days so provisional values that USGS later revises are replaced. The first run pulls about 50 years of daily data and takes a few minutes.

```bash
uv run gsl-pipeline [--skip-covariates]
```

Elevation commits in its own transaction before the covariates, so an AWDB outage leaves the target series current. The current calendar month is excluded from `monthly_elevation` so a partial month is never treated as a full-month average.

USGS WaterServices, the old source, is decommissioned in early 2027; the pipeline uses the replacement Water Data API. An API key is optional and raises the rate limit; set `USGS_API_KEY` in the environment to use one.

### Run forecasts

Fits the production subset of models (see `src/forecasting/registry.py`) on history from `train_start` and writes forward predictions to the `forecasts` table, tagged with `run_id`, `experiment_id`, and `data_max` so every prediction is traceable to a run and a data vintage.

```bash
uv run gsl-forecast [--horizon 24] [--train-start 1960-01-01] [--experiment-db forecast_experiments.db] [--export forecasts/2026-09.csv --intervals outputs/cv_results_<stamp>.parquet] [--allow-incomplete]
```

`--export` writes a dated forecast file (issue month, target month, lead, model, point forecast, and q05-q95 when `--intervals` names a CV results file to take empirical error quantiles from) and a `<stamp>.meta.json` sidecar with the data vintage. Before fitting, the CLI checks that the series ends last month, that the last month has at least 28 daily readings, and that the snowpack covariates are present at the cutoff; `--allow-incomplete` overrides the last two checks, a stale series always stops the run. The monthly GitHub Actions workflow (`.github/workflows/forecast.yml`) runs pipeline, CV, forecast and verification, retries the forecast with `--allow-incomplete` if the strict run fails, and commits the file under `forecasts/`.

### Walk-forward cross-validation

Uses every month-end cutoff in the last `history_years` (about 170) by default, fits every registered model at each cutoff, evaluates at h=1..24, and logs per-horizon MAE, RMSE, and MAE relative to `naive_last` to the experiment tracker. Per-cutoff results are saved as parquet under `outputs/` so errors can be sliced by season of cutoff.

```bash
uv run gsl-cv [--n-cutoffs 20] [--horizon 24] [--history-years 15] [--train-start 1960-01-01] [--output-dir outputs] [--no-plots]
```

Pass `--n-cutoffs N` for a seeded random sample instead of all cutoffs, and `--models a,b` to evaluate a subset (the `naive_last` baseline is always included).

Besides per-horizon MAE, CV logs the two headline scalars by issue date (`peak_mae_jan` … `peak_mae_may` and `wyend_mae_jan` … `wyend_mae_aug`: the spring peak and September level as forecast from data ending the previous month) and probabilistic scores (`crps_h1` … and `cov90_h1` …) from leave-one-year-out empirical intervals. A `headline_<stamp>.parquet` sits next to the per-cutoff parquet.

The intervals hold out by cutoff year, but a cutoff late in year Y shares target months with cutoffs early in Y+1, so scores at long leads are slightly optimistic.

### Benchmark against NRCS

```bash
uv run gsl-benchmark [--model ets_damped_s12]
```

Fits the named model at each published NRCS issue date and prints the NRCS outlook record from `data/benchmarks/nrcs_outlooks.csv` (issue date, implied peak, actual peak) next to the model's spring-peak forecast. NRCS actuals are daily peaks; ours are peaks of the monthly mean, so one column rescores NRCS against the monthly-mean actual. The last columns place the published median inflow forecast (from `nrcs_inflow_forecasts`, since 2024) next to `inflow_chain`'s stage-one volume for the same period and the gauged volume. The NRCS point is a synthetic Bear-Weber-Provo total with its own normal, so compare percent of normal rather than kaf.

### Hindcast from a past cutoff

```bash
uv run gsl-hindcast 2022-03 [--models swe_regression,ets_damped_s12] [--horizon 24] [--cv outputs/.../cv_results_<stamp>.parquet] [--output-dir outputs/<today>]
```

Fits the named models on data through the given month, charts their forecasts (with q05-q95 from other years' CV errors) against the observed monthly means, and writes `<YYYYMMDD>_gsl_hindcast.png` and `.csv` under `outputs/<today>/`. Prints MAE by lead block, the spring-peak forecast vs observed, and 90% coverage.

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

Walk-forward CV, 157 month-end cutoffs (August 2011 to August 2024), 24-month horizon, training from 1960, data through August 2026. MAE in feet; ratio is MAE divided by `naive_last` MAE at the same horizon. CRPS is the mean pinball loss over q05-q95 with intervals taken from other years' errors (leave-one-year-out).

| Horizon | swe_regression | inflow_chain | ets_damped_s12 | naive_last | Ratio (swe) |
|---|---|---|---|---|---|
| 1 | 0.13 | 0.15 | 0.14 | 0.34 | 0.37 |
| 3 | 0.31 | 0.37 | 0.45 | 0.90 | 0.35 |
| 6 | 0.55 | 0.67 | 0.82 | 1.33 | 0.41 |
| 9 | 0.80 | 0.97 | 1.08 | 1.26 | 0.64 |
| 12 | 1.11 | 1.31 | 1.24 | 1.28 | 0.86 |
| 18 | 1.65 | 1.87 | 1.65 | 1.91 | 0.86 |
| 24 | 2.08 | 2.36 | 1.92 | 1.80 | 1.16 |

CRPS at h=6: swe_regression 0.19, inflow_chain 0.18, ets_damped_s12 0.26, naive_last 0.40.

Headline scalars by issue date (MAE, ft). Issue date means the outlook made from data through the previous month, matching the NRCS schedule.

| Target | Issue | swe_regression | inflow_chain | ets_damped_s12 | naive_last |
|---|---|---|---|---|---|
| Spring peak | Jan 1 | 0.84 | 0.82 | 1.01 | 1.62 |
| Spring peak | Feb 1 | 0.63 | 0.58 | 0.87 | 1.39 |
| Spring peak | Mar 1 | 0.44 | 0.40 | 0.70 | 1.03 |
| Spring peak | Apr 1 | 0.27 | 0.25 | 0.41 | 0.62 |
| Spring peak | May 1 | 0.13 | 0.12 | 0.18 | 0.29 |
| Water-year end | Jan 1 | 0.91 | 1.09 | 1.28 | 1.23 |
| Water-year end | Apr 1 | 0.43 | 0.73 | 0.94 | 1.58 |
| Water-year end | Jun 1 | 0.32 | 0.53 | 0.56 | 1.78 |
| Water-year end | Jul 1 | 0.29 | 0.37 | 0.55 | 1.61 |
| Water-year end | Aug 1 | 0.19 | 0.19 | 0.30 | 1.05 |

Snowpack resolves the winter case: from a January 1 issue the univariate model's peak error was no better than naive; with month-end SWE it roughly halves. After the peak, the summer decline is also predictable: from a June 1 issue the September level is known to about a third of a foot, against 1.8 ft for persistence. `inflow_chain_area` (lake area in place of level in the bucket step) scores within 0.02 ft of `inflow_chain` at every lead, so the hypsometry does not yet add skill. Beyond 18 months the covariate models lose to the univariate ones, since snowpack known today says nothing about the next winter, so the 24-month product should blend toward `ets_damped_s12` at long leads (on the roadmap).

### Against the NRCS record

`gsl-benchmark --refit` fits a model at each published NRCS issue date and scores its spring peak against the peak of the monthly mean (NRCS is scored against the same, last column). Errors in feet, signed (forecast minus actual).

| Issue | NRCS | swe_regression | inflow_chain | ets_damped_s12 |
|---|---|---|---|---|
| 2024-03-01 | +0.30 | -0.47 | -0.17 | -0.19 |
| 2024-04-01 | 0.00 | +0.04 | +0.21 | +0.19 |
| 2025-02-01 | +0.07 | -0.09 | +0.24 | +0.43 |
| 2025-04-01 | +0.47 | +0.20 | +0.42 | +0.41 |
| 2025-05-01 | +0.17 | +0.01 | +0.12 | +0.15 |
| 2026-01-01 | +0.14 | +0.37 | +0.46 | +0.50 |
| 2026-04-01 | +0.04 | -0.11 | +0.07 | +0.38 |
| Mean absolute | 0.17 | 0.18 | 0.24 | 0.32 |

Seven issues is too few to rank anyone. Two caveats favour the models: they use today's data vintage rather than what was available at the time, and the actual here is the monthly-mean peak rather than the daily peak NRCS is usually judged on. Two favour NRCS: its numbers were published in advance, and it stops in May while these models also produce the autumn low and the next year.

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

CV runs log `mae_h<h>`, `rmse_h<h>`, and `mae_ratio_h<h>` (MAE divided by `naive_last` MAE at the same horizon) for every lead per model, so any horizon is directly queryable. `uv run gsl-results <experiment_id>` prints a ranked table.

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
| `inflow_chain_area` | The same with lake area from the USGS hypsometry in place of the level, so the evaporation term scales with area |

All models implement the `Forecaster` ABC (`src/forecasting/base.py`) with `fit(df)`, `predict(h)`, and `get_metrics()`. The single list of models is `all_forecasters()` in `src/forecasting/registry.py`; `production_forecasters()` is the subset written by `gsl-forecast`.

## Project Structure

```
src/
  pipeline/
    usgs.py             # USGS Water Data API fetcher, retry, upsert
    elt.py              # South-arm elevation into DuckDB, monthly_elevation, transactions
    covariates.py       # SNOTEL, reservoirs, discharge, north arm, NRCS forecasts; monthly_covariates
    climate.py          # NOAA nClimDiv monthly temperature and precipitation
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
    benchmark.py        # gsl-benchmark: refit peaks and inflow next to the NRCS record
    hypsometry.py       # South-arm area and volume from elevation (USGS 2023 tables)
    verify.py           # gsl-verify: score dated forecasts in forecasts/
    hindcast.py         # gsl-hindcast: chart a past cutoff against observations
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
  config.json           # Site IDs, AWDB station sets, DB path, modelling defaults
tests/                  # One file per module; in-memory DuckDB and fake HTTP responses
data/
  benchmarks/nrcs_outlooks.csv   # Published NRCS outlooks vs actual peaks, 2024-2026
  external/gsl_south_arm_hypsometry.csv  # USGS 2023 elevation-area-volume table, 0.1 ft steps
forecasts/              # Dated forecast CSVs and meta sidecars committed by the monthly workflow
docs/                   # Surveys, literature review, and the autoresearch program
outputs/                # gitignored: CV parquet and PNGs
```

## Data

Everything is stored in DuckDB (`./data/gsl.db`). Every source is a live API, so a forecast issued in any future month has the same inputs as this one.

| Table | Source | Content |
|-------|--------|---------|
| `usgs_water_surface_elevation_daily` | USGS 10010000 (Saltair, south arm) | Daily mean elevation, 1847-present, with approval status |
| `monthly_elevation` | Derived | Monthly avg/min/max/count, complete months only |
| `forecasts` | Model output | Monthly predictions with run_id, experiment_id, data_max |
| `snotel_sites`, `snotel_daily` | NRCS AWDB | Active SNOTEL sites in HUC 1601 (Bear), 160201 (Weber), 160202 (Provo-Jordan); daily SWE, water-year precipitation, 8-inch soil moisture, and the 1991-2020 median of SWE and precipitation, 1978-present |
| `reservoir_sites`, `reservoir_monthly` | NRCS AWDB (Bureau of Reclamation stations) | End-of-month storage, kaf, for the 21 reservoirs in the same units (Bear Lake from 1911, Utah Lake from 1932, Jordanelle from 1993) |
| `usgs_discharge_daily` | USGS 10126000 (Bear), 10141000 (Weber), 10170490 (Jordan plus Surplus Canal), 10010020 (causeway breach) | Daily mean discharge, cfs |
| `usgs_north_arm_elevation_daily` | USGS 10010100 (Saline) | Daily north-arm elevation, 1966-present |
| `climdiv_monthly` | NOAA nClimDiv, Utah divisions 03 (North Central) and 05 (Northern Mountains) | Monthly mean temperature and precipitation, 1895-present; a month is released around the 8th of the next month, so the cutoff month is always missing at issue time |
| `nrcs_inflow_forecasts` | NRCS AWDB forecast point 10010000:UT:USGS | Published Great Salt Lake inflow forecast at 10/30/50/70/90 percent exceedance and the period normal, monthly January-May since 2024 |
| `monthly_covariates` | Derived | One row per complete month: `swe_eom_*`, `prec_wy_eom_*`, `swe_pct_median_*`, `prec_pct_median_*`, `sms_eom_*` per basin and pooled (`_gsl`) with `n_snotel_sites`; `res_kaf_*` per basin and `res_kaf_total` with `n_reservoirs`; `inflow_kaf_*` per river and `inflow_kaf_total`; `breach_kaf`; `north_arm_ft` and `head_diff_ft` (south minus north); `tavg_f_gsl` and `prcp_in_gsl` |

Snowpack at month end is the mean over sites reporting that day. The raw mean drifts as the roster grows (18 sites in 1979, 55 in 2026), which the percent-of-median columns avoid; young sites without a 30-year median count in the mean but not in the percent. Reservoir storage is summed over the stations reporting, so sums before a dam was built are smaller for a physical reason. The south-arm level is also managed at the causeway: the breach berm was raised in 2022, overtopped in 2023, and HB1001 (2025) lets the state raise it to 4,192 ft when the south arm is at or below 4,190 ft; `head_diff_ft` and `breach_kaf` carry that signal.

Before 1960 the daily table holds about one reading per month, and before 1980 about two, so `monthly_elevation` rows from that era are single readings rather than averages. That is the main reason training from 1960 onward beats the full 1847-present series, and why `train_start` is a config setting rather than a flag to remember.

## Tests and lint

```bash
uv run --frozen pytest
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
```

CI runs the same three commands on every push.
