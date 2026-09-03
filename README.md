# GSL Forecast

A dated, verified, year-round forecast of the Great Salt Lake's water surface elevation.

## Goal

Build the best live forecast of the water surface elevation of the Great Salt Lake south arm. A railway causeway divides the lake into a north arm and a south arm. The south arm holds the cities, the industry and almost all of the inflow, so it is the arm that decisions are about. The gauge is USGS 10010000 at Saltair, run by the United States Geological Survey. Elevation is in feet above sea level, and each value is the mean over the days of one calendar month.

The forecast runs every month of the year and covers the next 24 months. It gives a range, not one number. Every forecast carries the date it was made, and the project scores it against the gauge as the observations arrive.

The models use snowpack and streamflow measurements and a simple water balance. A model that uses the lake record alone is the comparison for them.

Two numbers carry most of the decision weight:

- Spring peak elevation: the highest monthly mean over April, May and June
- Water-year-end elevation: the September monthly mean, which is the annual low. A water year runs from October 1 to September 30, so September is its last month

The first 6 months are the operational window, where snowpack makes the lake predictable. Months 6 to 24 are the gap that no other forecast fills.

## Terms

The lake and its measurements:

| Term | Meaning |
|---|---|
| North arm, south arm | A railway causeway divides the lake. The south arm receives almost all of the inflow, and it is the arm this project forecasts |
| Head difference | The level of the south arm minus the level of the north arm. The state manages the gap in the causeway, so this number records that decision |
| Water year | October 1 to September 30. The lake fills over the winter and falls over the summer, so this is the natural year for it |
| SWE | Snow water equivalent: the depth of water held in the snowpack, in inches. It is the best early measure of how much water will reach the lake |
| SNOTEL | The NRCS network of automated snow measurement sites in the mountains around the lake |
| NRCS | The Natural Resources Conservation Service, part of the United States Department of Agriculture. It runs SNOTEL and issues the only other dated forecast of the lake |
| USGS | The United States Geological Survey. It runs the lake and river gauges |
| kaf | Thousand acre-feet, the unit for a volume of water |
| Hypsometry | The table that converts a lake elevation into a surface area and a volume |

The forecast and how it is scored:

| Term | Meaning |
|---|---|
| Cutoff | The last month with data behind a forecast |
| Issue date | The first day of the month after the cutoff, and the date on the forecast |
| Lead (`h`) | The number of months from the issue date to the month being forecast |
| Horizon | The longest lead the forecast covers, here 24 months |
| Interval (q05-q95) | The range that holds the correct value in 90 of 100 past forecasts at the same lead |
| Walk-forward cross-validation | Repeat the whole procedure at many past cutoffs, using only the data available at each one, then score the results |
| MAE | Mean absolute error, in feet |
| CRPS | One score for the whole range, not only the middle of it. Lower is better |
| Coverage | The share of actual values that fall inside the 90% interval. 0.90 is correct |
| Data vintage | The state of the input data on the issue date |
| Headline model | The model that supplies the public number, named in `config/config.json` |
| Production models | The subset written to dated files each month, in `src/forecasting/registry.py` |

## Why

The statistical specification of the models, and the date each input becomes available, are in `docs/model-spec.md`. A survey of existing forecasts is in `docs/operational-forecasts-survey.md` and the literature review in `docs/literature-review.md`. In short:

- The only routine, dated product that targets lake elevation is the NRCS Utah Snow Survey's advisory rise-to-peak outlook, issued January through May since 2024. Its April-issue peak error was 0.1-0.4 ft in 2024-2026 against a stated band of about plus or minus half a foot. It stops in May, so nothing operational forecasts the autumn low or anything beyond six months.
- CBRFC issues ensemble streamflow forecasts for the tributaries (about 16-18% April error on April-July volume) but no lake product.
- Long-range models (USU Climate Center's climate-oscillation regression, the Strike Team's 30-year Monte Carlo, the state's GSLIM planning model) are scenario tools or multi-year statistical forecasts with roughly 3 ft RMSE at 8 years, and none is verified as a dated forecast.

In season, the forecast to beat is the NRCS outlook. Out of season, the comparison is this project's own model that uses the lake record alone. From a winter cutoff that model does no better than a repeat of the last value (see [Current results](#current-results)).

## Roadmap

- [x] Univariate baselines with walk-forward CV and experiment tracking
- [x] Survey of operational and gray-literature forecasts
- [x] Score the headline scalars (spring peak, water-year-end low) by issue month and place them next to the NRCS record in `data/benchmarks/`
- [x] Ingest covariates: SNOTEL basin snow water equivalent and precipitation, USGS inflow gauges (Bear 10126000, Weber 10141000, Jordan 10170490)
- [x] Multivariate models: SWE regression (the NRCS method) and a reduced-form inflow-chain water balance
- [x] Probabilistic output (q05-q95) from walk-forward errors, scored with CRPS and 90% coverage in CV
- [x] Monthly GitHub Actions run that commits dated forecasts to `forecasts/`, plus `gsl-verify`, which scores them against the gauge as their target months arrive
- [x] Feature store: percent-of-median snowpack, soil moisture, reservoir storage, north-arm level and breach flow, the issued NRCS inflow forecast (all from live APIs)
- [x] Autoresearch program for the multivariate models (`docs/program.md`); one pass run, recorded in `docs/autoresearch.log`
- [x] Bathymetry (USGS elevation-area-volume table) as `inflow_chain_area`; climate-division temperature and precipitation ingested
- [x] One blended official model for the 24-month product (`blend`), and a public page on GitHub Pages
- [ ] Reservoir storage and percent-of-median snowpack in the production models (via the program loop)
- [ ] A performance page for out-of-sample accuracy

## Overview

The pipeline fetches the daily south-arm elevation from USGS, aggregates it to monthly means, joins the monthly covariates, runs a suite of time-series forecasters, and tracks experiments with [experiment-tracker](https://github.com/jcblsn/experiment-tracker). Results can be visualized and compared across models and forecast horizons.

Data sources: the USGS Water Data API and the NRCS Air and Water Database (AWDB) API. Storage: DuckDB.

The models fall into 2 groups. The first group uses the lake record alone, and it sets the level that a useful model must clear. The second group adds snowpack, streamflow and the difference in level between the 2 arms of the lake. The public number comes from a model that mixes one from each group. See [Current results](#current-results).

## Setup

```bash
uv sync
uv run --frozen pytest
```

Use `--frozen` on every command. Without it, `uv run` re-resolves the environment and can
change `uv.lock` as a side effect.

If `gsl-pipeline` appears to hang, the cause is usually IPv6. On some networks the route to
`api.waterdata.usgs.gov` accepts the connection and then stops. `requests` tries IPv6 first
and waits for the timeout, so each call takes about 40 seconds instead of less than 1
second. Force IPv4 for local runs:

```bash
uv run --frozen python -c "
import urllib3.util.connection as c, socket
c.allowed_gai_family = lambda: socket.AF_INET
from src.pipeline.elt import main; main()
"
```

Append the pipeline flags after the closing quote, for example `--skip-covariates`.

Modelling choices live in `config/config.json` under `forecasting`: `train_start`, `horizon` (24 months), `experiment_db`, `output_dir`, `headline_model`, and the CV cutoff policy. `headline_model` names the model that supplies the public headline, so that role can move to another model without a code change. CLI flags override config; anything not passed falls back to config.

## CLI Commands

### Run the ELT pipeline

Fetches the daily south-arm elevation, then the covariates (see the [Data](#data) section), and populates the local DuckDB. Incremental: each table is fetched from its max date, and USGS series re-fetch the trailing 45 days so provisional values that USGS later revises are replaced. The first run pulls about 50 years of daily data and takes a few minutes.

```bash
uv run --frozen gsl-pipeline [--skip-covariates]
```

Elevation commits in its own transaction before the covariates, so an AWDB outage leaves the target series current. The current calendar month is excluded from `monthly_elevation` so a partial month is never treated as a full-month average.

USGS WaterServices, the old source, is decommissioned in early 2027; the pipeline uses the replacement Water Data API. An API key is optional and raises the rate limit; set `USGS_API_KEY` in the environment to use one.

### Run forecasts

Fits the production subset of models (see `src/forecasting/registry.py`) on history from `train_start` and writes forward predictions to the `forecasts` table, tagged with `run_id`, `experiment_id`, and `data_max` so every prediction is traceable to a run and a data vintage.

```bash
uv run --frozen gsl-forecast [--horizon 24] [--train-start 1960-01-01] [--experiment-db forecast_experiments.db] [--export forecasts/2026-09-01.csv --intervals outputs/cv_results_<stamp>.parquet] [--site-data-dir site/data] [--allow-incomplete]
```

`--export` writes a dated forecast file (issue month, target month, lead, model, point forecast, and q05-q95 when `--intervals` names a CV results file to take empirical error quantiles from) and a `<stamp>.meta.json` sidecar with the data vintage and the headline calibration. A complete headline issue also writes a `<stamp>.explain.json` sidecar with the input contributions for all 24 target dates. `--site-data-dir` writes the JSON files the public page reads; an incomplete run updates the status file only, so the published headline stays in place. Before fitting, the CLI checks that the series ends last month, that the last month has at least 28 daily readings, and that the snowpack covariates and the arm head difference are present at the cutoff; `--allow-incomplete` overrides the last two checks, a stale series always stops the run. The monthly GitHub Actions workflow (`.github/workflows/forecast.yml`) runs pipeline, CV, forecast and verification, retries the forecast with `--allow-incomplete` if the strict run fails, and commits the file under `forecasts/`.

### Walk-forward cross-validation

Uses every month-end cutoff in the last `history_years` (about 170) by default, fits every registered model at each cutoff, evaluates at h=1..24, and logs per-horizon MAE, RMSE, and MAE relative to `naive_last` to the experiment tracker. Per-cutoff results are saved as parquet under `outputs/` so errors can be sliced by season of cutoff.

```bash
uv run --frozen gsl-cv [--n-cutoffs 20] [--horizon 24] [--history-years 15] [--train-start 1960-01-01] [--output-dir outputs] [--no-plots]
```

Pass `--n-cutoffs N` for a seeded random sample instead of all cutoffs, and `--models a,b` to evaluate a subset (the `naive_last` baseline is always included).

Besides per-horizon MAE, CV logs the two headline scalars by issue date (`peak_mae_jan` … `peak_mae_may` and `wyend_mae_jan` … `wyend_mae_aug`: the spring peak and September level as forecast from data ending the previous month) and probabilistic scores (`crps_h1` … and `cov90_h1` …) from leave-one-year-out empirical intervals. A `headline_<stamp>.parquet` sits next to the per-cutoff parquet.

The intervals hold out by cutoff year, but a cutoff late in year Y shares target months with cutoffs early in Y+1, so scores at long leads are slightly optimistic.

### Benchmark against NRCS

```bash
uv run --frozen gsl-benchmark [--model ets_damped_s12]
```

Fits the named model at each published NRCS issue date and prints the NRCS outlook record from `data/benchmarks/nrcs_outlooks.csv` (issue date, implied peak, actual peak) next to the model's spring-peak forecast. NRCS actuals are daily peaks; ours are peaks of the monthly mean, so one column rescores NRCS against the monthly-mean actual. The last columns place the published median inflow forecast (from `nrcs_inflow_forecasts`, since 2024) next to `inflow_chain`'s stage-one volume for the same period and the gauged volume. The NRCS point is a synthetic Bear-Weber-Provo total with its own normal, so compare percent of normal rather than kaf.

### Hindcast from a past cutoff

```bash
uv run --frozen gsl-hindcast 2022-03 [--models swe_regression,ets_damped_s12] [--horizon 24] [--cv outputs/.../cv_results_<stamp>.parquet] [--output-dir outputs/<today>]
```

Fits the named models on data through the given month, charts their forecasts (with q05-q95 from other years' CV errors) against the observed monthly means, and writes `<YYYYMMDD>_gsl_hindcast.png` and `.csv` under `outputs/<today>/`. Prints MAE by lead block, the spring-peak forecast vs observed, and 90% coverage.

### Verify issued forecasts

```bash
uv run --frozen gsl-verify [--forecast-dir forecasts]
```

Joins every dated forecast in `forecasts/` to the observed monthly means and writes MAE, bias and 90% coverage by model and lead to `forecasts/verification.csv`. Every row uses the data vintage that was available when the forecast was issued.

### Build the public page

```bash
quarto render site
```

Renders `site/` to `site/_site`. The page reads `site/data/latest.json` and
`site/data/status.json`, which `gsl-forecast --site-data-dir` writes, so a render never
refits a model. `.github/workflows/pages.yml` deploys the result to GitHub Pages on a push
to `main` that changes the site or the model specification.

### Plot forecasts

Generates a plotnine chart of historical elevation + all model forecasts.

```bash
uv run --frozen gsl-plot [--history-years 10] [--output outputs/gsl_forecast.png]
```

## Current results

Walk-forward cross-validation: 157 month-end cutoffs from August 2011 to August 2024,
24-month horizon, training from 1960, data through August 2026. Every number below comes
from one run, `GSL_CV_20260903_0004`, and `data/results/` holds that run. MAE is in feet.
The ratio is `blend` MAE divided by `naive_last` MAE at the same lead, so below 1.00 beats
a repeat of the last value. `gsl-results --tables` prints these 3 tables from those files,
so a published number and the run behind it cannot drift apart.

| Lead | blend | swe_head | swe_regression | blend_swe | inflow_chain | ets_damped_s12 | naive_last | Ratio (blend) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.13 | 0.13 | 0.12 | 0.12 | 0.15 | 0.14 | 0.34 | 0.38 |
| 3 | 0.32 | 0.32 | 0.31 | 0.31 | 0.37 | 0.45 | 0.90 | 0.35 |
| 6 | 0.52 | 0.51 | 0.55 | 0.57 | 0.68 | 0.82 | 1.33 | 0.39 |
| 9 | 0.77 | 0.76 | 0.79 | 0.82 | 0.98 | 1.08 | 1.26 | 0.62 |
| 12 | 1.07 | 1.07 | 1.07 | 1.07 | 1.32 | 1.25 | 1.28 | 0.83 |
| 18 | 1.61 | 1.72 | 1.56 | 1.53 | 1.86 | 1.65 | 1.91 | 0.85 |
| 24 | 2.00 | 2.32 | 1.95 | 1.86 | 2.31 | 1.92 | 1.79 | 1.11 |

CRPS and 90% coverage. CRPS scores the whole range, and coverage should be 0.90:

| Lead | blend | swe_head | swe_regression | blend_swe | inflow_chain | ets_damped_s12 | naive_last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.19 / 0.88 | 0.18 / 0.88 | 0.19 / 0.89 | 0.19 / 0.89 | 0.18 / 0.87 | 0.26 / 0.89 | 0.40 / 0.89 |
| 12 | 0.35 / 0.89 | 0.35 / 0.89 | 0.36 / 0.89 | 0.35 / 0.88 | 0.32 / 0.89 | 0.38 / 0.88 | 0.38 / 0.87 |

The 2 headline numbers, by the date the forecast goes out (MAE, ft):

| Target | Issue | blend | swe_head | swe_regression | blend_swe | inflow_chain | ets_damped_s12 | naive_last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spring peak | Jan 1 | 0.71 | 0.70 | 0.85 | 0.87 | 0.84 | 1.01 | 1.62 |
| Spring peak | Feb 1 | 0.57 | 0.56 | 0.61 | 0.63 | 0.62 | 0.87 | 1.39 |
| Spring peak | Mar 1 | 0.43 | 0.42 | 0.44 | 0.46 | 0.42 | 0.70 | 1.03 |
| Spring peak | Apr 1 | 0.30 | 0.30 | 0.26 | 0.27 | 0.27 | 0.41 | 0.62 |
| Spring peak | May 1 | 0.12 | 0.12 | 0.14 | 0.14 | 0.12 | 0.18 | 0.29 |
| Water-year end | Jan 1 | 0.90 | 0.89 | 0.97 | 1.02 | 1.13 | 1.28 | 1.23 |
| Water-year end | Apr 1 | 0.50 | 0.50 | 0.46 | 0.50 | 0.78 | 0.94 | 1.58 |
| Water-year end | Jun 1 | 0.36 | 0.36 | 0.32 | 0.32 | 0.53 | 0.56 | 1.78 |
| Water-year end | Jul 1 | 0.30 | 0.29 | 0.26 | 0.29 | 0.37 | 0.55 | 1.61 |
| Water-year end | Aug 1 | 0.19 | 0.18 | 0.18 | 0.18 | 0.19 | 0.30 | 1.05 |

Snowpack settles the winter case. From a January 1 issue, the model that uses the lake record
alone has a peak error of 1.01 ft, which is no better than a repeat of the last value. Adding
snowpack and the head difference between the arms brings it to 0.70 ft. The summer decline is
also predictable: from a June 1 issue, the September level is known to about a third of a
foot, against 1.78 ft for a repeat of the last value.

`blend` carries the headline because it is best or tied on both headline numbers, and it
holds that position through lead 12. It is not best everywhere. Past lead 18 it loses to
`blend_swe` and to `swe_regression`, because it inherits the long-lead weakness of
`swe_head`. At lead 24 no model beats a repeat of the last value: `naive_last` is 1.79 ft
against 1.86 ft for the best model. The 24-month path is therefore honest about its own
limit rather than useful at the far end.

`inflow_chain_area`, which puts lake area from the hypsometry table in place of the level,
scores within 0.04 ft of `inflow_chain` at every lead, so the hypsometry does not yet help.

The 90% interval covers 0.87 to 0.89 of the actual values at leads 6 and 12, against a
nominal 0.90. Section 7 of `docs/model-spec.md` explains why the long-lead figure is
slightly optimistic.

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

CV runs log `mae_h<h>`, `rmse_h<h>`, and `mae_ratio_h<h>` (MAE divided by `naive_last` MAE at the same horizon) for every lead per model, so any horizon is directly queryable. `uv run --frozen gsl-results <experiment_id>` prints a table of the metrics the autoresearch loop compares, ranked by `mae_h6`; `--metric` changes the rank and `--all-metrics` prints every logged lead.

The tracker database is a scratchpad, and `.gitignore` excludes it. The record behind every published number is `data/results/`, which each `gsl-cv` run replaces:

| File | Content |
|---|---|
| `cv_summary.csv` | MAE, RMSE, MAE ratio, CRPS and 90% coverage per model and lead |
| `headline_summary.csv` | Spring-peak and water-year-end MAE per model and issue month |
| `cv_summary.meta.json` | The run label, cutoff span, horizon, `train_start`, data vintage and git commit |

`uv run --frozen gsl-results --tables` prints the markdown in [Current results](#current-results) from those files, so the published tables and the committed record cannot drift apart.

## Models

The table gives the name of each model in the code. The first 9 use the lake record alone.
`swe_regression` and below add other measurements. `blend` is the headline model. See
[Terms](#terms) for SWE, head difference and hypsometry.

| Model | Description |
|-------|-------------|
| `naive_last` | Repeat last observed value |
| `naive_seasonal` | Repeat same month from prior year |
| `ma_simple_{3,6,12}` | Simple moving average over N months |
| `drift_{12,24,60}m` | Project average slope over last N months |
| `ets_add_s12` | Holt-Winters: additive trend + additive seasonal |
| `ets_damped_s12` | Holt-Winters: damped additive trend + additive seasonal (the best model that uses the lake record alone) |
| `ets_add_noseas` | Holt linear trend, no seasonal component |
| `ets_damped_noseas` | Holt damped trend, no seasonal component |
| `theta` | Theta method: SES plus half the linear trend slope |
| `swe_regression` | For the cutoff's calendar month and each lead, regresses the change in elevation on current level, basin month-end SWE and water-year precipitation across past years (the NRCS outlook generalised to every month and lead) |
| `inflow_chain` | Snowpack predicts each future month's tributary inflow; a fitted monthly bucket step (change as a function of that month's inflow and the starting level) rolls the elevation forward |
| `swe_head` | `swe_regression` plus the south-minus-north head difference; the best spring-peak model from January and February issues in CV, worse beyond lead 15 |
| `inflow_chain_area` | The same with lake area from the USGS hypsometry in place of the level, so the evaporation term scales with area |
| `state_space` | The same water balance as a state-space model: the level is a latent state, a Kalman filter gives the likelihood, and the 24-month path is 1 recursion. Better than `inflow_chain` past lead 12, worse to lead 8 |
| `blend` | The official model: `w` on `swe_head` and `1 - w` on `ets_damped_s12`, with `w` fitted for each lead and each issue season, and forced to fall with the lead |
| `blend_swe` | The same with `swe_regression` as the snowpack component |

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
    cutoffs.py          # The rule that selects walk-forward cutoffs
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
      regression.py     # Standardised ridge with a GCV penalty, and the fallback rule
      swe_regression.py
      inflow_chain.py
      blend.py          # The official model: a per-season, per-lead mix of 2 components
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
forecasts/              # One CSV per issue, named for the issue date (YYYY-MM-DD), with its sidecars
site/                   # Quarto public page and the JSON data it reads
docs/                   # Model spec, surveys, literature review, and the autoresearch program
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
| `climdiv_monthly` | NOAA nClimDiv, Utah divisions 03 (North Central) and 05 (Northern Mountains) | Monthly mean temperature and precipitation, 1895-present; a month is released around the 8th of the next month, so the cutoff month is always missing at issue time. Only the `_lag1` copies reach `monthly_covariates`, so a model cannot read a value that does not exist at issue time |
| `nrcs_inflow_forecasts` | NRCS AWDB forecast point 10010000:UT:USGS | Published Great Salt Lake inflow forecast at 10/30/50/70/90 percent exceedance and the period normal, monthly January-May since 2024 |
| `monthly_covariates` | Derived | One row per complete month: `swe_eom_*`, `prec_wy_eom_*`, `swe_pct_median_*`, `prec_pct_median_*`, `sms_eom_*` per basin and pooled (`_gsl`) with `n_snotel_sites`; `res_kaf_*` per basin and `res_kaf_total` with `n_reservoirs`; `inflow_kaf_*` per river and `inflow_kaf_total`; `breach_kaf`; `north_arm_ft` and `head_diff_ft` (south minus north); `tavg_f_gsl_lag1` and `prcp_in_gsl_lag1`, the climate columns shifted one month |

Snowpack at month end is the mean over sites reporting that day. The raw mean drifts as the roster grows (18 sites in 1979, 55 in 2026), which the percent-of-median columns avoid; young sites without a 30-year median count in the mean but not in the percent. Reservoir storage is summed over the stations reporting, so sums before a dam was built are smaller for a physical reason. The south-arm level is also managed at the causeway: the breach berm was raised in 2022, overtopped in 2023, and HB1001 (2025) lets the state raise it to 4,192 ft when the south arm is at or below 4,190 ft; `head_diff_ft` and `breach_kaf` carry that signal.

Before 1960 the daily table holds about one reading per month, and before 1980 about two, so `monthly_elevation` rows from that era are single readings rather than averages. That is the main reason training from 1960 onward beats the full 1847-present series, and why `train_start` is a config setting rather than a flag to remember.

## Tests and lint

```bash
uv run --frozen pytest
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
quarto render site
```

CI runs the same checks on every push.
