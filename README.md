# GSL Forecast

An experimental prototype for dated, year-round forecasts of Great Salt Lake water-surface
elevation. This is not yet a quality release or a validated operational product.

The retrospective development record has important limits. Target observations before 1990
have materially different temporal support. The displayed band is a nominal central 90%
interval calibrated from retrospective errors, at 1 lead at a time and for the season the
issue falls in; observed coverage is about 85–90% at the reported key leads. The published line is the model's
point forecast and is not necessarily the interval median. There is no demonstrated advantage
over persistence at 12–24 months; at lead 24 the published `blend` is worse than persistence.

## Goal

Develop and evaluate a live forecast of the water surface elevation of the Great Salt Lake
south arm. A railway causeway divides the lake into a north arm and a south arm. The south arm
holds the cities, the industry and almost all of the inflow, so it is the arm that decisions are
about. The gauge is USGS 10010000 at Saltair, run by the United States Geological Survey.
Elevation is in feet above sea level, and each value is the mean over the days of one calendar
month.

The prototype runs every month of the year and covers the next 24 months. It gives a range,
not one number. Every experimental issue carries the date and model version that produced it,
and the project scores it against the gauge as observations arrive.

The models use lake elevation, snowpack, streamflow and the difference between the north and
south arms. The current inflow-based model is an empirical elevation recursion, not a closed
physical water balance. Models that use the lake record alone provide comparisons.

Two numbers carry most of the decision weight:

- Maximum April–June monthly mean
- September mean (water-year end). A water year runs from October 1 to September 30;
  September is its last month, but it is not necessarily the seasonal or annual minimum

The first 6 months are the decision-relevant window where snowpack adds predictive information.
Months 6 to 24 remain an experimental research horizon.

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
| Interval (q05-q95) | A nominal central 90% interval calibrated from retrospective errors at the same lead and in the same issue season. It is a band at 1 lead, not a sample from a trajectory |
| Walk-forward cross-validation | Repeat the whole procedure at many past cutoffs, using only the data available at each one, then score the results |
| MAE | Mean absolute error, in feet |
| Mean pinball loss | The unweighted mean of pinball losses at q05, q25, q50, q75 and q95. Lower is better |
| Coverage | The share of actual values inside the nominal central 90% interval |
| Data vintage | The state of the input data on the issue date |
| Headline model | The model that supplies the public number, named in `config/config.json` |
| Production models | The subset written to dated files each month, in `src/forecasting/registry.py` |

## Why

The statistical specification of the models, and the date each input becomes available, are in `docs/model-spec.md`. A survey of existing forecasts is in `docs/operational-forecasts-survey.md` and the literature review in `docs/literature-review.md`. In short:

- The only routine, dated product that targets lake elevation is the NRCS Utah Snow Survey's advisory rise-to-peak outlook, issued January through May since 2024. Its April-issue peak error was 0.1-0.4 ft in 2024-2026 against a stated band of about plus or minus half a foot. It stops in May, so nothing operational forecasts the September mean or anything beyond six months.
- CBRFC issues ensemble streamflow forecasts for the tributaries (about 16-18% April error on April-July volume) but no lake product.
- Long-range models (USU Climate Center's climate-oscillation regression, the Strike Team's 30-year Monte Carlo, the state's GSLIM planning model) are scenario tools or multi-year statistical forecasts with roughly 3 ft RMSE at 8 years, and none is verified as a dated forecast.

In season, the forecast to beat is the NRCS outlook. Out of season, the comparison is this
project's own model that uses the lake record alone. From a winter cutoff that model does no
better than a repeat of the last value (see [Frozen development results](#frozen-development-results)).

## Roadmap

- [x] Univariate baselines with walk-forward CV and experiment tracking
- [x] Survey of operational and gray-literature forecasts
- [x] Score the maximum April–June monthly mean and September mean by issue month and place the first next to the NRCS record in `data/benchmarks/`
- [x] Ingest covariates: SNOTEL basin snow water equivalent and precipitation, USGS inflow gauges (Bear 10126000, Weber 10141000, Jordan 10170490)
- [x] Multivariate models: SWE regression (the NRCS method) and an inflow-driven elevation recursion
- [x] Probabilistic output (q05-q95) from walk-forward errors, scored with mean pinball loss and coverage in CV
- [x] Monthly GitHub Actions run that commits dated forecasts to `forecasts/`, plus `gsl-verify`, which scores them against the gauge as their target months arrive
- [x] Feature store: percent-of-median snowpack, soil moisture, reservoir storage, north-arm level and breach flow, the issued NRCS inflow forecast (all from live APIs)
- [x] Historical autoresearch pass for the multivariate models, recorded in `docs/autoresearch.log`; the active keep/revert loop is retired
- [x] Bathymetry (USGS elevation-area-volume table) as `inflow_chain_area`; climate-division temperature and precipitation ingested
- [x] One blended prototype headline model (`blend`), and an experimental page on GitHub Pages
- [ ] Reservoir storage and stable percent-of-median snowpack in candidate models
- [ ] A smooth weight curve over the issue month, before a 3-component blend takes the headline
- [ ] A performance page for out-of-sample accuracy

## Overview

The pipeline fetches the daily south-arm elevation from USGS, aggregates it to monthly means, joins the monthly covariates, runs a suite of time-series forecasters, and tracks experiments with [experiment-tracker](https://github.com/jcblsn/experiment-tracker). Results can be visualized and compared across models and forecast horizons.

Data sources: the USGS Water Data API and the NRCS Air and Water Database (AWDB) API. Storage: DuckDB.

The models fall into 2 groups. The first group uses the lake record alone, and it sets the level
that a useful model must clear. The second group adds snowpack, streamflow and the difference in
level between the 2 arms of the lake. The displayed prototype number comes from a model that
mixes one from each group. See [Frozen development results](#frozen-development-results).

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

Modelling choices live in `config/config.json` under `forecasting`: `train_start`, `horizon`,
`experiment_db`, `output_dir`, `headline_model`, `issue_status`, and `forecast_version`.
The separate `evaluation_policy` object is the single machine-readable source for fixed
development, sealed confirmation, and prospective cohorts.

## CLI Commands

### Run the ELT pipeline

Fetches the daily south-arm elevation, then the covariates (see the [Data](#data) section), and populates the local DuckDB. Incremental: each table is fetched from its max date, and USGS series re-fetch the trailing 45 days so provisional values that USGS later revises are replaced. The first run pulls about 50 years of daily data and takes a few minutes.

```bash
uv run --frozen gsl-pipeline [--skip-covariates]
```

Elevation commits in its own transaction before the covariates, so an AWDB outage leaves the
target series current. The current calendar month is excluded from `monthly_elevation` so a
partial month is never treated as a full-month average. The same table retains the last valid
daily elevation, its age at month end, robust 3- and 7-day endpoint estimates, their support,
and provisional-observation counts.

USGS WaterServices, the old source, is decommissioned in early 2027; the pipeline uses the replacement Water Data API. An API key is optional and raises the rate limit; set `USGS_API_KEY` in the environment to use one.

### Run forecasts

Fits the production subset of models (see `src/forecasting/registry.py`) on history from `train_start` and writes forward predictions to the `forecasts` table, tagged with `run_id`, `experiment_id`, and `data_max` so every prediction is traceable to a run and a data vintage.

```bash
uv run --frozen gsl-forecast [--horizon 24] [--train-start 1989-10-01] [--experiment-db forecast_experiments.db] [--export forecasts/2026-09-01.csv --intervals] [--site-data-dir site/data] [--allow-incomplete]
```

`--export` writes a dated CSV and required metadata sidecar. The sidecar records the issue
status, forecast version, code commit and dirty-tree state, evaluation-policy version, data
vintage, and headline calibration. A complete headline issue also writes an explanation
sidecar. The three dated paths are write-once: if any already exists, the export fails before
publishing anything. Mutable `site/data` views may be regenerated from the issue. Before
fitting, the CLI checks data recency, gauge support in the last month, and required covariates.

### Walk-forward cross-validation

Uses the configured named development split by default: exactly 157 monthly cutoffs from
2011-08-01 through 2024-08-01 at horizon 24. It logs the evaluation-policy version and exact
bounds. The sealed confirmation split is rejected. Per-cutoff results are saved under
`outputs/`; no committed results snapshot is written by default.

```bash
uv run --frozen gsl-cv [--split development] [--n-cutoffs 20] [--train-start 1989-10-01] [--output-dir outputs] [--no-plots] [--results-dir new/empty/path]
```

Pass `--n-cutoffs N` for a seeded sample within the named cohort, and `--models a,b` to
evaluate a subset. `--results-dir` is opt-in and refuses any nonempty target directory.

Besides per-horizon MAE, CV logs the targets `apr_jun_monthly_mean_max` and
`september_monthly_mean` by issue date. It also logs `mean_pinball_loss` and `cov90` by lead
from leave-one-year-out empirical intervals. A `headline_<stamp>.parquet` sits next to the
per-cutoff parquet.

The intervals hold out by cutoff year, but a cutoff late in year Y shares target months with cutoffs early in Y+1, so scores at long leads are slightly optimistic.

### Benchmark against NRCS

```bash
uv run --frozen gsl-benchmark [--model ets_damped_s12]
```

Fits the named model at each published NRCS issue date and compares the maximum April–June
monthly mean with the NRCS daily-peak outlook. The estimands differ, so both definitions remain
explicit in the output.

### Hindcast from a past cutoff

```bash
uv run --frozen gsl-hindcast 2022-03 [--models swe_regression,ets_damped_s12] [--horizon 24] [--cv outputs/.../cv_results_<stamp>.parquet] [--output-dir outputs/<today>]
```

Fits the named models on data through the given month, charts their forecasts with retrospective
q05-q95 bands, and writes a PNG and CSV under `outputs/<today>/`.

### Verify issued forecasts

```bash
uv run --frozen gsl-verify [--forecast-dir forecasts]
```

Requires and validates each dated issue's metadata, joins its forecast to observed monthly
means, and writes MAE, bias and coverage grouped separately by issue status, forecast version,
model, and lead. Specifications are never silently pooled.

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

## Frozen development results

Walk-forward cross-validation: 157 month-end cutoffs from August 2011 to August 2024,
24-month horizon, training from 1960, data through August 2026. This repeatedly used cohort is
development evidence, not an untouched test set. Every number below comes from
`GSL_CV_20260903_0004`; `data/results/manifest.json` freezes and hashes its snapshot. MAE is in feet.
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

Mean pinball loss and nominal central-90% coverage. Each cell is loss / observed coverage:

| Lead | blend | swe_head | swe_regression | blend_swe | inflow_chain | ets_damped_s12 | naive_last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.19 / 0.88 | 0.18 / 0.88 | 0.19 / 0.89 | 0.19 / 0.89 | 0.18 / 0.87 | 0.26 / 0.89 | 0.40 / 0.89 |
| 12 | 0.35 / 0.89 | 0.35 / 0.89 | 0.36 / 0.89 | 0.35 / 0.88 | 0.32 / 0.89 | 0.38 / 0.88 | 0.38 / 0.87 |

The 2 reported summaries, by issue date (MAE, ft):

| Target | Issue | blend | swe_head | swe_regression | blend_swe | inflow_chain | ets_damped_s12 | naive_last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maximum April–June monthly mean | Jan 1 | 0.71 | 0.70 | 0.85 | 0.87 | 0.84 | 1.01 | 1.62 |
| Maximum April–June monthly mean | Feb 1 | 0.57 | 0.56 | 0.61 | 0.63 | 0.62 | 0.87 | 1.39 |
| Maximum April–June monthly mean | Mar 1 | 0.43 | 0.42 | 0.44 | 0.46 | 0.42 | 0.70 | 1.03 |
| Maximum April–June monthly mean | Apr 1 | 0.30 | 0.30 | 0.26 | 0.27 | 0.27 | 0.41 | 0.62 |
| Maximum April–June monthly mean | May 1 | 0.12 | 0.12 | 0.14 | 0.14 | 0.12 | 0.18 | 0.29 |
| September mean (water-year end) | Jan 1 | 0.90 | 0.89 | 0.97 | 1.02 | 1.13 | 1.28 | 1.23 |
| September mean (water-year end) | Apr 1 | 0.50 | 0.50 | 0.46 | 0.50 | 0.78 | 0.94 | 1.58 |
| September mean (water-year end) | Jun 1 | 0.36 | 0.36 | 0.32 | 0.32 | 0.53 | 0.56 | 1.78 |
| September mean (water-year end) | Jul 1 | 0.30 | 0.29 | 0.26 | 0.29 | 0.37 | 0.55 | 1.61 |
| September mean (water-year end) | Aug 1 | 0.19 | 0.18 | 0.18 | 0.18 | 0.19 | 0.30 | 1.05 |

Snowpack settles the winter case. From a January 1 issue, the model that uses the lake record
alone has an April–June monthly-mean maximum error of 1.01 ft, which is no better than a repeat of the last value. Adding
snowpack and the head difference between the arms brings it to 0.70 ft. The summer decline is
also predictable: from a June 1 issue, the September level is known to about a third of a
foot, against 1.78 ft for a repeat of the last value.

`blend` is the current prototype headline because it is best or tied on both summaries, and it
has the lowest point estimate of MAE through lead 12. The uncertainty in paired comparisons
does not demonstrate an advantage over persistence at leads 12–24. Past lead 18 it loses to
`blend_swe` and to `swe_regression`, because it inherits the long-lead weakness of
`swe_head`. At lead 24 no model beats a repeat of the last value: `naive_last` is 1.79 ft
against 1.86 ft for the best model and 2.00 ft for `blend`.

`inflow_chain_area`, which puts lake area from the hypsometry table in place of the level,
scores within 0.04 ft of `inflow_chain` at every lead, so the hypsometry does not yet help.

The nominal central 90% interval is calibrated from retrospective errors, at 1 lead at a time
and for the season the issue falls in. One band over every issue month gave the `blend` a
coverage of 0.82 at lead 6 from an accumulation issue and 0.98 from a recession issue, both
2.31 ft wide. The season-conditional band is 2.89 ft and 1.65 ft wide, at 0.85 and 0.85. The
band is still marginal at each lead, so it does not give a probability for the spring maximum
or the date of the minimum. The published point line is not necessarily q50. Section 7 of
`docs/model-spec.md` gives the limits.

### Against the NRCS record

`gsl-benchmark --refit` fits a model at each published NRCS issue date and scores its maximum
April–June monthly mean. NRCS normally targets a daily peak, so the comparison retains both
definitions. Errors are signed forecast minus actual, in feet.

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

Seven issues is too few to rank anyone. The refits use today's data vintage rather than the
vintage available at issue time. NRCS forecasts were genuinely issued in advance, and its daily
peak target differs from the monthly-mean maximum scored for these models.

## Querying Results

The experiment tracker CLI (`expt`) can query any logged metric. It finds
`forecast_experiments.db` by itself, so no `--db` flag is needed from the repository root:

```bash
# List experiments, newest first
expt list

# Runs for an experiment, with their notes
expt runs <experiment_id>

# The best model at a lead. Lowest wins, because the metric is an error
expt best <experiment_id> --metric mae --dim h=6

# MAE by lead, one column per lead
expt metrics --metric mae --pivot h

# What changed between 2 runs
expt diff <run_a> <run_b> --metric mae

# One line per run, the shape of docs/autoresearch.log
expt log <experiment_id> --metric mae --dim h=6

# Recompute a stored metric from the prediction rows behind it
expt audit <run_id> --metric mae --dim h=6
```

CV runs log `mae`, `rmse`, `mae_ratio`, `mean_pinball_loss`, and `cov90`. Lead, target,
and issue month are dimensions rather than parts of stored metric names. `gsl-results
<experiment_id>` prints the historical research view; `--all-metrics` prints every logged row.

The tracker database is a working file, and `.gitignore` excludes it. `data/results/` is the
frozen development snapshot behind the tables above. `gsl-cv` never replaces it by default;
an explicit snapshot must target a new, empty directory:

| File | Content |
|---|---|
| `experiment.json` | The run label, cutoff span, horizon, `train_start`, data vintage, git commit, tree state and Python version |
| `runs.csv` | One row per model, with its parameters, status and note |
| `metrics.csv` | One row per metric and dimension, so every lead and issue month is in one table |
| `manifest.json` | Development-only status, source run and commit, limitations, numeric-value digest, and SHA-256 hashes |

`gsl-results --verify-manifest` verifies the hashes in CI. `gsl-results --tables` renders the
snapshot without treating it as untouched confirmation evidence.

Prediction rows stay in the database rather than the snapshot, because there are about 87,000
of them per run. So `expt audit` checks a published number against its own rows locally, and
the committed snapshot carries the summary rather than the evidence.


## Models

The table gives the name of each model in the code. The univariate baselines and experimental
`state_space` model use the lake record alone; the regression, recursion and blend models add
other measurements. `blend` is the experimental headline model. See [Terms](#terms) for SWE,
head difference and hypsometry.

| Model | Description |
|-------|-------------|
| `naive_last` | Repeat the latest monthly mean |
| `naive_seasonal` | Repeat same month from prior year |
| `endpoint_seasonal` | Select the last reading, 3-day median, or 7-day median by expanding training error, then add the historical median change for the issue month and lead |
| `ma_simple_{3,6,12}` | Simple moving average over N months |
| `drift_{12,24,60}m` | Project average slope over last N months |
| `ets_add_s12` | Holt-Winters: additive trend + additive seasonal |
| `ets_damped_s12` | Holt-Winters: damped additive trend + additive seasonal (the best model that uses the lake record alone) |
| `ets_add_noseas` | Holt linear trend, no seasonal component |
| `ets_damped_noseas` | Holt damped trend, no seasonal component |
| `theta` | Theta method: SES plus half the linear trend slope |
| `swe_regression` | For the cutoff's calendar month and each lead, regresses the change in elevation on current level, basin month-end SWE and water-year precipitation across past years (the NRCS outlook generalised to every month and lead) |
| `inflow_chain` | Snowpack predicts monthly tributary inflow; an empirical elevation-change regression rolls elevation forward. It does not conserve storage or close a physical water balance |
| `swe_head` | `swe_regression` plus the south-minus-north head difference; strong for the April–June maximum from winter issues, worse beyond lead 15 |
| `inflow_chain_area` | The same recursion with hypsometric area instead of elevation as one regressor; it is not a storage balance |
| `state_space` | Experimental structural state-space baseline: local linear trend and monthly seasonality evolve hypsometric south-arm storage before conversion back to elevation. It uses no forecast inflow and is outside production |
| `blend` | The prototype headline: `w` on `swe_head` and `1 - w` on `ets_damped_s12`, with `w` fitted for each lead and issue season |
| `blend3_swe`, `blend3_chain` | Historical three-component blend candidates retained outside production |
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
    headline.py         # April-June maximum and September-mean scoring by issue month
    quantiles.py        # Empirical intervals, mean pinball loss, coverage
    benchmark.py        # gsl-benchmark: refit peaks and inflow next to the NRCS record
    hypsometry.py       # South-arm area and volume from elevation (USGS 2023 tables)
    verify.py           # gsl-verify: score dated forecasts in forecasts/
    hindcast.py         # gsl-hindcast: chart a past cutoff against observations
    multivariate/
      regression.py     # Standardised ridge with a GCV penalty, and the fallback rule
      swe_regression.py
      inflow_chain.py
      blend.py          # Prototype headline: a per-season, per-lead mix of components
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
| `monthly_elevation` | Derived | Monthly mean/min/max and support plus last-valid, 3-day median, and 7-day median endpoint states, complete months only |
| `forecasts` | Model output | Monthly predictions with run_id, experiment_id, data_max |
| `snotel_sites`, `snotel_daily` | NRCS AWDB | SNOTEL sites in HUC 1601 (Bear), 160201 (Weber), 160202 (Provo-Jordan); daily SWE, water-year precipitation, 8-inch soil moisture, and the 1991-2020 median of SWE and precipitation, 1978-present |
| `snotel_roster` | Derived | The versioned set of sites the snow features use, with the basin and the basin weight of each site. `config/config.json` names it; the current version is `gsl-modern-complete-v1` |
| `reservoir_sites`, `reservoir_monthly` | NRCS AWDB (Bureau of Reclamation stations) | End-of-month storage, kaf, for the 21 reservoirs in the same units (Bear Lake from 1911, Utah Lake from 1932, Jordanelle from 1993) |
| `usgs_discharge_daily` | USGS 10126000 (Bear), 10141000 (Weber), 10170490 (Jordan plus Surplus Canal), 10010020 (causeway breach) | Daily mean discharge, cfs |
| `usgs_north_arm_elevation_daily` | USGS 10010100 (Saline) | Daily north-arm elevation, 1966-present |
| `climdiv_monthly` | NOAA nClimDiv, Utah divisions 03 (North Central) and 05 (Northern Mountains) | Monthly mean temperature and precipitation, 1895-present; a month is released around the 8th of the next month, so the cutoff month is always missing at issue time. Only the `_lag1` copies reach `monthly_covariates`, so a model cannot read a value that does not exist at issue time |
| `nrcs_inflow_forecasts` | NRCS AWDB forecast point 10010000:UT:USGS | Published Great Salt Lake inflow forecast at 10/30/50/70/90 percent exceedance and the period normal, monthly January-May since 2024 |
| `monthly_covariates` | Derived | One row per complete month: `swe_eom_*`, `prec_wy_eom_*`, `swe_pct_median_*`, `prec_pct_median_*`, `sms_eom_*` per basin and pooled (`_gsl`) with `n_snotel_sites`, `n_snotel_prec` and `n_snotel_sms`, and `snotel_roster_version`; `res_kaf_*` per basin and `res_kaf_total` with `n_reservoirs`; `inflow_kaf_*` per river and `inflow_kaf_total` with `inflow_day_coverage`; `breach_kaf`; `north_arm_ft` and `head_diff_ft` (south minus north); `tavg_f_gsl_lag1` and `prcp_in_gsl_lag1`, the climate columns shifted one month |

Snowpack at month end is the mean over the roster sites reporting in the last 5 days of the month, each site at its own last valid day, and each variable counting its own reporting sites. Every pooled (`_gsl`) column averages the basins under the roster's declared basin weights, so the basin with the most sites does not decide the index. The roster is fixed and versioned, so the index does not change when AWDB retires a site or when an earlier run leaves an extra site in the database. Young sites without a 30-year median count in the mean but not in the percent. Reservoir storage is summed over the stations reporting, so sums before a dam was built are smaller for a physical reason. The south-arm level is also managed at the causeway: the breach berm was raised in 2022, overtopped in 2023, and HB1001 (2025) lets the state raise it to 4,192 ft when the south arm is at or below 4,190 ft; `head_diff_ft` and `breach_kaf` carry that signal.

Before October 1989 the target observations have materially different temporal support; many
nominal monthly means are based on sparse readings rather than near-daily coverage. New model
fits therefore default to 1989-10 onward. Earlier rows remain available only for explicit
sensitivity experiments.

## Tests and lint

```bash
uv run --frozen pytest
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
quarto render site
```

CI runs the same checks on every push.
