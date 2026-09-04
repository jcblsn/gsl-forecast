# Great Salt Lake monthly elevation forecast

This repository produces dated, experimental forecasts of the Great Salt Lake’s south-arm water-surface elevation. The target is the calendar-month mean at [USGS gauge 10010000](https://waterdata.usgs.gov/monitoring-location/USGS-10010000/), reported in feet above the National Geodetic Vertical Datum of 1929 (NGVD 29). It is an elevation, not the lake’s depth.

The project tests whether the current lake state, mountain snowpack, and recent precipitation improve short-range forecasts over persistence, which repeats the last observed value. The main development result is a large reduction in mean absolute error (MAE) through 6 months. At lead 6, MAE falls from 1.33 ft for persistence to 0.58 ft for the prototype headline model. At lead 24, the headline model is worse than persistence: 1.92 versus 1.79 ft.

This is a research prototype, not a validated operational forecast. The evidence comes from a repeatedly used development period, not an independent test. Do not use it as the sole basis for water management, infrastructure, navigation, or public-health decisions.

[View the experimental forecast](https://jcblsn.github.io/gsl-forecast/) or read the [current method specification](site/methodology.qmd).

## Current evidence

The frozen evaluation uses 157 monthly cutoffs from August 2011 through August 2024. Each fit uses observations from October 1989 through its cutoff and predicts the next 24 monthly means. Lead 1 is the first unobserved month, which is also the issue month. Lead 24 is 23 calendar months after the issue date.

| Lead | Headline MAE (ft) | Persistence MAE (ft) | MAE improvement (ft), 95% interval |
| ---: | ---: | ---: | ---: |
| 1 | 0.13 | 0.34 | 0.21 [0.17, 0.25] |
| 3 | 0.33 | 0.90 | 0.57 [0.45, 0.68] |
| 6 | 0.58 | 1.33 | 0.75 [0.60, 0.90] |
| 12 | 1.08 | 1.28 | 0.21 [0.07, 0.36] |
| 18 | 1.56 | 1.91 | 0.35 [0.14, 0.55] |
| 24 | 1.92 | 1.79 | -0.13 [-0.46, 0.20] |

Improvement is persistence MAE minus headline MAE, so positive values favor the headline model. The intervals come from a paired circular moving-block bootstrap with 24-month blocks. They describe sensitivity to the evaluated sequence; they are not formal confidence intervals under a fully specified probability model.

### Results

The prototype has its clearest advantage at leads 1–6. Its estimated improvement remains positive at leads 12 and 18, but absolute error grows to 1.08 and 1.56 ft. At lead 24, the interval includes equal performance and the point estimate favors persistence.

The headline model is not the best candidate at every lead. In the same development data, the snowpack model has lower MAE at leads 3–18. A seasonal endpoint baseline has MAE of 0.10 ft at lead 1 and 1.70 ft at lead 24. These comparisons informed model development, so they are not independent selection evidence.

The nominal central 90% interval contains 87% of observations at leads 6 and 12 in aggregate. At lead 6, coverage ranges from 77% for April–June issues to 92% for July–October issues.

### Interpretation

The development record supports a practical claim that the current hydrologic state improves near-term monthly forecasts. It does not establish 24-month skill or show that the prototype blend is the best specification. The regression inputs are correlated, so their fitted contributions describe the model and do not identify causal effects.

### Limits

- Adjacent forecasts share most target months. The 157 cutoffs represent about 13 water years, not 157 independent cases.
- The development period has guided repeated model choices. A sealed confirmation period has not been opened, and prospective forecasts only begin in September 2026.
- Historical evaluation uses the latest revised source data rather than the values available on each historical issue date.
- Observations before October 1989 often have sparse temporal support. The default fit excludes them rather than treating them as modern daily records.
- The interval is calibrated separately for each lead and issue season. It is not a joint distribution for the 24-month path, the spring maximum, or the date of a minimum.
- The point forecast is fitted separately from the interval and need not equal its median.
- Long leads use little information about future weather, runoff, diversions, or causeway operations. Structural change can invalidate relationships estimated from 1989 onward.

The exact summary values are in [data/results/metrics.csv](data/results/metrics.csv). The [manifest](data/results/manifest.json) records the evaluation status, limitations, and file hashes.

## Forecast definition

A railway causeway separates the lake’s north and south arms. Most surface inflow enters the south arm, and gauge 10010000 represents its water-surface elevation. The pipeline averages the available daily means within each complete calendar month. It excludes the current partial month.

Each issue contains a point forecast and the 5th, 25th, 50th, 75th, and 95th percentiles calibrated from retrospective errors. The public page emphasizes 2 summaries of the monthly path:

- Maximum April–June monthly mean. This differs from a daily seasonal peak.
- September monthly mean, which is the end of the water year. A water year runs from October 1 through September 30. September is not necessarily the annual minimum.

The project does not forecast the north arm, a daily peak, salinity, lake area, or lake volume as public targets.

## Method summary

The headline model, `blend`, combines 2 forecasts:

- `swe_head` fits the elevation change separately at each lead. Its predictors are current elevation, snow water equivalent, water-year precipitation, and the difference between the 2 arms. Snow water equivalent (SWE) is the depth of liquid water stored in the snowpack.
- `ets_damped_s12` estimates a damped trend and a 12-month additive seasonal pattern from the lake record alone.

The blend weight varies by lead and by 3 issue seasons: snow accumulation, snowmelt, and summer recession. A nested walk-forward fit chooses weights that minimize absolute error while the covariate-model share cannot increase with lead. The direct regressions use only past years with the same cutoff month. Each lead is fitted separately.

This model is empirical. It does not conserve water or represent forecast uncertainty in future inflow. The repository also evaluates persistence, moving-average, drift, seasonal, inflow-recursion, structural state-space, and experimental storage-balance models. The [registry](src/forecasting/registry.py) defines the complete set and the subset exported in each issue.

The [method page](site/methodology.qmd) gives the equations, data timing, interval algorithm, evaluation design, and source links. The implementation is under [src/forecasting](src/forecasting).

## Install and test

The package requires Python 3.11 or later and uses [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
```

The site also requires [Quarto](https://quarto.org/). CI uses Quarto 1.7.32.

## Run the workflow

The following sequence refreshes the data, evaluates the models, writes a dated issue, verifies all observed issues, and renders the site. Replace `YYYY-MM` with the issue year and month.

```bash
uv run --frozen gsl-pipeline
uv run --frozen gsl-cv --no-plots
uv run --frozen gsl-forecast \
  --export forecasts/YYYY-MM-01.csv \
  --intervals \
  --site-data-dir site/data
uv run --frozen gsl-verify
quarto render site
```

`gsl-pipeline` writes `data/gsl.db`. The database and evaluation outputs are ignored by Git. The full development evaluation fits every registered model at all 157 cutoffs. Use `gsl-cv --n-cutoffs N` only for a faster diagnostic sample, not for a reported result.

A dated export is write-once. The command fails if its CSV, metadata sidecar, or explanation sidecar already exists. It also refuses stale or incomplete inputs unless `--allow-incomplete` is set. That option records the problems and suppresses a complete headline issue; it is intended for diagnosis, not routine publication.

Every command accepts `--help`. The main entry points are:

| Command | Purpose |
| --- | --- |
| `gsl-pipeline` | Refresh the target and covariates from their sources |
| `gsl-forecast` | Fit configured issue models and optionally export a dated issue |
| `gsl-cv` | Run walk-forward evaluation on an open policy split |
| `gsl-verify` | Score dated issues as observations become available |
| `gsl-benchmark` | Compare spring-maximum refits with the small NRCS outlook record |
| `gsl-hindcast` | Plot forecasts made from specified historical cutoffs |
| `gsl-audit` | Report closure errors in the experimental storage balance |
| `gsl-results` | Inspect an experiment or verify the frozen result snapshot |
| `gsl-plot` | Plot stored forecasts with recent observations |

Source services can be unavailable or can revise provisional values. The pipeline commits the target before it fetches covariates. A covariate outage can therefore leave those groups at different dates; the forecast command checks for this condition. Some networks also expose a broken IPv6 route to the USGS API, which can make each request stall. Test the endpoint over IPv4 if a refresh pauses repeatedly without an HTTP error.

## Reproducibility and provenance

The current configuration is [config/config.json](config/config.json). It fixes the gauge IDs, the 29-site SNOTEL roster, basin weights, training start, horizon, headline model, and evaluation policy. Data ingestion lives in [src/pipeline](src/pipeline), and tests for data availability and temporal leakage live in [tests](tests).

Primary data sources are:

- [USGS Water Data for gauge 10010000](https://waterdata.usgs.gov/monitoring-location/USGS-10010000/) for the target elevation. The [configuration](config/config.json) identifies the other USGS gauges.
- [NRCS Air and Water Database REST API](https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html) for SNOTEL observations, reservoir storage, and issued seasonal inflow forecasts.
- [USGS elevation-area-volume tables](https://doi.org/10.5066/P9DGG75W), preserved locally as [data/external/gsl_south_arm_hypsometry.csv](data/external/gsl_south_arm_hypsometry.csv).
- [NOAA NCEI daily summaries](https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation) for experimental airport weather inputs.
- [NOAA nClimDiv](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00005%3Bview%3Diso) for experimental climate-division inputs.
- The [Utah Geological Survey brine database](https://geology.utah.gov/popular/great-salt-lake/) for experimental salinity and salt-mass inputs.

The frozen summary in [data/results](data/results) is content-addressed, but its [experiment record](data/results/experiment.json) reports a dirty source worktree. The source commit therefore cannot reconstruct the exact evaluation code by itself. The snapshot preserves the reported numbers, not the row-level predictions used to calculate them.

Each file under [forecasts](forecasts) has a metadata sidecar. Current exports record the issue status, forecast version, code state, resolved-configuration hash, modeling-table hash, data support, and interval calibration. Older artifacts can predate some provenance fields. Keep results from different forecast versions and issue statuses separate.

## Repository map

| Path | Contents |
| --- | --- |
| `src/pipeline/` | Source ingestion, quality fields, and monthly feature construction |
| `src/forecasting/` | Models, intervals, evaluation, exports, and verification |
| `config/config.json` | Source IDs, rosters, model defaults, and evaluation policy |
| `data/results/` | Frozen development summary and integrity manifest |
| `forecasts/` | Immutable dated issues and sidecars |
| `site/` | Quarto source and generated forecast data |
| `tests/` | Unit, leakage, export, and end-to-end tests |
| `.archive/docs/` | Historical reviews, surveys, specifications, and experiment records |

The [archive index](.archive/docs/README.md) explains why each former document was retired.
