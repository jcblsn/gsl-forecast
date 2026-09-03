# Model specification

This file states what the forecast predicts, which inputs it uses, and how each model
estimates its coefficients. The README gives the commands and the project structure. This
file gives the statistics.

## 1 Target

The target is `avg_elevation`: the mean of the daily water surface elevation of the Great
Salt Lake south arm, in feet, at USGS gauge 10010000 (Saltair). The pipeline averages the
daily gauge readings to one value per calendar month. The pipeline excludes the current
month, so a partial month never enters the series.

Two scalars carry the decision weight. The models do not target them directly. The scoring
code extracts both from the monthly path.

| Identifier | Public label | Definition | Issue dates scored |
|---|---|---|---|
| `apr_jun_monthly_mean_max` | Maximum April–June monthly mean | The maximum of the April, May and June monthly means | January 1 to May 1 |
| `september_monthly_mean` | September mean (water-year end) | The September monthly mean; not necessarily the seasonal or annual minimum | January 1 to August 1 |

An issue date is the first day of the month after the data cutoff. An outlook issued
February 1 uses data through January 31. This matches the NRCS schedule.

The forecast horizon is 24 months. New fits default to the homogeneous modern target record
beginning 1989-10-01. Earlier observations remain available for explicit sensitivity runs,
but they do not enter the default likelihood as if they had the same observation operator.

The target remains the monthly mean. The monthly table separately carries the last valid daily
elevation, its distance from calendar month end, and medians over the final 3 and 7 calendar
days. These endpoint fields are issue-time state estimates, not replacements for the target.
Approval support and the observation counts behind the monthly mean and endpoint estimates
remain alongside them.

## 2 Inputs and their dates

All inputs come from live APIs. The table gives the first and last month with a value in
`monthly_covariates` on 2026-09-02, and the release delay at issue time.

| Column | Source | First month | Last month | Delay at issue |
|---|---|---|---|---|
| `avg_elevation` | USGS 10010000 | 1847-10 | 2026-08 | None; provisional same day |
| `last_elevation`, `endpoint_3d_median`, `endpoint_7d_median` | USGS 10010000 | 1847-10 | 2026-08 | None; provisional same day |
| `swe_eom_gsl`, `prec_wy_eom_gsl` | NRCS SNOTEL | 1978-10 | 2026-08 | None; daily values post next day |
| `swe_pct_median_gsl` | NRCS SNOTEL | 1978-10 | 2026-05 | None, but October to May only |
| `prec_pct_median_gsl` | NRCS SNOTEL | 1978-10 | 2026-08 | None |
| `sms_eom_gsl` | NRCS SNOTEL | 1999-11 | 2026-08 | None |
| `inflow_kaf_total` | USGS 10126000, 10141000, 10170490 | 1949-10 | 2026-08 | None; provisional same day |
| `breach_kaf` | USGS 10010020 | 2008-10 | 2026-08 | None |
| `north_arm_ft`, `head_diff_ft` | USGS 10010100 | 1966-04 | 2026-08 | None |
| `res_kaf_total` | NRCS AWDB, 21 Reclamation stations | 1911-01 | 2026-08 | A few days |
| `tavg_f_gsl_lag1`, `prcp_in_gsl_lag1` | NOAA nClimDiv | 1895-02 | 2026-08 | None; the column is already shifted 1 month |
| `nrcs_inflow_forecasts` | NRCS AWDB forecast point | 2024-01 | 2026-05 | None; January to May only |

Three availability rules control which model may use which column.

1. The percent-of-median snowpack columns are NULL in June, July, August and September. The
   median of the site sum is 0 in those months, and the transform divides by NULLIF of that
   sum. A model that uses these columns has no features in the summer.
2. The nClimDiv values are 1 month behind at issue time. NOAA releases a month around the
   8th of the next month. The monthly workflow runs on the 2nd. So the cutoff month has no
   temperature or precipitation value when the forecast runs. Cross-validation reads the
   finished table and does not see this gap. Therefore `monthly_covariates` holds only the
   `_lag1` copies. The unlagged values stay in `climdiv_monthly`, which no model reads, and
   `tests/test_leakage.py` checks that no model names one.
3. The published NRCS inflow forecast exists for January to May of 2024, 2025 and 2026. That
   is 15 publication dates. This is too few to fit a coefficient on.

Two roster effects change the meaning of a raw mean over time. The SNOTEL roster grows from
18 sites in 1979 to 55 sites in 2026, so a raw basin mean drifts. The reservoir roster grows
as dams are built, so early storage sums are smaller for a physical reason.

## 2.1 The endpoint seasonal baseline

`endpoint_seasonal` is the strong state-only baseline. Within each fit it compares the last
daily elevation, the median of the final 3 calendar days, and the median of the final 7 calendar
days. Candidate comparison uses expanding one-step errors from targets already observed inside
that fit. For each lead, the forecast is the selected current endpoint plus the historical
median endpoint-to-target change among origins in the same calendar month. The monthly mean
remains the target; the endpoint is only the initial state. If endpoint fields are unavailable,
the latest monthly mean provides an explicit compatibility fallback.

## 3 The swe_regression model

This model generalises the NRCS outlook to every calendar month and every lead.

Write `y_t` for the elevation at the cutoff month `t`, and `m` for the calendar month of
`t`. For each lead `h` from 1 to 24, the model collects every past year `i` whose month is
also `m` and whose row `i + h` is exactly `h` months later. It then fits:

```
y_(i+h) - y_i = b0 + b1 * y_i + b2 * SWE_i + b3 * PREC_i + e
```

`SWE` is `swe_eom_gsl`, the mean month-end snow water equivalent over the reporting SNOTEL
sites in the Bear, Weber and Provo-Jordan basins. `PREC` is `prec_wy_eom_gsl`, the mean
water-year precipitation to date at the same sites. The forecast is:

```
y_hat(t+h) = y_t + b0 + b1 * y_t + b2 * SWE_t + b3 * PREC_t
```

Four properties follow from this design.

- The model fits each lead directly. It does not iterate a 1-step model, so it does not
  accumulate 1-step error. The 24 leads are separate fits, so the forecast path carries no
  smoothness constraint and the leads can disagree.
- Each fit uses 1 row per year. The rows inside one fit do not overlap in time. So the fit
  does not need an autocorrelation correction.
- The fit is per calendar month, so the model needs no seasonal term.
- The `b1 * y_i` term is a mean-reversion term. It sets how far the lake returns toward its
  own level over `h` months.

The effective sample is small. Snowpack starts in 1978-10, so a cutoff in 2011 gives 32
rows per fit and a cutoff in 2026 gives 47 rows, against 4 parameters.

The model uses a declared fallback rule. It drops the snowpack terms and fits
`y_(i+h) - y_i = b0 + b1 * y_i` when either condition holds:

- Fewer than `min_obs` rows (default 10) carry every feature.
- Any feature is NULL at the cutoff.

The registered variant `swe_head` adds `head_diff_ft`, the south arm level minus the north
arm level. The causeway berm controls this difference, so the term carries the management
signal.

## 4 The inflow_chain model

This model is an empirical inflow-driven elevation recursion in 2 stages. It does not conserve
storage or close a physical water balance.

Stage 1 predicts the tributary inflow volume. It uses the same design as section 3, with the
inflow volume at lead `h` as the target instead of the elevation change:

```
Q_(i+h) = g0 + g1 * SWE_i + g2 * PREC_i + e
```

The prediction is clipped at 0. When the fit has too few rows, or a feature is NULL at the
cutoff, the model falls back to the mean inflow for the target calendar month.

Stage 2 is a monthly elevation-change regression. For each calendar month `m` it fits, over
the whole record:

```
y_(s+1) - y_s = a_m + b_m * Q_(s+1) + c_m * S_s
```

`S_s` is the level `y_s`, or the lake area from the USGS hypsometry in the registered
variant `inflow_chain_area`. The `a_m` term absorbs the average effect of every omitted
process in that calendar month. The `c_m` term gives the fitted state dependence. Neither is
a measured evaporation, diversion or closure term. Stage 2 pools all years, so it has about
63 rows per calendar month.

The model then rolls the level forward 1 month at a time from the cutoff, and feeds each
month the stage-1 inflow for that lead.

This design has 2 known weaknesses.

- Stage 2 fits on observed inflow but runs on predicted inflow. The predicted inflow has
  less variance than the observed inflow, so the fitted `b_m` overstates the response of the
  lake to the predicted volume.
- The recursion accumulates error. The `c_m` term damps the path, but a stage-1 error at
  lead 3 still moves every later month.
- The state is elevation rather than conserved storage, and precipitation, evaporation,
  diversion and causeway exchange are not explicit. Its coefficients therefore do not have
  a physical water-balance interpretation.

## 4.1 The state_space model

This experimental model is a standard structural time-series model in storage coordinates.
It first converts each monthly mean elevation to south-arm storage with the USGS hypsometry
table and scales the result to million acre-feet. It then fits:

```
x_t       = mu_t + gamma_t
mu_t      = mu_(t-1) + beta_(t-1) + eta_t
beta_t    = beta_(t-1) + zeta_t
gamma_t   = -sum(gamma_(t-j), j=1..11)
```

Here `x_t` is transformed observed storage, `mu_t` is a stochastic local level, `beta_t` is
a stochastic local slope, and `gamma_t` is deterministic 12-month seasonality. The level and
slope disturbances are independent zero-mean Gaussian variables. The implementation uses
exact diffuse initialization and maximum likelihood through the Kalman filter. Forecasts are
Gaussian in storage and are transformed back to elevation through hypsometry.

This is a coherent state-space model: the full path evolves from a shared latent level and
slope, and its model-based variance propagates with the horizon. It is not a water-balance
model. It has no inflow or other forcing variables, no explicit observation-error term, and
no physical closure claim.

The replacement was evaluated at all 157 cutoffs in the open development cohort. These
results are additional development evidence and are not part of the frozen snapshot in
section 9:

| Lead | MAE (ft) | MAE / persistence | Mean pinball loss | Nominal 90% coverage |
|---|---:|---:|---:|---:|
| 1 | 0.170 | 0.51 | 0.054 | 0.89 |
| 3 | 0.566 | 0.63 | 0.177 | 0.88 |
| 6 | 1.106 | 0.83 | 0.346 | 0.88 |
| 9 | 1.585 | 1.26 | 0.500 | 0.87 |
| 12 | 1.971 | 1.53 | 0.649 | 0.88 |
| 18 | 3.094 | 1.62 | 1.039 | 0.88 |
| 24 | 3.881 | 2.16 | 1.329 | 0.88 |

It trails `ets_damped_s12` at every lead and persistence after lead 7. The registry keeps it
as a structural baseline for experiments, while `PRODUCTION_MODELS` excludes it.

## 5 The blend

No model is the best model at each lead. The current prototype headline model mixes 2 of
them:

```
pred(h) = w(s, h) * swe_head + (1 - w(s, h)) * ets_damped_s12
```

The model fits `w(s, h)`. It is a function of the forecast lead `h` and the issue season
`s`.

The lead is necessary because the snowpack signal stops. A weight that decreases with the
lead moves the forecast to the univariate model at the same rate.

The season is necessary because the same lead gives a different month in each season. A
lead of 6 months from a February issue gives the spring peak. The current snowpack controls
that value. The same lead from an August issue gives a month in the next accumulation
season. The current snowpack does not control that value. The 3 seasons are:

| Season | Issue months |
|---|---|
| Accumulation | November to March |
| Melt | April to June |
| Recession | July to October |

The fit is a walk-forward pass in the training data. At each inner cutoff the code predicts
both components for 24 leads and records the actual values. For each season the code then
selects the weight path with the lowest total absolute error. The path must not increase
with the lead. The grid is 0 to 1 in steps of 0.01.

A dynamic program finds that path directly. This is different from a fit of each lead
separately, followed by a correction of the sequence. The correction minimises a different
quantity: the squared error of the weights against the free fit. The dynamic program
minimises the stated error under the stated condition. For equal error, it selects the
smaller weight on the snowpack model.

The inner pass reads no row after its own cutoff. Therefore the harness scores the complete
procedure, and not a weight curve that used later data. `tests/test_leakage.py` changes each
value after the cutoff and requires the same result.

A direct inner pass fits both components one time for each outer cutoff, which is 157 times.
A prediction at an inner cutoff uses only the rows on or before that cutoff, so the value is
the same at each outer cutoff. The code keeps these values in a cache. The cost is then
linear in the number of cutoffs.

The inner pass uses the same 15-year window as the harness in section 8. A longer window
gives less noise in the weights, but it gives a biased result. Before approximately 1995 the
SNOTEL record is too short to fit the snowpack model. A 30-year window records a failure
that cannot occur now, and gives the snowpack model approximately half of the correct
weight.

Each season needs a minimum of 20 usable inner cutoffs at each lead. Below that number the
model holds a fixed ramp in place of a fitted curve. The ramp is a placeholder, not a
result. Therefore `gsl-forecast` refuses to publish a headline when the issue season holds
the ramp.

The snowpack component is `swe_head`. The registry also holds `blend_swe`, which is the same
model with `swe_regression` in that position. `docs/autoresearch.log` records the comparison.

## 5.1 Input contributions

Each lead is a separate direct fit. Therefore the forecast is the exact sum of a set of
terms. For the input `j` at the lead `h`:

```
contribution(j, h) = w(s, h) * beta(j, m, h) * (x(j, t) - mean(j, m, h))
```

`mean(j, m, h)` is the mean of that input over the rows of that fit. The term for the
current level also holds the direct `y_t` term from outside the regression. The reference
path holds 2 parts: the result of the snowpack fit for a cutoff with average inputs, and the
univariate part, which has no terms for the inputs.

The reference path plus each input contribution equals the point forecast. A test asserts
this.

These terms are parts of the fitted model. They are not causal effects. The inputs move
together, so one term is not the effect of a change to that input alone.

## 6 Estimation

Every direct regression fit in sections 3 and 4 uses the same estimator, in
`src/forecasting/multivariate/regression.py`.

The estimator standardises the design. It centres and scales each non-intercept column by
the training mean and standard deviation, solves the penalised normal equations, and maps
the coefficients back to the original scale. The intercept carries no penalty. This matters
because the level column is about 4192 and the snowpack column is about 10. Without
standardisation a single penalty cannot act on both columns.

The estimator chooses the penalty `alpha` per fit by generalised cross-validation over a
fixed grid. GCV uses only the rows inside the fit, so no future data enters the choice. A
caller can pass a fixed `alpha` to switch the search off.

The direct regression models do not report standard errors. The displayed prototype
uncertainty comes from the walk-forward errors in section 7.

## 7 Uncertainty

The point forecast gets an empirical interval. For each model and lead, the code takes the
quantiles of the walk-forward errors `actual - pred` and adds them to the point forecast.
The quantile set is 0.05, 0.25, 0.50, 0.75 and 0.95.

Two scores measure the intervals: the unweighted mean pinball loss over the quantile set and
the share of actuals inside the nominal central 90 percent interval. The loss is only the
unweighted five-quantile mean, not an integral over the full forecast distribution.

The displayed point forecast is fitted independently from this retrospective calibration.
It is not necessarily the q50 interval value. Aggregate observed coverage is about 87–89%
at the reported key leads, rather than 90%, and coverage varies by issue season.

The scoring holds out 1 year at a time. Each cutoff takes its interval from the errors of
other years. A cutoff late in year Y still shares target months with cutoffs early in year
Y plus 1, so scores at long leads are slightly optimistic.

## 8 Validation

The harness is walk-forward cross-validation in `src/forecasting/cross_validate.py`. Its
versioned policy defines three cohorts:

- `development`: exactly 157 monthly cutoffs from 2011-08-01 through 2024-08-01, horizon 24,
  and status `open_development`. This cohort has been consulted repeatedly and is unsuitable
  as untouched test evidence.
- `limited_confirmation`: monthly cutoffs from 2024-09-01 through 2025-08-01, maximum horizon
  12, and status `sealed`. `gsl-cv` refuses to open it. It may be opened once only after a
  candidate specification, its code and its acceptance rules are frozen.
- `prospective`: immutable monthly issues beginning 2026-09-01. They remain experimental,
  are grouped by forecast version, and are never reused to tune the version that produced
  them.

At each development cutoff the harness trains every model on data from 1989-10 through the
cutoff, then predicts 24 months. It records the policy version and exact bounds along with
MAE and RMSE per model and lead, MAE relative to `naive_last`, mean pinball loss, nominal
90% coverage, and the 2 headline scalars by issue month.

Two caveats apply to every number this harness produces.

1. The harness reads today's data. USGS revises provisional elevation and discharge, and the
   SNOTEL roster grew. A forecast issued in 2013 did not have these values. The live record
   in `forecasts/` is the only vintage-correct score.
2. Overlapping target months make long-lead interval scores slightly optimistic. See section
   7.

## 9 Frozen development accuracy

The maintained retrospective tables come from one cross-validation run,
`GSL_CV_20260903_0004`: 157
cutoffs from 2011-08 to 2024-08, data through 2026-08. `data/results/` holds a snapshot of
that run. Its manifest records the development-only status, limitations and hashes, and CI
verifies those hashes. The README section "Frozen development results" holds the tables, and
`gsl-results --tables` prints them from the snapshot. This repeatedly used record is not
untouched test evidence. `docs/autoresearch.log` is a historical experiment log.

The experiment tracker database is the working file for a run in progress. `.gitignore`
excludes it, so the snapshot rather than an experiment id is the citation.

In short:

- `blend` is best or equal on both headline numbers, and it holds that position through lead
  12. It is the model the page shows.
- `swe_head` is the best model for the maximum April–June monthly mean from a January issue,
  at 0.70 ft against 1.62 ft for a repeat of the last value.
- Past lead 18 `blend` loses to `blend_swe` and to `swe_regression`, because it uses
  `swe_head`, which is the weakest covariate model at long leads.
- At lead 24 no model beats a repeat of the last value. `naive_last` is 1.79 ft, against
  1.86 ft for the best model.
- The 90 percent interval covers 0.87 to 0.89 of the actual values at leads 6 and 12.

## 10 Open questions

1. The forecast path is 24 independent fits, except in `state_space`. No rule makes the
   direct-model path smooth, and no rule makes its intervals wider at longer leads. Section
   4.1 gives one coherent-path baseline, but it trails `ets_damped_s12` throughout the
   development cohort.
2. Stage 2 of `inflow_chain` fits on the observed inflow and runs on the predicted inflow.
   A future physical storage model must represent forecast-inflow uncertainty, precipitation,
   evaporation, diversion and residual closure jointly. The structural model in section 4.1
   deliberately does not claim to solve that problem.
3. Reservoir storage lowers accuracy as a plain extra regressor. Storage moves with the lake
   level and with the same trend, so it adds a collinear column and no new information. A
   measure of the deficit below capacity can separate the 2 signals.
4. The percent-of-median snowpack columns stop in June. The open experiment is a feature set
   for each season: snowpack from October to May, and soil moisture or year-to-date inflow
   from June to September.
5. The blend mixes 2 components. `inflow_chain` holds information that the other 2 do not
   hold, so a mix of 3 components can improve the middle leads. The weight search then needs
   a grid over a simplex.
6. No model uses the nClimDiv columns. The lagged copies from section 2 are now available,
   so this is an experiment and not a pipeline task.
7. The model fits the 3 seasonal weight curves separately. Each curve uses 39 to 65 cutoffs,
   so each curve holds more noise than one pooled curve. A smooth function of the issue
   month can give a better result than both.
