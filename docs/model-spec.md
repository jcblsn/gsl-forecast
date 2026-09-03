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

| Scalar | Definition | Issue dates scored |
|---|---|---|
| Spring peak | The maximum of the monthly mean over April, May and June | January 1 to May 1 |
| Water-year end | The September monthly mean | January 1 to August 1 |

An issue date is the first day of the month after the data cutoff. An outlook issued
February 1 uses data through January 31. This matches the NRCS schedule.

The forecast horizon is 24 months. The models train on data from 1960-01-01. Before 1960 the
daily table holds about 1 reading per month, so earlier monthly rows are single readings.

## 2 Inputs and their dates

All inputs come from live APIs. The table gives the first and last month with a value in
`monthly_covariates` on 2026-09-02, and the release delay at issue time.

| Column | Source | First month | Last month | Delay at issue |
|---|---|---|---|---|
| `avg_elevation` | USGS 10010000 | 1847-10 | 2026-08 | None; provisional same day |
| `swe_eom_gsl`, `prec_wy_eom_gsl` | NRCS SNOTEL | 1978-10 | 2026-08 | None; daily values post next day |
| `swe_pct_median_gsl` | NRCS SNOTEL | 1978-10 | 2026-05 | None, but October to May only |
| `prec_pct_median_gsl` | NRCS SNOTEL | 1978-10 | 2026-08 | None |
| `sms_eom_gsl` | NRCS SNOTEL | 1999-11 | 2026-08 | None |
| `inflow_kaf_total` | USGS 10126000, 10141000, 10170490 | 1949-10 | 2026-08 | None; provisional same day |
| `breach_kaf` | USGS 10010020 | 2008-10 | 2026-08 | None |
| `north_arm_ft`, `head_diff_ft` | USGS 10010100 | 1966-04 | 2026-08 | None |
| `res_kaf_total` | NRCS AWDB, 21 Reclamation stations | 1911-01 | 2026-08 | A few days |
| `tavg_f_gsl`, `prcp_in_gsl` | NOAA nClimDiv | 1895-01 | 2026-07 | 1 month |
| `nrcs_inflow_forecasts` | NRCS AWDB forecast point | 2024-01 | 2026-05 | None; January to May only |

Three availability rules control which model may use which column.

1. The percent-of-median snowpack columns are NULL in June, July, August and September. The
   median of the site sum is 0 in those months, and the transform divides by NULLIF of that
   sum. A model that uses these columns has no features in the summer.
2. The nClimDiv columns are 1 month behind at issue time. NOAA releases a month around the
   8th of the next month. The monthly workflow runs on the 2nd. So the cutoff month has no
   temperature or precipitation value when the forecast runs. Cross-validation reads the
   finished table and does not see this gap. A model must use a lagged copy of these
   columns, never the unlagged column at the cutoff.
3. The published NRCS inflow forecast exists for January to May of 2024, 2025 and 2026. That
   is 15 publication dates. This is too few to fit a coefficient on.

Two roster effects change the meaning of a raw mean over time. The SNOTEL roster grows from
18 sites in 1979 to 55 sites in 2026, so a raw basin mean drifts. The reservoir roster grows
as dams are built, so early storage sums are smaller for a physical reason.

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

This model is a reduced-form water balance in 2 stages.

Stage 1 predicts the tributary inflow volume. It uses the same design as section 3, with the
inflow volume at lead `h` as the target instead of the elevation change:

```
Q_(i+h) = g0 + g1 * SWE_i + g2 * PREC_i + e
```

The prediction is clipped at 0. When the fit has too few rows, or a feature is NULL at the
cutoff, the model falls back to the mean inflow for the target calendar month.

Stage 2 is a monthly bucket step. For each calendar month `m` it fits, over the whole
record:

```
y_(s+1) - y_s = a_m + b_m * Q_(s+1) + c_m * S_s
```

`S_s` is the level `y_s`, or the lake area from the USGS hypsometry in the registered
variant `inflow_chain_area`. The `a_m` term absorbs the mean net evaporation and diversion
for that month. The `c_m` term scales the loss with the size of the lake. Stage 2 pools all
years, so it has about 63 rows per calendar month.

The model then rolls the level forward 1 month at a time from the cutoff, and feeds each
month the stage-1 inflow for that lead.

This design has 2 known weaknesses.

- Stage 2 fits on observed inflow but runs on predicted inflow. The predicted inflow has
  less variance than the observed inflow, so the fitted `b_m` overstates the response of the
  lake to the predicted volume.
- The recursion accumulates error. The `c_m` term damps the path, but a stage-1 error at
  lead 3 still moves every later month.

## 5 The blend

No single model wins at every lead, so the official model mixes 2 of them:

```
pred(h) = w(h) * swe_regression + (1 - w(h)) * ets_damped_s12
```

The weight `w(h)` is fitted, not set by hand. A walk-forward pass inside the training data
scores both models at every lead. The weight is the value on a grid from 0 to 1, in steps of
0.05, that gives the lowest absolute error at that lead. A pool-adjacent-violators step then
forces `w(h)` to fall with the lead. So the blend gives up the snowpack term as the snowpack
signal expires, and it never oscillates.

The inner pass uses the same 15-year cutoff window as the harness in section 8. A wider
window gives less noisy weights, but it gives a biased answer. Before about 1995 the SNOTEL
record was too short to fit the snowpack model on. A 30-year window sees that model fail for
a reason that no longer applies, and gives it 0.60 weight at lead 6, where its true error is
0.51 ft against 0.82 ft for the univariate model.

The inner pass would refit both models once per outer cutoff, which is 157 times over. A
prediction at an inner cutoff uses only rows at or before it, so the result is the same at
every outer cutoff. The code memoises it on the training slice.

## 6 Estimation

Every fit in sections 3 and 4 uses the same estimator, in
`src/forecasting/multivariate/regression.py`.

The estimator standardises the design. It centres and scales each non-intercept column by
the training mean and standard deviation, solves the penalised normal equations, and maps
the coefficients back to the original scale. The intercept carries no penalty. This matters
because the level column is about 4192 and the snowpack column is about 10. Without
standardisation a single penalty cannot act on both columns.

The estimator chooses the penalty `alpha` per fit by generalised cross-validation over a
fixed grid. GCV uses only the rows inside the fit, so no future data enters the choice. A
caller can pass a fixed `alpha` to switch the search off.

The models do not report standard errors. All published uncertainty comes from the
walk-forward errors in section 7.

## 7 Uncertainty

The point forecast gets an empirical interval. For each model and lead, the code takes the
quantiles of the walk-forward errors `actual - pred` and adds them to the point forecast.
The quantile set is 0.05, 0.25, 0.50, 0.75 and 0.95.

Two scores measure the intervals: the mean pinball loss over the quantile set, which this
project reports as CRPS, and the share of actuals inside the central 90 percent interval.

The scoring holds out 1 year at a time. Each cutoff takes its interval from the errors of
other years. A cutoff late in year Y still shares target months with cutoffs early in year
Y plus 1, so scores at long leads are slightly optimistic.

## 8 Validation

The harness is walk-forward cross-validation in `src/forecasting/cross_validate.py`.

- Cutoffs: every month end in the last 15 years that has 24 months of actuals after it. This
  gives 157 cutoffs, from 2011-08 to 2024-08.
- At each cutoff the harness trains every model on data from 1960 through the cutoff, then
  predicts 24 months.
- The harness records MAE and RMSE per model and lead, MAE relative to `naive_last`, CRPS
  and 90 percent coverage, and the 2 headline scalars by issue month.

Two caveats apply to every number this harness produces.

1. The harness reads today's data. USGS revises provisional elevation and discharge, and the
   SNOTEL roster grew. A forecast issued in 2013 did not have these values. The live record
   in `forecasts/` is the only vintage-correct score.
2. Overlapping target months make long-lead interval scores slightly optimistic. See section
   7.

## 9 Current skill

Every published number comes from one cross-validation run: experiment 10 in
`forecast_experiments.db`, 157 cutoffs from 2011-08 to 2024-08, data through 2026-08.

See the README section "Current results" for the MAE table by horizon, the headline table by
issue date, and the comparison against the published NRCS record. In short:

- `blend` is the best model at lead 1 and at leads 19 to 22, and it matches `swe_regression`
  in between. It is the model the page shows.
- `swe_head` is the best model for the spring peak from a January or February issue. Its
  error at lead 24 is 2.32 ft, against 1.89 ft for `blend`.
- Past lead 23 no model beats persistence.
- The 90 percent interval covers 0.87 to 0.89 of the actuals at leads 6 and 12.

## 10 Open questions

1. The forecast path is 24 independent fits. Nothing forces the path to be smooth or the
   intervals to widen with the lead. A path-level constraint may help at long leads.
2. Stage 2 of `inflow_chain` fits on observed inflow and runs on predicted inflow. An
   errors-in-variables correction, or a joint fit, may remove the bias in section 4.
3. Reservoir storage lowers skill as a plain extra regressor. Storage moves with the level
   and with the same trend, so it adds a collinear column rather than new information. A
   deficit measure against capacity may separate the 2 signals.
4. The percent-of-median snowpack columns stop in June. A season-aware feature set, which
   uses snowpack from October to May and soil moisture or year-to-date inflow from June to
   September, is the open experiment.
5. `swe_head` beats `swe_regression` at leads 5 to 12 and loses badly past lead 15. The blend
   uses `swe_regression` as its snowpack component. A 3-way blend, or `swe_head` in that
   place, may take the middle leads.
6. No model uses the nClimDiv columns. They need the lagged copy from section 2 first.
7. The forecast issued 2026-09-01 has no `.meta.json` vintage sidecar. The export predates
   that feature. The record stays append-only, so the file is not backfilled.
