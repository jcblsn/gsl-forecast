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
| `elevation_eom_ft` | USGS 10010000, under the configured endpoint rule | 1847-10 | 2026-08 | None; provisional same day |
| `swe_eom_gsl`, `prec_wy_eom_gsl` | NRCS SNOTEL | 1978-10 | 2026-08 | None; daily values post next day |
| `swe_pct_median_gsl` | NRCS SNOTEL | 1978-10 | 2026-05 | None, but October to May only |
| `prec_pct_median_gsl` | NRCS SNOTEL | 1978-10 | 2026-08 | None |
| `sms_eom_gsl` | NRCS SNOTEL | 1999-11 | 2026-08 | None |
| `inflow_kaf_total` | USGS 10126000, 10141000, 10170490 | 1949-10 | 2026-08 | None; provisional same day |
| `inflow_kaf_lake` | The same 3 gauges, divided by `delivery_ratio` | 1949-10 | 2026-08 | None |
| `inflow_kaf_reported`, `n_inflow_gauges` | The same 3 gauges, partial sums allowed | 1907-10 | 2026-08 | None |
| `breach_kaf` | USGS 10010020 | 2008-10 | 2026-08 | None |
| `north_arm_ft`, `head_diff_ft` | USGS 10010100 | 1966-04 | 2026-08 | None |
| `res_kaf_total` | NRCS AWDB, 13 roster Reclamation stations | 1911-01 | 2026-08 | A few days |
| `tmax_f_kslc`, `tmin_f_kslc`, `wind_mps_kslc`, `prcp_in_kslc` | NOAA NCEI, GHCN-D USW00024127 | 1948-01 | 2026-08 | None; a day posts the next day |
| `salt_mass_mt`, `salt_mass_age_days` | UGS brine chemistry, site AS2 | 1966-06 | 2026-08 | Carried forward between campaigns |
| `salinity_gl_lag1` | `salt_mass_mt` over the volume of the month before | 1966-07 | 2026-08 | None; the column is already lagged |
| `tavg_f_gsl_lag1`, `prcp_in_gsl_lag1` | NOAA nClimDiv | 1895-02 | 2026-08 | None; the column is already shifted 1 month |
| `nrcs_inflow_forecasts` | NRCS AWDB forecast point | 2024-01 | 2026-05 | None; January to May only |

Four availability rules control which model may use which column.

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
4. Salinity is dissolved salt divided by volume, and the lake level is a function of volume.
   This month's salinity is therefore partly this month's level: the 2 correlate at -0.68 in
   this record. A model that read salinity to predict level would be predicting the level
   partly from itself. Only `salinity_gl_lag1` reaches `monthly_covariates`, and
   `UNAVAILABLE_AT_ISSUE` names `salinity_gl`. `water_balance` works its salinity out from
   the volume it is already tracking, not from the table.

   The KSLC weather columns are the opposite case and carry no restriction. NCEI publishes a
   day's weather about a day later, so the cutoff month is complete when the workflow runs on
   the 2nd. A model can use measured weather for the cutoff month, which the nClimDiv columns
   cannot supply.

### 2.0.1 The SNOTEL roster

The snow features come from a versioned roster, not from the sites AWDB reports as active on
the day of the run. `config/config.json` names the roster under `covariates.snotel.roster`.
The roster in use is `gsl-modern-complete-v1`: the 29 sites with a month-end SWE value in
every month from 1989-10 to 2026-08. `snotel_roster` holds one row per site with the roster
version, the basin and the basin weight, and `monthly_covariates` carries the version in
`snotel_roster_version`.

A discovered roster changes with the AWDB active flag and with the sites an earlier run left
in `snotel_sites`, so the same code gave different features on a fresh database and on an old
one. A fixed roster removes both effects. It also removes the drift of a raw mean over a site
count that grows from 18 sites in 1979 to 55 sites in 2026. The pipeline still ingests every
discovered site, so a later roster version can use a site this one leaves out.

A site's month-end value is its last valid value in the last 5 days of the month, not its
value on the last day. Exact last-day matching dropped a site whose last day was missing even
when the day before was present. Each variable takes its own last valid day and its own count
of reporting sites, so the count of reporting SWE sites no longer weights the precipitation
and soil-moisture averages. Every pooled (`_gsl`) column averages the basins under the
roster's declared basin weights.

#### What the fixed roster costs

The fix costs accuracy on the development cohort. Against the discovered roster, `swe_head`
MAE rises by 0.002, 0.020, 0.034, 0.034, 0.024 and 0.024 ft at leads 1, 3, 6, 12, 18 and 24,
and `swe_regression` by 0.002, 0.013, 0.026, 0.040, 0.023 and 0.049 ft.

The change is kept anyway, for 3 reasons. The cost is well inside the block-bootstrap
interval of section 8.1, which spans about 0.27 ft at lead 6. The development cohort has
guided many earlier decisions, so a 0.03 ft edge on it is weak evidence. And the discovered
index counts 30 sites in 1990 and 55 in 2026, so its composition changes through the record
and is densest in the recent years the cohort scores most.

The reservoir roster is still discovered. It grows as dams are built, so early storage sums
are smaller for a physical reason.

### 2.0.2 Discharge day coverage

A month of discharge needs at least 25 daily values. The sum of those values is then scaled
to the whole month, which assumes the missing days flowed like the days that reported. Before
this rule a 28-day sum was published as a 31-day volume. `inflow_day_coverage` records the
lowest share of calendar days any inflow gauge reported that month, so a reader can see how
much of a value the scaling supplied. In the current database the lowest share among months
that pass the threshold is 28 of 31 days, so the rule corrects a latent defect and not the
published scores.

USGS records an approval status and a qualifier with every daily value. Those fields were
stored and then ignored. `inflow_provisional_days`, `inflow_estimated_days` and
`inflow_ice_days` count the days behind each month's inflow that USGS marks provisional,
estimated, or affected by ice, so a model or a reader can tell an approved month from a month
USGS will revise. Ice matters for the winter Bear River record and it reaches the balance
directly.

`data_status` now refuses to issue a forecast when `inflow_day_coverage` is below 1, so a
scaled month can no longer reach a forecast unannounced.

There is deliberately no check on how much of the month is still provisional. USGS approves a
month long after it ends, so the cutoff month is 100% provisional at every single issue. A
check on it would fire every month, and a warning that always fires is one nobody reads. The
issue metadata records the share instead, where it describes the data without blocking it.

### 2.0.3 Two approval vocabularies

USGS labels its data 2 different ways in the record this project holds. Rows before 2025 use
the single letters `A` and `P`. Later rows use the words `Approved` and `Provisional`. The
code searched for the word and so missed the letter. It counted 0 provisional days for
January to May 2025; the correct count is 122.

`src/pipeline/quality.py` now holds one definition of each label. New rows are normalised as
they arrive, and the readers accept both forms, because the pipeline never revisits a row
once it falls outside the 45-day refetch window.

### 2.0.4 The reservoir roster

`res_kaf_total` used to add up whichever reservoirs AWDB called active on the day the
pipeline ran, plus any an earlier run had left behind in `reservoir_sites`. The station count
moved from 1 to 21 to 19 over the record. The column was therefore measuring a different set
of reservoirs in every era, and even between 2 runs on consecutive days.

A roster now fixes the set at the 13 Reclamation stations that report in every month from
1989-10. It follows the same rule and refuses the same 4 conditions as the SNOTEL roster.

### 2.0.5 The end-of-month state

`avg_elevation` averages the days of the month. A volume calculation needs 2 instants, not 2
averages, so `elevation_eom_ft` carries a month-end value and `sources.endpoint_rule` chooses
which one.

Over 1989 to 2026, more smoothing always makes the water add up better. The leftover is 0.136
ft per month using the last daily reading, 0.129 using the median of the last 3 days, and
0.121 using the median of the last 7 days.

Forecast accuracy does not follow. Lead-1 MAE is 0.094, 0.093 and 0.103 ft for the same 3
rules, so the last reading and the 3-day median are indistinguishable, and the 7-day median is
worse. Smoothing helps the arithmetic and eventually hurts the forecast, because it moves the
starting point away from where the lake actually was.

The default is `median_3d`. It matches the best forecast, and a single reading taken in high
wind cannot move it.

## 2.1 The endpoint seasonal baseline

`endpoint_seasonal` is the strong state-only baseline. Within each fit it compares the last
daily elevation, the median of the final 3 calendar days, and the median of the final 7 calendar
days. Candidate comparison uses expanding one-step errors from targets already observed inside
that fit. For each lead, the forecast is the selected current endpoint plus the historical
median endpoint-to-target change among origins in the same calendar month. The monthly mean
remains the target; the endpoint is only the initial state. If endpoint fields are unavailable,
the latest monthly mean provides an explicit compatibility fallback.

`endpoint_analog` is the same model with `n_analogs` set to 8. It takes the median over the 8
past origins whose own level was closest to the level now, instead of over every past origin
in the same calendar month. The change from a 4,190 ft origin is not the change from a 4,200
ft origin: the surface area differs, so the same volume moves the level by a different amount,
and the lake reverts toward its own long-run level. This is the level-conditioned
seasonal-change climatology the review asks for as a stronger baseline than `naive_last`.

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
- Each fit uses 1 row per year. The rows do not overlap in time when `h` is 12 or less. Past
  12 months 2 adjacent origin-to-target windows share months, so the rows do overlap. Lake
  endpoints are also serially dependent even when the windows do not overlap. This affects
  the uncertainty and the stability of the penalty search, not the mechanical validity of
  the point prediction, and it is why a reported difference needs the block bootstrap of
  section 8.1.
- The fit is per calendar month, so the model needs no seasonal term.
- The `b1 * y_i` term is a mean-reversion term. It sets how far the lake returns toward its
  own level over `h` months.

The effective sample is small. The training era starts in 1989-10, so 1 fit reads about 1 row
per year against 4 parameters. This is the main reason the review calls the model too
fragmented; a pooled model across issue months and leads is future work.

The model drops 1 feature at a time under a declared rule. It drops a feature when:

- The feature is NULL at the cutoff.
- Fewer than `min_obs` rows (default 10) carry the feature.
- The standard deviation of the feature among those rows is 1% or less of its standard
  deviation over the whole training frame.

The last rule holds because a feature depends on the issue season. Snow water equivalent is
structurally 0 at an August cutoff. The standardised ridge divides by that near-zero standard
deviation and returns a coefficient of hundreds of feet per inch. The forecast contribution
stays small because the input is near 0, but the coefficient is a diagnostic failure.

The model then fits on the rows that carry every feature it kept. If fewer than `min_obs`
rows do, it drops every feature and fits `y_(i+h) - y_i = b0 + b1 * y_i`. Before this rule a
single missing feature dropped all of them.

10 rows against 4 or 5 parameters is thin, and the review calls the bar too permissive.
Raising it was measured and reverted. The bar never binds on an outer fit: `swe_head` scores
0.127, 0.326, 0.555, 1.035, 1.509 and 2.006 ft at leads 1, 3, 6, 12, 18 and 24 with the bar
at 10, at 15 and at 20. Where it does bind, on the early inner cutoffs of the blend's weight
pass, it degrades the weights rather than protecting a fit: the `blend` lead-6 MAE rises from
0.578 to 0.595 to 0.626 ft. The real repair is a pooled model across issue months and leads,
not a higher bar on 288 separate fits.

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

## 4.2 The water_balance model

Every other model here predicts a lake level from earlier lake levels. This model counts the
water instead. Each month it adds the water that arrives and subtracts the water that leaves.
The result is a volume, and the hypsometry table turns that volume into a level.

The difference matters because the other models cannot separate the causes. One coefficient
in `inflow_chain` holds evaporation, rain on the lake, unmeasured rivers and causeway flow
together in a single number. This model gives each of those its own term.

The model tracks the volume of water in the lake at the end of each month:

```
V[t] = V[t-1] + b_Q * Q[t] + b_P * P[t] * A[t-1] - b_E * E[t] * f(S[t-1]) * A[t-1]
       + a[month] * A[t-1] + R[month] * A[t-1] + c
```

| Symbol | Meaning | Source |
|---|---|---|
| `V` | South-arm storage, kaf | `elevation_eom_ft` through the USGS bathymetry |
| `A` | South-arm area, acres, at the month before | The same bathymetry |
| `Q` | Gauged tributary inflow, kaf | `inflow_kaf_total` |
| `P` | Precipitation depth on the lake, ft | `prcp_in_kslc` |
| `E` | Hargreaves reference evaporation depth, ft | `tmax_f_kslc`, `tmin_f_kslc` |
| `S` | Salinity, g/L, at the month before | `salt_mass_mt` over `V[t-1]` |
| `f` | Evaporation suppression by salt | `1 - k * S / 1000`, `k` fitted |
| `a` | Season term, ft of depth | Fitted, 11 terms with January as the reference |
| `R` | `net_unmeasured_flux`, ft of depth | The closure residual, pooled by calendar month |

`R` is the water the sum does not account for. It is not evaporation, and it is not water
that people used. It mixes 4 things together: flow through the causeway, error in how much
river water reaches the lake, error in the map of the lake bed, and the part of evaporation
that temperature alone cannot predict. Its name says only that, and claims nothing more.

The area and the salinity both come from the previous month, for the same reason. A larger
lake has more surface to evaporate from, so the area belongs in the sum. But the area is also
a consequence of the volume, which is the answer. Using this month's area would use the answer
to compute the answer.

Salinity has the same problem, and worse. Salinity is dissolved salt divided by volume, and
the level is a function of volume, so this month's salinity is partly this month's level. The
2 measurements correlate at -0.68 in this record. The model therefore works its salinity out
from the volume it is already tracking, and never reads it from the table.
`UNAVAILABLE_AT_ISSUE` lists `salinity_gl` so that no other model can read it either.

### Observation operator

The forecast this project publishes is an average over a month. The volume the model tracks
is a single instant, at the month's end. These are not the same thing. Publishing the
month-end value as if it were the monthly average would shift the whole forecast half a month
late. The model therefore averages the 2 month-end values on either side of the month.

The gap between the 2 is not small. Take the difference between 2 monthly averages and call
it the change in volume, and the error has a standard deviation of 75 kaf. The real month to
month change has a standard deviation of 184 kaf. So 41% of the signal is noise created by
mixing up an average with an instant.

### Forcing beyond the cutoff

Nobody knows next year's weather. The river inflow comes from the same snowpack regression
`inflow_chain` uses. The weather comes from the KSLC average for that calendar month over the
training years. Neither reads anything from after the cutoff. Past about 12 months the
snowpack regression shows no skill at all, so from there the model runs on averages and does
not pretend otherwise.

### What the balance closes to

Fitted over 1989-11 to 2026-08, 442 closed monthly steps:

| Specification | Residual |
|---|---|
| Gauged inflow alone | 143 kaf, 0.304 ft/month |
| Full balance | 58 kaf, 0.129 ft/month |

Fitted terms: evaporation scale 1.09 on Hargreaves, precipitation scale 0.82 on KSLC,
salinity suppression 2.75 (a 36% reduction at 130 g/L), and a gauged-to-delivered inflow
ratio of 1.28.

That last number contradicts the assumption it was meant to test, so it is worth stating
plainly. The configured `delivery_ratio` of 0.8246 says the gauges measure less water than
the lake actually receives, because small streams and groundwater arrive below them. The fit
says the reverse: 1 kaf passing the gauges raises the lake by only about 0.78 kaf. Water lost
in the wetlands and canals between the gauges and the open lake would explain that.

Do not treat the fitted number as a measurement yet. The season terms and the leftover absorb
part of the same signal, so the 3 cannot be cleanly separated. `get_metrics` prints both the
assumption and the fit on every run, so the next round of work can pull them apart.

### Residual stability

`gsl-audit` reports the leftover in kaf, and breaks it down by how full the lake was and by
years the fit never saw. Across 4 bands of lake level its standard deviation is 53 to 69 kaf
and its average is within 7 kaf of 0. Across 3 blocks of held-out years it is 0.114, 0.124
and 0.164 ft per month. The block from 2015 to 2026 is the worst; the lake is at its lowest
in those years. A leftover that stayed small only in the years it was fitted on would be a
fitted constant, not a measurement.

### Accuracy

These numbers come from the development split and its 157 cutoffs. At lead 1 `water_balance`
beats `swe_head` by 0.024 ft, and the bootstrap interval of [0.009, 0.039] does not contain
0, so that gain is real. At every other lead the interval does contain 0, which means the 2
models cannot be told apart there. The registry scores the model and `PRODUCTION_MODELS`
leaves it out, under the same rule that leaves out `state_space`.

The salt term pays for itself at long leads. Against `water_balance_nosalt` it is worth 0.067
ft at lead 18 and 0.106 ft at lead 24, and neither interval contains 0. That is the return on
the UGS brine record.

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

The band is conditional on the issue season. The errors are strongly heteroskedastic by
season: 1 band over every issue month gave the `blend` a coverage of 0.82 at lead 6 from an
accumulation issue and 0.98 from a recession issue, both with the same width of 2.31 ft.
Aggregate coverage of 0.87 hid both.

A season cell holds about 13 to 22 errors at 1 lead. That is enough to estimate a centre and
a width, and too few to read a 5% or a 95% quantile from. Therefore the centre and the width
are conditional on the season and the shape of the tail is pooled:

1. The centre of a season is the median of its errors, and the width is their mean absolute
   deviation from that median.
2. Each is pulled toward its pooled value by `n / (n + 10)`, where `n` counts the errors in
   the cell. A season with few errors keeps close to the pooled value.
3. Every error is divided by the centre and the width of its own season, and the quantiles
   of those standardised errors are pooled over the seasons.
4. A season's band is its centre plus its width multiplied by that pooled shape.

Standardising before pooling matters. Pooling the raw errors would give every season the
tail of whichever season has the widest errors, so a band scaled down for a narrow season
would keep a skew the narrow season does not have.

At lead 6 the `blend` band is now 3.36 ft wide from an accumulation issue and 1.54 ft from a
recession issue, with coverages of 0.89 and 0.93. The rule moves width to where the errors
are. It does not manufacture information: the accumulation season is genuinely harder, and
its band is genuinely wider.

Three scores measure the intervals: the weighted interval score, the unweighted mean pinball
loss over the quantile set, and the share of actuals inside the nominal central 90 percent
interval.

The weighted interval score is the recognized finite-quantile approximation to the continuous
ranked probability score. It adds the absolute error of the median to the interval score of
each central interval, each weighted by its own alpha, and divides by the number of terms.
For a symmetric quantile set such as this one it is exactly twice the unweighted mean pinball
loss, so it adds no information. It is reported because it carries a recognized name and a
recognized definition, which the mean of 5 pinball losses does not. Neither score is an
integral over the full forecast distribution.

`gsl-cv`
prints coverage and width per issue season and writes them to
`outputs/season_coverage_<stamp>.parquet`, because an aggregate coverage near 0.90 hides a
season at 0.82 and a season at 0.98.

The displayed point forecast is fitted independently from this retrospective calibration.
It is not necessarily the q50 interval value.

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

### 8.1 How precise these numbers are

The 157 cutoffs overlap heavily. A 24-month forecast from one cutoff shares 23 of its target
months with the next cutoff, so the cohort holds about 13 hydrologic years and not 157
independent cases. A 3-decimal rank table therefore overstates what the record settles.

`src/forecasting/bootstrap.py` resamples the cutoff sequence with a circular moving block of
24 months, which is the horizon and therefore the span over which 2 cutoffs can share a
target month. `gsl-cv` prints the MAE of the headline model with a 95% interval, and its
paired improvement over `naive_last` on the same resampled cutoffs. It writes both to
`outputs/mae_intervals_<stamp>.parquet` and `outputs/improvements_<stamp>.parquet`.

For the frozen development run the `blend` MAE intervals are:

| Lead | MAE | 95% interval | Improvement over `naive_last` |
|---:|---:|---|---|
| 1 | 0.124 | 0.102-0.148 | +0.210 [+0.170, +0.253] |
| 3 | 0.330 | 0.279-0.388 | +0.568 [+0.451, +0.681] |
| 6 | 0.573 | 0.451-0.736 | +0.753 [+0.601, +0.901] |
| 12 | 1.070 | 0.788-1.398 | +0.210 [+0.071, +0.361] |
| 18 | 1.553 | 1.090-2.020 | +0.349 [+0.141, +0.548] |
| 24 | 1.915 | 1.294-2.522 | -0.129 [-0.461, +0.197] |

The improvement excludes 0 to lead 18 and includes it at 24. A difference of a few
hundredths of a foot between 2 models is not evidence of a better model: the interval at
lead 6 spans about 0.29 ft.

These are descriptive sensitivity estimates. They are not formal sampling intervals under a
fully specified data-generating process, and they do not prove stationarity.

### 8.2 Caveats on every number

1. The harness reads today's data. USGS revises provisional elevation and discharge. A
   forecast issued in 2013 did not have these values. The live record in `forecasts/` is the
   only vintage-correct score. Each issue's `.meta.json` records the SHA-256 of the whole
   modeling table and its column list, the SHA-256 of the resolved configuration, and the
   SNOTEL roster version, so a later reader can tell whether the table it holds is the table
   the issue used.
2. Overlapping target months make long-lead interval scores slightly optimistic. See section
   7.

## 9 Frozen development accuracy

The maintained retrospective tables come from one cross-validation run,
`GSL_CV_20260903_1751`: 157 cutoffs from 2011-08 to 2024-08, training from 1989-10, data
through 2026-08. `data/results/` holds a snapshot of that run. Its manifest records the
development-only status, limitations and hashes, and CI verifies those hashes. The README
section "Frozen development results" holds the tables, and `gsl-results --tables` prints them
from the snapshot. This repeatedly used record is not untouched test evidence.
`docs/autoresearch.log` is a historical experiment log.

The experiment tracker database is the working file for a run in progress. `.gitignore`
excludes it, so the snapshot rather than an experiment id is the citation.

In short:

- The state-only baselines are strong. `endpoint_seasonal` is the best model at lead 1 at
  0.10 ft, at lead 24 at 1.70 ft against 1.79 ft for a repeat of the last value, and on the
  water-year-end target from June, July and August issues. It reads no snowpack and no
  streamflow. `endpoint_analog`, which conditions the same change on the current level, does
  not improve on it here.
- `swe_head` is the best model from lead 3 to lead 17, and the best model for the maximum
  April–June monthly mean from a January issue, at 0.74 ft against 1.62 ft for a repeat of
  the last value.
- `blend` is still the prototype headline and no longer wins anywhere outright. `swe_head`
  matches or beats it at every lead to 18, and `endpoint_seasonal` beats both at 24. Its
  paired improvement over `naive_last` excludes no improvement at leads 1, 3, 6, 12 and 18,
  and includes it at 24. Removing blend degrees of freedom unless they win a locked test
  remains open work.
- The 90 percent interval covers 0.87 of the actual values at lead 6 and 0.87 at lead 12 in
  aggregate. Section 7 gives the coverage per issue season, which is what a decision needs.

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
