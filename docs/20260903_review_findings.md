# Quantitative review findings

Date: 2026-09-03  
Status: pre-implementation review complete; no model or pipeline changes have been made  
Scope: target construction, data ingestion, covariates, model specifications, uncertainty,
validation, and the usefulness of the published product

## Executive conclusion

The repository contains a real and useful forecasting signal, but the current headline should
not be described as a strong 24-month forecast and the `state_space` model should not be
described as a proper water-balance state-space model.

The strongest result is the near-term one: the covariate models reduce 1-, 3-, and 6-month MAE
from 0.34, 0.90, and 1.33 ft for a last-value baseline to about 0.13, 0.32, and 0.52 ft. That is
meaningful skill. At 12 months the advantage is modest; at 24 months the published blend is
worse than the last-value baseline (2.00 versus 1.79 ft). Uncertainty around these estimates is
large because 157 overlapping monthly cutoffs represent only about 13 independent water years.

There are two distinct limitations, and both matter:

1. **Internal specification and evaluation problems.** The target changes measurement character
   over the training record; the models discard the latest end-of-month state; the physical
   models operate in elevation rather than storage; the state-space model conditions on point
   forecasts of future inflow and omits their uncertainty; the direct model has hundreds of
   small, weakly pooled regressions; the uncertainty layer is marginal rather than trajectory-
   based; and repeated experimentation has used the same small evaluation period.
2. **Unknown future forcing.** Beyond roughly one snow season, current SWE and precipitation-to-
   date contain little information about the weather, runoff, evaporation, withdrawals, releases,
   and causeway operations that will determine the lake balance. A single unconditional path
   silently substitutes average future conditions. At horizons people care about, explicit
   forcing and management scenarios are not optional decoration; they are part of the estimand.

An oracle experiment separates these limitations. Starting the training record in the modern
daily-measurement era and replacing forecast tributary inflow with the subsequently observed
inflow lowers `state_space` MAE from 1.85 to 0.96 ft at lead 24. Thus roughly half of its remaining
long-lead error is associated with not knowing future inflow, and roughly half remains after
inflow is known. Better external forcing is necessary for long leads, but adding data to the
current elevation-ARX equation will not be sufficient.

The pragmatic redesign is likely smaller than the current model suite:

- a compact, aggressively validated statistical forecast for the next 1--9 or 12 months and for
  direct decision targets such as spring peak;
- a storage-based stochastic water balance for longer horizons, driven by coherent ensembles of
  future weather/runoff and explicit management/use scenarios;
- one common set of coherent trajectory ensembles from which monthly quantiles, peak/low
  distributions, and threshold probabilities are derived.

## What is already sound

This is not a finding that the repository is careless throughout. Several choices provide a
good base for the next version:

- Forecast evaluation refits at historical cutoffs, and the blend's weights are estimated only
  from inner cutoffs available to each outer fit. The temporal intent is correct even though the
  available historical vintages and repeated outer-period model selection limit the result.
- The current calendar month is excluded from the target aggregation, and production refuses a
  stale or thin latest month unless degradation is explicitly allowed
  ([`elt.py`](../src/pipeline/elt.py#L14-L27),
  [`run_forecast.py`](../src/forecasting/run_forecast.py#L35-L57)).
- The USGS ingest stores approval/qualifier information and refetches recent provisional data.
  The present weakness is that those fields are not used in modeling or historical vintage
  reconstruction, not that they were discarded.
- The south-arm gauge and local hypsometry are both expressed in NGVD29 feet. The source table
  also retains NAVD88, so no datum mismatch was found in the current area/volume lookup
  ([`hypsometry.py`](../src/forecasting/hypsometry.py#L1-L35)).
- Climate inputs that would not be complete at an issue date are lagged. Model fallbacks are
  declared and logged rather than arising accidentally from implicit missing-value behavior.
- The repository preserves dated forecast artifacts and already distinguishes live issued
  verification from retrospective CV. This will make genuine prequential evaluation possible as
  the issued record grows.
- The state transition's one-month statsmodels intercept alignment, a common source of subtle
  errors, appears correct after the existing fix.

The important distinction is between software correctness and statistical adequacy. The test
suite and temporal plumbing are better than the model's present scientific interpretation.

## What the forecast should mean

The model currently attempts to answer one question with one path: “what will monthly mean lake
elevation be for each of the next 24 months?” In reality there are three different prediction
problems:

| Horizon | What is substantially known | Honest product |
|---|---|---|
| 1--3 months | Current lake state, much of the seasonal hydrograph, short-range weather | A calibrated operational forecast |
| 3--9/12 months | Current snow, soil, reservoir, and water-year state; much future weather is unknown | A hydrologic ensemble with meaningful conditional probabilities |
| 9--24+ months | The next snow season, evaporative demand, operations, use, and policy are largely unknown | A declared mixture of weather and management scenarios, not a single unexplained path |

At the longer horizons, the object is more accurately written as

```text
p(lake trajectory | state at issue,
                     future meteorology/runoff ensemble,
                     consumptive-use and reservoir scenario,
                     causeway configuration)
```

There can still be an unconditional “most likely” distribution, but only after declaring the
scenario weights used to mix those conditional distributions. This prevents the current hidden
assumption in which unknown future predictors are effectively replaced by historical regression
averages.

Reliability and sharpness must be kept separate. At 18--24 months, a forecast can be excellent by
being calibrated, beating an ensemble-climatology distribution on a proper score, and correctly
quantifying action-relevant risks while still being wide. It cannot be both reliably narrow and
agnostic about the next winter's weather. Conditional scenarios can be much sharper because they
answer a different, explicitly conditional question.

## Findings by severity

### Critical: the target is not homogeneous over the training period

[`src/pipeline/elt.py`](../src/pipeline/elt.py#L14-L27) takes the unweighted average of whatever
daily records exist in a calendar month. The resulting series is called a monthly mean regardless
of whether a month contains one observation, two observations, or every day.

The local database on 2026-09-03 shows:

| Period | Typical observations per month |
|---|---:|
| Before 1960 | 1 |
| 1960s--1970s | 2 |
| 1980s | usually 2, then daily late in the decade |
| 1990 onward | 28--31 |

Of the 800 months from 1960-01 through 2026-08, 357 contain fewer than 25 observations. Continuous
near-daily coverage begins in 1989-10. The USGS historical report is consistent with this break:
it reports mean daily elevations beginning in October 1989 and notes that wind can produce
substantial short-term surface changes ([USGS Great Salt Lake records](https://pubs.usgs.gov/wdr/2004/wdr-ut-04/PDF/6Great_Salt_Lake.pdf)).

As a diagnostic, applying the old first-and-fifteenth sampling pattern to the modern daily record
and comparing it with the actual monthly mean produces an MAE of 0.085 ft and RMSE of 0.106 ft,
with a maximum absolute discrepancy of 0.383 ft. The error is seasonal: the sparse sample tends
to be low in winter and high in summer. This is material beside the reported lead-1 model MAE of
0.126 ft.

This does not mean the pre-1990 record is useless. It means it should not enter the likelihood as
if it had the same observation operator and precision as the modern record. Defensible choices are:

- train the operational model on the homogeneous record beginning in 1989-10;
- reconstruct a consistent twice-monthly estimand across the full record; or
- represent the actual observation dates and sampling uncertainty in a measurement model.

The first option is the simplest and should be the default benchmark. The longer record can then
be used as a sensitivity analysis or to inform strongly pooled climatological priors.

### Critical: the stated headline targets do not quite match their real-world concepts

The “spring peak” is implemented as the maximum of April--June monthly means in
[`src/forecasting/headline.py`](../src/forecasting/headline.py#L12-L13). From 1990 through 2026,
the actual daily spring maximum exceeds that monthly-mean maximum by 0.198 ft on average (median
0.180 ft; 90th percentile 0.293 ft; maximum 0.364 ft). This estimand difference is larger than the
reported 0.122-ft May-issue peak MAE. The repository does disclose the difference when comparing
with NRCS, but the public target should be named “peak April--June monthly mean” unless it is
changed to the daily peak.

The fixed window is a smaller but real distinction. In 6 of the 36 complete calendar years from
1990--2025, the highest January--September monthly mean occurred outside April--June (five times
in March and once in July), although the April--June proxy missed it by no more than 0.069 ft.
Daily maxima occurred outside April--June in 8 of those 36 years. The operational target should
therefore specify both its temporal window and whether “peak” means daily or monthly mean.

Likewise, September is a useful administrative water-year-end target, but it is not reliably the
annual low as claimed in [`README.md`](../README.md#L13-L18). Across water years 1991--2025,
September was the minimum monthly mean in 20 of 35 years; October was the minimum in 10 and
November in 4, with January accounting for the remaining year. Call it “September mean” or
“water-year-end elevation.” If users care about the seasonal low, forecast the minimum and its
date directly from coherent trajectories.

These are not semantic niceties. A forecast can have excellent error against the wrong target and
still disappoint its users.

### High: the operational models discard the latest observed lake state

Every model initializes from the cutoff month's **average** elevation. At an issue on the first
or second of the next month, the end-of-month daily level is also observed and is a substantially
better state estimate for the start of the forecast. Across the 157 evaluation cutoffs, the last
daily level minus the monthly average has a standard deviation of 0.218 ft and ranges from -0.500
to +0.757 ft. That is not negligible relative to a 0.126-ft lead-1 MAE.

A leakage-safe expanding diagnostic, trained only on same-calendar-month origins whose targets
were already observed, forecast the next monthly mean as the latest daily elevation plus the
historical median seasonal change. Its lead-1 MAE is 0.104 ft, compared with 0.126 for the blend,
despite using no snow or streamflow. Its MAE is also 1.69 ft at lead 24, compared with 2.00 for the
blend and 1.79 for last-value persistence. Its retrospective lead-1 central-90% residual width is
0.43 ft, versus 0.55 ft for the blend. Paired block-bootstrap intervals do include no improvement
at leads 1 and 24, so this is a strong missing baseline rather than a declared winner.

An exploratory endpoint-anchored version of the existing `swe_head` regression, trained from the
modern era and with near-zero columns removed, produced MAEs of 0.092, 0.292, 0.532, 0.760, 1.005,
1.487, and 1.969 ft at leads 1, 3, 6, 9, 12, 18, and 24. It was examined on the same development
period and therefore requires a locked test; it is not a publishable performance claim. It does
show that the timing/state representation is a plausible, low-complexity route to immediate
near-term improvement.

The practical change is to retain both concepts: end-of-month elevation as the initial state and
monthly mean as the forecast target. The input's availability, provisional status, and last-valid-
day rule should be frozen in each issue vintage. Because wind and rounding affect a single daily
value, compare the last observation with a robust 3- or 7-day endpoint estimate inside the
training folds rather than assuming the literal last day is optimal.

### Critical: `state_space` is an elevation ARX model, not a closed water balance

The implemented transition is

```text
m[t] = phi * m[t-1] + a[month] + b * Q[t] + process error
y[t] = m[t] + measurement error
```

as shown in [`state_space.py`](../src/forecasting/multivariate/state_space.py#L71-L125). This is a
perfectly recognizable linear Gaussian state-space specification, but its state is elevation. A
lake conservation equation closes in volume, not elevation. The conversion from a unit of water
to feet varies with elevation because surface area changes materially. The USGS elevation-area-
volume relation is already in this repository and an authoritative current table is available
([USGS elevation-area-volume data](https://data.usgs.gov/datacatalog/data/USGS%3A6467b42fd34ec11ae4a8afb1)).

The fitted constant `b` therefore cannot have the physical interpretation the documentation gives
it. In the current fit it is approximately 0.0014 ft/kaf. Local hypsometry implies a response of
roughly 0.0027 ft/kaf near 4190 ft, 0.0020 near 4198 ft, and 0.0016 near 4210 ft before evaporation,
exchange, or other fluxes. The fitted coefficient is absorbing omitted losses, omitted inflows,
nonlinear geometry, and serial dynamics.

There is also a time-support mismatch. `avg_elevation[t]` is an average over month `t`, while
`Q[t]` is the volume accumulated during that calendar month. A conservation step naturally
relates storage at the end of one interval to storage at the end of the next. The change between
two monthly-average states is centered roughly one month apart and, in a simple approximation,
is affected by portions of both adjacent months' fluxes—not only all of the destination month's
flow. The existing one-index alignment is correct for the equation as coded, but the equation's
physical timing remains wrong. A storage model should use an end-of-month state (with a declared
last-valid-day rule), integrate submonthly fluxes, or explicitly formulate a centered observation
operator for monthly means.

The transition omits direct precipitation on the lake, evaporation, groundwater, minor and
ungauged inflows, consumptive use/diversions, reservoir operations, salinity effects, and
north/south-arm exchange. Monthly intercepts and `phi` proxy for all of them. A classical USGS
monthly Great Salt Lake budget explicitly includes surface inflow, groundwater inflow, direct
precipitation, and evaporation ([USGS water-budget model](https://pubs.usgs.gov/publication/ofr79258)).
A modern probabilistic lake-budget example separately represents component uncertainty, process
error, and level measurement error and closes the balance over suitable windows
([NOAA-hosted water-balance paper](https://repository.library.noaa.gov/view/noaa/25958/noaa_25958_DS1.pdf)).

The better core equation is

```text
V[t+1] = V[t]
         + surface_inflow_at_lake_boundary[t]
         + groundwater[t]
         + precipitation[t] * area(V[t])
         - evaporation[t] * area(V[t], salinity)
         + north_south_exchange[t]
         - direct_lake_outflow_or_withdrawal[t]
         + process_error[t]

elevation[t] = hypsometry_inverse(V[t]) + small_measurement_error[t]
```

The accounting boundary must be explicit. Downstream gauged flows already reflect much upstream
consumptive use and reservoir operation. A scenario model may (a) forecast net flow at the lake
boundary, or (b) begin with naturalized runoff and subtract diversions/depletions and route
reservoir operations. It must not use net gauged flow and subtract the same upstream use again.
Use/conservation scenarios should be translated consistently into lake-boundary flow, with
wetland, canal, and reach losses treated on the correct side of that boundary.

If the causeway is operationally important, north and south storage should be coupled states. A
simpler first implementation can hold the north arm/causeway scenario fixed and forecast the
south-arm balance conditionally.

### Critical: the state-space implementation is not joint with future inflow

The documentation says placing inflow in the state equation removes the observed-versus-predicted
substitution problem. It does not. Historical `b` is fitted on observed tributary inflow, while
the forward recursion inserts one deterministic point forecast from `InflowChainForecaster`
([`state_space.py`](../src/forecasting/multivariate/state_space.py#L161-L200)). The model does not
carry a latent or jointly forecast inflow state, nor does it integrate over future inflow
uncertainty. It changes where the substitution occurs, not whether it occurs.

Missing historical inflow is replaced by a calendar-month mean and then treated as observed
without error ([`state_space.py`](../src/forecasting/multivariate/state_space.py#L151-L159)). This
shrinks variability and can bias both the inflow coefficient and variance decomposition.

The model's analytic predictive variance contains only fitted level process variance and
measurement variance. It excludes future inflow uncertainty, parameter uncertainty, future
meteorology, management regimes, and correlations among forcings. Moreover, these analytic
intervals are not the production intervals; [`predict_quantiles`](../src/forecasting/multivariate/state_space.py#L217-L239)
is bypassed by the empirical residual interval layer.

The practical consequence is severe underdispersion. For the current fit, the model's own central
90% interval widths are approximately 0.48, 0.82, 1.13, 1.55, 1.84, and 2.05 ft at leads 1, 3, 6,
12, 18, and 24. Historical state-space errors require empirical widths of approximately 0.61,
1.39, 2.35, 3.58, 5.23, and 6.03 ft.

### High: the fitted state-space observation layer is degenerate and inference diagnostics are broken

Across inspected cutoffs, the fitted measurement standard deviation rounds to 0.0000 ft; in the
current fit, measurement variance is approximately `2e-10`. The supposedly latent level is thus,
for practical purposes, the observed elevation. That is not intrinsically illegal, but it means
the Kalman layer is not doing the job used to motivate the model.

There is also an implementation-level numerical problem. `_sigmoid` and the variance assignments
cast complex values to Python `float` inside `update`
([`state_space.py`](../src/forecasting/multivariate/state_space.py#L55-L56),
[`state_space.py`](../src/forecasting/multivariate/state_space.py#L105-L125)). Statsmodels' complex-
step numerical differentiation consequently emits “casting complex values to real discards the
imaginary part” warnings. Those warnings are suppressed during fit
([`state_space.py`](../src/forecasting/multivariate/state_space.py#L173-L180)). Standard errors for
several parameters collapse to signed zero and the covariance matrix is singular in inspected
fits. A nonconverged optimization is logged only at debug level and still yields a forecast.

The Gaussian innovation assumptions also fail on the real fit. Statsmodels diagnostics on the
standardized one-step errors give Ljung--Box statistics of about 119 at lag 12 and 143 at lag 24
(both p-values below `1e-18`), and a Jarque--Bera statistic of about 281 (`p < 1e-60`, skew 0.58,
kurtosis 5.66). In other words, the transition has not whitened the dynamics and the innovations
are heavy-tailed. Even if the derivative bug were repaired, iid Gaussian process noise would not
support the advertised analytic intervals without a richer residual structure or robust
calibration.

The state/intercept alignment fix appears correct. The problem is not an off-by-one error; it is
the estimand, forcing, variance, identifiability, and numerical differentiation.

### High: the result is sensitive to the arbitrary training start

Changing only the state-space training start among 1960, 1978, 1989, 1995, and 2000 moves the
current lead-24 forecast across a range of about 1.15 ft. The fitted inflow coefficient ranges
from about 0.00140 to 0.00196 ft/kaf. That is substantial instability relative to the advertised
forecast accuracy.

A full walk-forward diagnostic using 1989-10 rather than 1960 improves state-space MAE by 0.01,
0.03, 0.08, 0.14, 0.20, and 0.28 ft at leads 1, 3, 6, 12, 18, and 24. The 24-month MAE becomes
1.85 rather than 2.13 ft. Comparable long-lead improvements occur in the direct SWE models.

This ablation changes three things together: measurement homogeneity, climatic/management era,
and sample size. It cannot establish which one causes the gain. It does establish that the 1960
start is not innocuous and that current state-space/direct long-lead performance is materially
affected by training composition.

### High: the direct SWE model is too fragmented for the available sample

[`swe_regression.py`](../src/forecasting/multivariate/swe_regression.py#L54-L88) fits a separate
regression for every issue month and lead: 12 × 24 = 288 fits. Each has only about 32--47 annual
observations and typically four or five parameters. Ridge shrinkage helps, but it does not share
information across adjacent issue months or leads, and generalized cross-validation tunes the
penalty inside each small sample.

The claim in [`docs/model-spec.md`](model-spec.md#L85-L97) that rows do not overlap and therefore
need no autocorrelation correction is false for horizons beyond 12 months: adjacent annual
origin-to-target windows share months, and lake endpoints are serially dependent even when the
windows do not overlap. This mainly affects uncertainty and tuning stability, not the mechanical
validity of the point prediction.

Features should also depend on issue season. At an August cutoff, SWE is structurally zero. The
nearly zero historical standard deviation leads to coefficients as large as hundreds of feet per
inch in current fits. The actual forecast contribution remains small because the input is nearly
zero, but the coefficient is an obvious diagnostic failure. Drop zero/near-zero columns, and use
a compact cyclic or hierarchically pooled model rather than 288 independent fits.

The fallback threshold of 10 complete observations is too permissive, and the all-or-nothing
fallback drops every covariate if any one is missing
([`swe_regression.py`](../src/forecasting/multivariate/swe_regression.py#L68-L82)).

`swe_head` also includes both current south-arm elevation and `head_diff_ft`, which is current
south-arm elevation minus north-arm elevation. This is not leakage—both are known at issue—but it
is an algebraically and physically entangled pair. The head term can serve as a useful regime
proxy; it should not be interpreted as the causal effect of causeway management, and its current
value does not specify future berm/opening operations. Public contribution language should retain
that distinction.

Its empirical value is lead-dependent: adding head difference lowers `swe_regression` MAE from
0.546 to 0.512 ft at lead 6, is neutral at lead 12, and raises MAE from 1.56 to 1.72 ft at lead 18
and from 1.95 to 2.32 ft at lead 24. Carrying one feature set through every horizon is therefore
not justified. Either pool a lead-varying effect strongly toward zero or reserve future causeway
effects for an explicit scenario.

### High: the snow and precipitation indices do not have a stable historical meaning

The pipeline discovers stations using `activeOnly=true`
([`covariates.py`](../src/pipeline/covariates.py#L46-L71)). A persistent database retains stations
found on earlier runs because the site table is upserted rather than rebuilt, while a fresh
database contains only stations active today. Thus the feature can depend on database history.

The production SWE and precipitation variables are raw equal-weight site means. In the local
data, the reporting roster grows from 18 sites in 1978 to 55 in 2026. Sites and basins are
implicitly weighted by station count, not drainage area or contribution to inflow. Exact
month-end matching drops a station if the last day is missing even when the prior day is present
([`covariates.py`](../src/pipeline/covariates.py#L327-L352)). The aggregate precipitation and soil-
moisture calculations reuse the count of nonmissing SWE records rather than their own nonmissing
counts.

Use a versioned, fixed site roster where possible; normalize at the site level; then aggregate
with declared basin/runoff weights. If sites begin late, build overlapping-record adjustments or
a hierarchical index. Also test whether the last valid observation within a short availability
window is more faithful to the operational month-end input than exact-last-day matching.

The climate lag is handled conscientiously for a generic issue-date predictor: the cutoff month's
nClimDiv value is unavailable when the workflow runs, so only lagged columns are exposed. It must
not, however, be used as if it were contemporaneous precipitation or evaporation forcing in a
physical balance. A balance model should align historical climate with the month in which the
flux occurred, accept a ragged latest edge, and use an explicit forecast/ensemble for future and
not-yet-released months. The present two-division mean is also a coarse proxy for over-lake flux.

### Medium: completeness and quality flags are not propagated into the modeled data

Monthly discharge is the sum of available daily values and is accepted when at least 25 days are
present ([`covariates.py`](../src/pipeline/covariates.py#L354-L364)). A 25-day sum is not a monthly
volume unless missing days are imputed or the sum is scaled under a justified rule. No 25--27-day
case currently contributes to the three-gauge total in the local database, so this is a latent
correctness problem rather than a cause of the published scores. Validate expected calendar-day
support per gauge and month and either fail, impute with uncertainty, or expose a coverage field.

USGS approval and qualifier fields are stored but do not affect aggregation, training weights, or
vintage metadata. Likewise, SNOTEL aggregate counts describe the SWE-selected roster, not the
nonmissing support for every aggregated variable. Quality, completeness, imputation, and roster
identifiers should travel with each model feature so retrospective fits can reproduce what was
actually knowable.

### High: the recorded “data vintage” is a date, not a reproducible vintage

The monthly job commits forecast CSV/JSON and aggregate CV summaries, but `.gitignore` excludes
`data/gsl.db`, per-cutoff CV parquet files, and the experiment database. The forecast metadata
stores `data_max`, latest observation/site counts, and calibration weights; it does not store the
training table, a content hash, the complete station roster, source retrieval times, or source
revision identifiers. `site/data/latest.json` carries only a short input sample and is overwritten
by the next complete issue.

The code also does not enforce issue immutability: exporting again to the same monthly path
overwrites the CSV and sidecars. Git history may retain the earlier version, but verification
loads only the current files in `forecasts/`. If issued performance is to be the gold-standard
record, an official issue should be write-once, or every rerun should receive a unique run ID and
an immutable manifest should designate the official vintage.

As of this review the repository contains one dated issue (`2026-09-01`) and none of its target
months has yet been scored. All accuracy claims are therefore retrospective; there is not yet an
out-of-sample issued-forecast track record. The distinction should remain prominent until enough
prequential cases accumulate.

Consequently, a third party cannot recreate an issued fit after USGS/NRCS revisions or audit the
committed MAE/coverage summaries from row-level errors using the repository alone. A maximum date
is useful provenance but is not a data vintage. For each issue and benchmark run, store a compact
content-addressed modeling-table snapshot (or immutable source snapshot), its schema and hash,
the full configuration and station roster, the code commit, and the row-level scored predictions
needed to reproduce summary metrics. This need not mean committing the large operational DuckDB.

### Medium: the inflow chain is reduced-form in ways that look physical but are not stable physically

The second stage in [`inflow_chain.py`](../src/forecasting/multivariate/inflow_chain.py#L47-L85)
fits 12 separate elevation-change equations and runs them on forecast inflow. Its inflow
coefficients vary by more than a factor of three across months in a full-history fit and even more
in the modern era. The area variant barely changes forecast accuracy because it still regresses
elevation change on a partial water balance rather than conserving volume.

The first-stage inflow forecast is genuinely useful at short leads but has no demonstrated skill
beyond one water year. In the 1989-era walk-forward diagnostic, its MAE relative to a calendar-
month inflow climatology was 0.63, 0.72, 0.78, 0.95, 1.04, and 1.07 at leads 1, 3, 6, 12, 18,
and 24. Current snow and water-year precipitation should not be expected to predict the next
unobserved snow season.

### High: uncertainty is not decision-ready

[`quantiles.py`](../src/forecasting/quantiles.py#L19-L34) pools residual quantiles by model and
lead across all issue months. The errors are strongly heteroskedastic by season. For example, the
blend's empirical central-90% width at lead 6 is about 3.50 ft for accumulation-season issues and
1.49 ft for recession-season issues. The corresponding conditional coverages are 0.83 and 0.98,
even though aggregate coverage is close to 0.90. Aggregate calibration conceals operationally
important miscalibration.

The public wording says the band contains the correct value in “90 of 100 past forecasts”
([`site/index.qmd`](../site/index.qmd#L47-L57)). The reported aggregate coverage is actually
0.87--0.89 at key leads, and with only about 13 hydrologic years there is not enough independent
tail information to make that frequency sound precise. Until a longer issued record exists, call
it a *nominal central 90% interval calibrated from retrospective errors* and publish observed
coverage with uncertainty. Blocked split-conformal calibration or pooled adjacent-season
calibration can give more honest finite-sample behavior, but cannot manufacture information in
unobserved regimes.

The reported “CRPS” is the unweighted mean of pinball losses at five quantiles
([`quantiles.py`](../src/forecasting/quantiles.py#L37-L55)). It is a mean quantile-loss proxy, not
the continuous ranked probability score. Rename it or compute a recognized approximation to
CRPS; weighted interval score (WIS) is particularly natural for a finite quantile set.

Intervals are constructed separately at each lead. They are marginal bands, not samples from a
joint trajectory distribution. One therefore cannot validly derive probabilities for the spring
maximum, date of minimum, duration below a threshold, or cumulative management exposure from
them. Maxima of marginal point forecasts are also not generally the median or expected maximum.

The central functional is not defined consistently. Ridge components are fitted by squared error
(mean-oriented), blend weights are selected by absolute error (median-oriented), and then an
empirical median residual is added to create `q50`. A weighted mixture of conditional-mean
predictions chosen by MAE is not automatically either a conditional mean or median. Choose the
decision functional first—normally median for MAE and asymmetric quantiles for threshold loss—and
fit/calibrate it consistently.

Finally, the published point and median are internally inconsistent. In the 2026-09 issue, the
lead-24 blend point is 4190.001 ft but its residual-calibrated `q50` is 4188.833 ft, a 1.168-ft
difference. Because the blend is tuned to MAE, the calibrated median is the natural headline if
its bias correction survives honest testing. A coherent ensemble would avoid presenting two
conflicting notions of the central forecast.

### High: validation uncertainty is much larger than the tables imply

The 157 cutoffs from 2011-08 through 2024-08 overlap heavily. A given issue-month/headline cell
contains only 13 cases. Treating 157 errors as independent would materially understate uncertainty.

A circular moving-block bootstrap over cutoff sequence, using 24-month blocks and 20,000 draws,
gives the following diagnostic 95% intervals for blend MAE:

| Lead | MAE | Approximate 95% interval |
|---:|---:|---:|
| 1 | 0.126 | 0.100--0.154 |
| 3 | 0.319 | 0.251--0.397 |
| 6 | 0.521 | 0.369--0.727 |
| 12 | 1.069 | 0.720--1.493 |
| 18 | 1.615 | 1.095--2.155 |
| 24 | 1.999 | 1.373--2.614 |

For the paired difference `naive_last MAE - blend MAE`, the corresponding intervals are clearly
positive through lead 6, but include zero at leads 12, 18, and 24. These are not formal sampling
intervals under a fully specified data-generating process, but they are a better representation
of evidential precision than a three-decimal rank table.

The small gains from increasingly complex blends are especially uncertain. The lead-12 MAE
improvement of `blend3_swe` over `blend` is 0.022 ft, with an approximate paired block-bootstrap
interval of -0.016 to 0.073 ft when expressed as improvement. Similar intervals at 18 and 24
months include zero. These differences do not support extra production complexity.

The existing blend also imposes a nonincreasing total covariate weight with lead
([`blend.py`](../src/forecasting/multivariate/blend.py#L101-L136)). That is a defensible variance-
control device, but not a scientific constraint: the relevance of current snow can increase as a
target approaches melt before it decreases after the current water year. Because “covariate” also
includes current elevation and head difference, the weight has no single physical interpretation.
If blending remains, use a much lower-dimensional gate whose form is chosen before the locked
evaluation rather than interpreting the current fitted curve substantively.

There are additional evaluation risks:

- The autoresearch loop repeatedly selects features and model classes against the same 2011--2024
  outer period and the same 13 headline cases. Inner CV protects weight fitting at each outer
  cutoff, but it does not protect the outer record from repeated human/model selection.
- Leave-one-cutoff-calendar-year-out interval scoring still allows adjacent cutoffs in another
  year to share target hydrologic events at long leads
  ([`quantiles.py`](../src/forecasting/quantiles.py#L59-L70)).
- Historical CV uses today's revised gauge values and today's reconstructed feature roster, not
  issue-vintage inputs. The documentation discloses this, but “walk-forward” alone should not be
  read as a fully operational hindcast.
- [`cross_validate.py`](../src/forecasting/cross_validate.py#L25-L50) aligns actuals by row number
  and lead rather than forecast target date. The present target series is continuous, so this is
  not causing current results, but one missing month would silently score predictions against the
  wrong observations.
- `naive_last` is weak at short seasonal horizons. The evaluation needs stronger operational
  baselines: a seasonal-change/level-conditioned climatology, analog or ensemble climatology, and
  direct comparisons with archived NRCS and CBRFC products where equivalent targets exist.

## Performance attribution: implementation versus external data

The evidence does not support choosing only one cause.

For leads 1--3, the priority is internal: use the latest observed state, fix target support, and
pool the regression appropriately. For the remainder of the current snow/runoff season, existing
SNOTEL and flow information is already valuable, with external ensemble streamflow likely to add
more. Beyond roughly 9--12 months, honest skill depends increasingly on future-forcing ensembles
and declared management scenarios.

### 1. Current implementation is a primary cause

Using a homogeneous modern training era materially improves the state/direct long leads. Giving
the physical models the actual future tributary inflow still leaves approximately 1 ft of
24-month state-space MAE. That residual is too large to attribute the full performance gap to
future inflow data. It points to omitted water-balance terms, working in elevation, incomplete
inflow coverage, regime changes, and parameter instability.

The target discontinuity is not, by itself, the explanation for every disappointing result.
Moving the ETS anchor to 1989 changes its MAE by only about one or two hundredths of a foot at
most evaluated leads, and changes the direct models little at short leads. Its larger benefit for
state/direct long leads is evidence of sensitivity, not a clean causal attribution: the ablation
also changes era and sample size. Future-forcing ignorance and structural misspecification remain
the dominant long-range limitations.

### 2. Future forcing is also a primary cause beyond about 9--12 months

With the modern training era, replacing forecast inflow with realized future inflow gives:

| Model | Lead | Operational inflow MAE | Oracle inflow MAE |
|---|---:|---:|---:|
| `inflow_chain` | 6 | 0.538 | 0.428 |
| `inflow_chain` | 12 | 1.064 | 0.671 |
| `inflow_chain` | 24 | 1.798 | 0.907 |
| `state_space` | 6 | 0.562 | 0.441 |
| `state_space` | 12 | 1.076 | 0.708 |
| `state_space` | 24 | 1.845 | 0.958 |

These oracle runs are diagnostics, not deployable forecasts. They demonstrate that unknown future
hydrology accounts for a growing share of error, while also showing that the present structural
model remains inadequate even after that uncertainty is removed.

### 3. More historical predictor columns alone are unlikely to solve it

The sample is small, nonstationary, and strongly dependent. A larger grab bag of climate indices
or a more flexible machine-learning model would raise variance and researcher degrees of freedom.
The highest-value new information is not another contemporaneous scalar; it is a coherent future
forcing distribution and explicit management assumptions.

## Data needed for the next credible forecast

### Use and repair what is already present first

Before another ingestion round:

1. Establish a homogeneous target and explicit daily-peak versus monthly-mean estimands.
2. Build stable site-level SNOTEL anomaly indices and version their roster.
3. Use the existing hypsometry as the state transformation.
4. Decide how existing reservoir storage, lagged climate, north-arm elevation, and breach flow
   belong in a causal balance or scenario definition rather than testing them as arbitrary extra
   regressors.
5. Repair validation, scoring names, season-conditional calibration, and trajectory uncertainty.

A useful first storage model does **not** require every flux to be separately observed. On the
modern record, convert end-of-month elevation to volume and calculate a residual net flux after
the measured lake-boundary inflows. Partially pool that residual by season and lake area, and
resample correlated residual sequences in the forecast. Call it `net_unmeasured_flux`; do not call
it evaporation or consumptive use. This immediately fixes units, geometry, and trajectory
propagation with existing data. Separate precipitation, evaporation, and use data become valuable
when they improve hindcast skill or make a policy scenario identifiable.

### Highest-value external/operational inputs

1. **CBRFC ensemble streamflow prediction traces or archived hindcasts.** ESP combines current
   hydrologic state with alternative historical future weather traces to produce flow ensembles;
   this is exactly the missing bridge from observed snow/soil state to future tributary volumes
   ([CBRFC ESP methodology](https://www.cbrfc.noaa.gov/wsup/doc/Water_Supply_ESP.pdf)). Archived or
   reforecast vintages are essential for honest validation; today's forecast alone is not enough.
2. **Meteorological ensembles for over-lake precipitation and evaporation.** At minimum: gridded
   precipitation, temperature, humidity, wind, and radiation, transformed to lake precipitation
   and evaporation and sampled as correlated trajectories. Historical reforecasts are preferable
   to a current-only API.
3. **Consumptive use, diversions/deliveries, and reservoir operations.** Policy and operations are
   often better treated as declared scenarios than forecast as if they were weather. Include
   current-use, planned-conservation, and high-delivery cases.
4. **Causeway configuration and exchange.** Represent berm/opening status as a scenario, or fit a
   coupled north/south storage-exchange model if data support it.
5. **Minor/ungauged surface inflow and groundwater.** The three selected gauges are sensible major
   inflows, but the USGS Great Salt Lake Hydro Mapper shows additional tributaries and bay flows
   that matter to closure ([USGS Hydro Mapper](https://webapps.usgs.gov/gsl/data.html)).
6. **NRCS exceedance forecasts as an operational benchmark or ensemble anchor.** The repository
   already ingests these, although the dated lake forecast series is short. NRCS reports the same
   hydrologic ingredients—snow, precipitation, soil moisture, reservoirs—and publishes exceedance
   distributions ([NRCS Great Salt Lake forecast page](https://www.nrcs.usda.gov/state-offices/utah/the-great-salt-lake)).

Lower priority is a large set of teleconnection indices, satellite area as a redundant state
measure, a Bayesian sampler before fixing the balance equation, or flexible ML on the current
annual-sized sample.

## Recommended model/product architecture

### Product A: operational statistical forecast

Purpose: best possible prediction over the horizon on which current state contains demonstrable
signal, likely 1--9 months and at most 12.

- Train primarily on the homogeneous modern target record.
- Initialize from a robust end-of-month state and forecast changes from a strong seasonal/level-
  conditioned baseline.
- Use compact cyclic seasonality or partial pooling across issue months and leads.
- Drop structurally absent predictors by season.
- Predict spring daily peak, peak monthly mean, September mean, and seasonal minimum directly when
  those are the decision targets; do not derive every scalar from unrelated marginal point fits.
- Produce calibrated trajectory samples or a compact multivariate residual process.
- Make this model the headline where it wins in locked testing.

### Product B: scenario water balance

Purpose: explain and quantify the 6--24+ month consequences of hydrologic and management
assumptions.

- Initialize south-arm storage from observed elevation and hypsometry.
- Advance storage with complete-enough flux components and explicit residual closure.
- Drive future inflow and meteorology with coherent ensemble traces, not independent marginal
  values or one mean path.
- Cross those traces with a small declared set of use, delivery, reservoir, and causeway scenarios.
- Estimate bias/process innovations on closed multi-month balance windows; do not ask a free AR
  coefficient to stand in for every missing flux.
- Convert each simulated storage trajectory back to elevation.
- Calibrate the ensemble against genuine hindcasts, including extremes and regime changes.

The state-space machinery is useful only if latent states or uncertain fluxes need reconciliation.
A deterministic storage balance plus stochastic forcing/residual ensembles would be simpler and
more honest than the current generic Kalman model. A Bayesian implementation can follow when its
priors encode actual physical/component knowledge rather than merely regularizing an incomplete
equation.

### Keep the two products distinct

Do not require one model to own every lead. Publish the empirical near-term forecast and the
conditional scenario forecast side by side, with a clearly declared transition. A low-degree-of-
freedom stack or horizon gate can be tested later. The current three-season-by-24-lead blend adds
many choices for gains of only hundredths of a foot, and its cache fingerprint ignores covariates
([`blend.py`](../src/forecasting/multivariate/blend.py#L139-L159)). Simplification is warranted.

## What would make the product practically useful?

“Good” is decision-relative. A 90% interval is useful when it changes an action, not merely when
its marginal coverage rounds to 0.90. The product should first elicit:

- the action (berm operation, water delivery, habitat/access planning, salinity response, etc.);
- the threshold or loss curve;
- the decision lead time;
- the cost of false action versus missed action.

For example, Utah's adaptive berm statute invokes a 4,190-ft condition
([Utah H.B. 453](https://le.utah.gov/~2024/bills/hbillint/HB0453S01.htm)). For such a sharp rule,
the useful output is `P(elevation <= 4190 at the decision date | scenario)`, plus the effect of
candidate actions, not a visually precise line. If the central 90% interval spans the threshold,
the forecast can still be useful—but only if the probability and loss asymmetry support a choice.
The current 2026-09 lead-1 interval, 4189.622--4190.173 ft, does span 4190; this is exactly a case
where a calibrated threshold probability is more decision-relevant than the point 4189.877.

Reasonable initial sharpness goals, to be confirmed with stakeholders, are:

- 1--3 months: a central 90% half-width of roughly 0.25--0.50 ft for threshold operations;
- spring peak: total 90% width near or below 1 ft by an April/May issue, and below about 2 ft by a
  January/February issue;
- water-year end: similar decision-specific width by the last date an intervention can matter;
- 12--24 months: no universal width target; scenario separation and calibration matter more than
  an unconditional band that spans several feet.

Current performance is mixed against these provisional standards. The 2026-09 blend interval is
about 0.55 ft wide at lead 1, 2.23 ft at lead 6, 4.41 ft at lead 12, and 7.23 ft at lead 24.
Historical peak interval widths are approximately 3.15, 2.27, 1.84, 1.20, and 0.60 ft for January
through May issues. The May peak and very near term are potentially actionable; the current
long-range unconditional bands are primarily statements of ignorance.

Another useful yardstick is whether uncertainty is smaller than a contemplated intervention. Near
4190 ft, the local hypsometry gives a first-order response of about 1.35 ft per 500 kaf before
subsequent losses and feedbacks. A 7.23-ft unconditional band cannot resolve an effect of that
size. Paired scenario simulations can still estimate the *difference* between action and no-action
under common weather traces much more precisely; that is one reason scenario deltas can be useful
when an unconditional level forecast is not.

The public product should emphasize:

- threshold probabilities at relevant dates;
- spring peak and true seasonal-low distributions, including timing;
- probability and duration below/above management thresholds;
- scenario deltas such as the lake response to an additional 250, 500, or 800 kaf;
- expected decision loss or regret relative to a simple baseline;
- the share of uncertainty due to weather, operations/use, parameter uncertainty, and residual
  imbalance where that decomposition is defensible.

## Validation standard for the next round

1. **Stop reusing the current outer period as a test.** It cannot be made untouched retroactively:
   its results have already guided many model and feature decisions. Cutoffs from 2024-09 through
   2025-08 have complete outcomes through lead 12 and have not been used as origins in the stored
   24-month CV, so they could be reserved as a limited confirmation set. Their target events do
   overlap earlier long-lead scores, however, so only future write-once issued forecasts are truly
   untouched. Use nested/rolling development before that prospective test.
2. **Reconstruct vintages where feasible.** Archive inputs at every production issue. For external
   ensembles, obtain reforecast/hindcast vintages rather than validating today's model on revised
   histories.
3. **Use water-year dependence in inference.** Report paired model differences with uncertainty
   clustered or block-bootstrapped by hydrologic year. Avoid declaring a winner from a 0.01-ft
   difference.
4. **Score the actual product.** Use MAE for medians, RMSE only where large misses carry quadratic
   cost, WIS/CRPS for distributions, Brier/reliability scores for thresholds, and multivariate or
   event scores for trajectory-derived peaks and lows.
5. **Report conditional calibration.** Slice by issue month/season, lead, lake regime, and extreme
   water years. Pool adjacent cells hierarchically when samples are too small.
6. **Use stronger baselines.** Include seasonal change conditioned on current level, analog/ESP
   climatology, and equivalent NRCS/CBRFC forecasts. The bar is not just last month's value.
7. **Stress test regimes.** Leave out major wet/dry episodes and causeway regimes in turn; inspect
   coefficient and forecast stability, not only aggregate error.
8. **Align by target date.** Fail loudly on missing months rather than assigning actuals by row
   position.

Suggested acceptance gates:

- a claimed operational improvement must have a paired uncertainty interval that excludes no
  improvement on the locked test, or be retained for a separately justified physical reason;
- 90% intervals should achieve approximately nominal coverage within important issue seasons, not
  only in aggregate, and should improve WIS over the baseline;
- all published event probabilities must be computed from coherent paths;
- the water-balance residual should be reported in volume units and remain stable across lake
  elevations and held-out regimes;
- a long-range forecast must state its meteorological and management scenario or the weights used
  to average scenarios.

## Recommended sequence of work

### Round 0: correct claims and freeze the benchmark

- Rename September “water-year-end elevation,” not “annual low.”
- Name the current peak target “maximum April--June monthly mean.”
- Rename the five-quantile score from CRPS to mean quantile loss, or replace it with WIS/CRPS.
- Mark the state-space model experimental and remove the claim that it solves the inflow
  substitution problem.
- Freeze the current results; predeclare the limited recent confirmation cutoffs and the future
  prospective evaluation before further iteration.

### Round 1: repair target, features, and evaluation

- Use 1989-10 onward as the main target era; quantify the value of earlier data separately.
- Retain the last valid daily elevation as the forecast's initial state and add the endpoint
  seasonal-change benchmark.
- Create a stable, versioned SNOTEL index.
- Fix date-based scoring and season-conditional interval calibration.
- Add block/cluster uncertainty for every reported model difference.
- Add strong seasonal/analog baselines.

### Round 2: simplify the operational model

- Replace hundreds of isolated regressions with a compact pooled seasonal/direct model.
- Forecast decision targets directly.
- Remove blend degrees of freedom unless they win on the locked test.
- Use the calibrated median consistently as the point forecast.

### Round 3: build the scenario water balance

- Move the state to volume and close the principal fluxes.
- Generate coherent paths under dry/median/wet forcing crossed with a small number of explicit
  management scenarios.
- Add a stochastic residual/latent-flux layer only where diagnostics show it is needed.
- Validate with archived forcing ensembles and issue vintages.

### Round 4: publish a decision product

- Publish near-term empirical forecast and longer-term conditional scenarios separately.
- Show threshold probabilities, event distributions, scenario deltas, and calibration history.
- Continue accumulating true issued-forecast performance; do not continually retune against it.

## Likely code simplification

This review supports a net reduction in modeling code. Once replacements are verified, likely
retirement candidates are:

- the current elevation-based `state_space` implementation;
- `inflow_chain_area`, which changes a regressor without changing the conserved state;
- most three-component and season-by-lead blend search machinery;
- structurally invalid issue-month/feature combinations;
- duplicated interval logic once all products are generated from calibrated trajectories.

The objective is not minimal code for its own sake. It is fewer statistical degrees of freedom,
fewer incompatible definitions of the center and interval, and a direct mapping from every model
term to either observed state, future forcing, management assumption, or residual uncertainty.

## Audit diagnostics and qualifications

All numerical diagnostics in this document used the repository's local `data/gsl.db` and result
artifacts as they existed on 2026-09-03. The existing test suite passes (`163 passed`). Passing
tests establish implementation consistency with current expectations; they do not test target
homogeneity, real-data state-space identifiability, conditional calibration, or decision value.

The additional review diagnostics were read-only and include:

- observation-count summaries for `monthly_elevation`;
- modern-record emulation of sparse first/fifteenth sampling;
- daily versus monthly spring-peak comparisons;
- water-year minimum-month frequencies;
- end-of-month seasonal-change and endpoint-anchored direct-model benchmarks;
- full walk-forward refits beginning in 1989-10;
- state-space parameter and interval inspection across cutoffs/training starts;
- oracle-future-inflow forecasts;
- inflow forecast comparison with calendar-month climatology;
- season-stratified empirical interval diagnostics;
- 24-month circular moving-block bootstrap summaries.

The block-bootstrap intervals are descriptive sensitivity estimates, not proofs of stationarity or
formal guarantees. The 1989 training-start ablation jointly changes record quality, era, and sample
size. The oracle-inflow experiment diagnoses information limits but is intentionally infeasible as
a forecast. These qualifications do not weaken the central conclusion: the near-term signal is
real, while credible longer-horizon forecasting requires both a better-specified storage balance
and explicit distributions/scenarios for predictors that have not yet occurred.
