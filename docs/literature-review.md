# Literature review: forecasting Great Salt Lake level, area, and volume

Survey date: 2026-09-02. Purpose: identify the best existing forecasts of Great Salt Lake (GSL) level, area, and volume, the variables and data they use, and how our univariate results compare, ahead of implementing multivariate methods.

Elevations are south arm (USGS 10010000, Saltair) in feet unless noted. 1 ft = 30.48 cm.

## 1. Summary

- Nobody publishes a routine monthly-resolution GSL level forecast. The only operational product is the USDA NRCS Utah Snow Survey advisory outlook (Jan-May, since water year 2024), which converts a SNOTEL-regression inflow forecast into a rise-to-seasonal-peak estimate.
- The only peer-reviewed multi-year forecasts with unit-bearing out-of-sample error are from the Utah State University climate group (Gillies, Wang, DeRose): annual level tendency regressed on lagged watershed precipitation, Pacific NINO4 sea surface temperature, and a 576-year tree-ring reconstruction.
- Sub-annual academic work (Lall, Abarbanel, Asefa, Moon; 1996-2007) forecasts biweekly or monthly lake volume with nonparametric nonlinear regression or SVMs, optionally with SOI/PNA/CNP indices. Reported errors are in normalized volume units and are not directly comparable.
- Process models (Mohammed and Tarboton mass balance, White et al. 2015, Dunn et al. 2025, GSLIM, Strike Team Monte Carlo) reproduce level almost perfectly when driven by observed inflow (NSE 0.99 annual), but are used for scenarios, not verified forecasts.
- Our ETS damped-trend seasonal model is roughly at parity with the published statistical work at the 12-month horizon and slightly behind the NRCS spring-peak outlook from a March cutoff. The literature is unanimous that the missing information is inflow, and inflow is predictable from snowpack.

## 2. Operational forecasts

### 2.1 NRCS Utah Water Supply Outlook Report (the benchmark)

- Publisher: USDA NRCS Utah Snow Survey. Monthly Jan-May. https://www.nrcs.usda.gov/state-offices/utah/water-supply-outlook-reports and https://www.nrcs.usda.gov/state-offices/utah/the-great-salt-lake
- Target: Apr-Jul (May-Jul from April) inflow volume to GSL at 10/30/50/70/90 percent exceedance; rise from issue date to seasonal peak; implied peak stage. Horizon 1-6 months.
- Method: standard NRCS statistical regression of seasonal volume on SNOTEL snow water equivalent (SWE) and accumulated precipitation, on a synthetic "GSL inflow" point built from the Weber, Provo, and Bear watersheds (about 38 years of record). Rise is derived from inflow volume via hypsometry. Explicitly advisory; ignores diversions, reservoir operations, berm and causeway management, and evaporation.
- Inputs and sources: SNOTEL SWE and precipitation (NRCS NWCC AWDB API); USGS inflow gauges (Bear near Corinne 10126000, Weber near Plain City 10141000, Jordan at 1700 S 10171000 or similar); USGS elevation-area-volume tables (Root 2023, DOI 10.5066/P9DGG75W).
- Verification (informal, from published outlooks vs USGS actuals):

| Issue date | Start stage | Most-probable rise | Implied peak | Actual peak | Error |
|---|---|---|---|---|---|
| 2024-03-01 | ~4194.1 | +1.2 ft | ~4195.3 | 4195.2 (Jun) | +0.1 |
| 2024-04-01 | 4194.3 | "just over 4195 ± 0.5" | ~4195.1 | 4195.2 | -0.1 |
| 2025-02-01 | 4192.8 | +0.83 ft | 4193.6 | 4193.6 (Jun) | 0.0 |
| 2025-04-01 | 4193.4 | +0.58 ft | ~4194 ± 0.5 | 4193.6 | +0.4 |
| 2025-05-01 | 4193.4 | +0.3 ft | 4193.7 | 4193.6 | +0.1 |
| 2026-03-01 | 4192.0 | 0 to +1 ft | ~4192.5 | 4192.6 (Apr) | ~0 |
| 2026-04-01 | 4192.3 | "essentially no change" | ~4192.3 | 4192.6 | -0.3 |

No formal skill statistics are published. Three seasons of verification is too few to estimate an error distribution, and 2024-2026 were all normal-to-dry springs; the model has not yet been tested on a 2023-type year (5.5 ft rise).

### 2.2 NOAA/NWS Colorado Basin River Forecast Center (CBRFC)

- No lake-level product. Publishes Apr-Jul unregulated volume forecasts for the GSL tributaries (sub-area "GL") by Ensemble Streamflow Prediction: current modelled snow and soil-moisture states run forward with 30 historical weather years, giving 10/50/90 percent exceedance volumes. https://www.cbrfc.noaa.gov/wsup/doc/doc.html
- April-issue verification (average absolute error of Apr-Jul volume): Weber at Oakley 18 percent, Provo at Woodland 16 percent, Big Cottonwood 16 percent.
- Useful as a covariate source: daily ESP guidance and monthly official forecasts are downloadable per forecast point.

### 2.3 Utah DWR, GSL Basin Integrated Plan, GSLIM

- Great Salt Lake Integrated Model (GoldSim, Jacobs 2017): river basin, wetland, and four-bay lake modules producing elevation and salinity for planning scenarios. Calibration statistics not published. The 2019 Advisory Council study projected 2019-2051 scenarios (baseline -3.3 to -3.8 ft) and explicitly recommended developing 5-10 year lake-level forecasts, which do not yet exist.
- GSL Basin Integrated Plan (due Nov 2026): VIC hydrology on MACA-downscaled CMIP6, RiverWare operations, upgraded GSLIM. No public projections yet. https://water.utah.gov/gsl-basin-integrated-plan/

### 2.4 USGS

- No forecast. Provides the observations (Hydro Mapper, https://webapps.usgs.gov/gsl/), the 2023 topobathymetric DEM and elevation-area-volume tables, and historical water and salt balance reports (Arnow 1978; Wold et al. 1997; Loving et al. 2000).

### 2.5 GSL Strike Team (U of U Gardner Institute and USU), Jan 2026

- Annual water-balance model in R (Tarboton) with USGS area-volume tables; 1000 Monte Carlo realizations resampling 2000-2025 annual inflow, precipitation, and evaporation; 30-year horizon under added-inflow scenarios (+250, +375, +800 kaf/yr). No hindcast verification. Code: https://www.hydroshare.org/resource/885979f4acdd412ab3cf09799eab7ead/

## 3. Peer-reviewed statistical forecasts

### 3.1 Gillies, Chung, Wang, Kokoszka (2011), J. Hydrometeorology, doi:10.1175/2010jhm1352.1

- Target: annual level tendency (dGSL), integrated to level. Horizon up to 8 years.
- Model: lagged regression. PLag uses lagged GSL-watershed precipitation; PCLag adds a principal component of Pacific SST (NINO4 region, quasi-decadal oscillation), with a notable 42-year lag.
- Data: Kaplan Extended SST; Utah Climate Center COOP station precipitation, 1900-2007.
- Skill: in-sample RMSE 0.86-0.90 ft on dGSL and 0.9-1.5 ft on level; out-of-sample 8-year forecasts issued annually after 1985 average 3.02 ft RMSE (PCLag) vs 4.02 ft (PLag). Skill collapses in ENSO-extreme years (1992-93, 1997-98).
- Live product: Utah Climate Center "GSL Annual Level Prediction", https://climate.usu.edu/GSL.php

### 3.2 Gillies, Chung, Wang, DeRose, Sun (2015), J. Hydrology 529:962-968

- Target: annual dGSL, 5-year-ahead, hold-out 2001-2005.
- Model: AR on observed dGSL (ObsAR) vs ARX with the DeRose et al. (2014) 576-year tree-ring dGSL reconstruction as exogenous input (TreeARX-1, -2).
- Skill: RMSE 40.2 cm/yr (ObsAR) vs 35.6 and 32.1 cm/yr (TreeARX); standard deviation of annual tendency is 32.8 cm/yr, so even the best model is barely better than climatology on this 5-point hold-out.

### 3.3 Wang, Gillies, Jin, Hipps (2009), J. Climate; Wang and Gillies (2010); Wang et al. (2011, 2018)

- Diagnostic: GSL level is coherent with the Pacific quasi-decadal oscillation (NINO4 SST) at 10-15 year periods, with precipitation leading level by several years. Basis for the USU multi-year predictor. No RMSE reported.

### 3.4 Moon, Lall, Kwon (2007), Int. J. Climatology, doi:10.1002/joc.1533

- Target: monthly GSL volume, 1-24 months, iterated and direct.
- Model: multivariate locally weighted polynomial regression on lagged volume plus SOI, PNA, and CNP atmospheric indices; lags chosen by average mutual information.
- Skill: improvement over volume-only model reported; numbers not in abstract. Closest analogue to our problem in resolution and horizon; worth retrieving the full text.

### 3.5 Asefa, Kemblowski, Lall, Urroz (2005), Water Resources Research, doi:10.1029/2004wr003785

- Target: biweekly volume (154-year record), 2 weeks to about 4 months.
- Model: SVM regression on embedded lagged volumes vs ANN. RMSE 0.022-0.027 in normalized units.

### 3.6 Lall, Sangoyomi, Abarbanel (1996), WRR; Abarbanel et al. (1996), Energy

- Nonlinear dynamical-systems view: biweekly volume 1847-1992 behaves as a low-dimensional attractor (about 3-4 degrees of freedom); local nonparametric predictors give useful skill for months to a few years. Qualitative skill only.

### 3.7 Chen group ARFIMA / FIGARCH (Sun et al. 2007; Li et al. 2007; Sheng and Chen 2009, 2010)

- Univariate long-memory models of elevation. Claim improvement over ARMA; no exogenous inputs, no unit-bearing metrics in abstracts.

### 3.8 Shrestha (2021), USU dissertation, doi:10.26076/cc3b-2508

- Multivariate relevance vector machine predicting Utah streamflow and GSL elevation at 1-5 year leads from past streamflow, snowpack, local meteorology, SST regions, and NAO/AMO/PDO/ENSO indices. Best skill at 2-4 year leads for streamflow; GSL-specific numbers not extracted.

### 3.9 Older stochastic level-probability studies (Utah Water Research Lab, 1979-1987)

- Multivariate AR(1) generation of streamflow, lake precipitation, and evaporation feeding a water balance; exceedance probabilities of level, no point-forecast error.

## 4. Process and mass-balance models

- Mohammed and Tarboton (2011, 2012), WRR: monthly GSL mass balance with salinity-adjusted Penman evaporation; k-NN resampled inputs for ensemble sensitivity. A 25 percent streamflow change gives ±0.55-0.66 m over 5 years; re-equilibration about 15 years. Bathymetry-driven preferred levels (multimodality).
- White, Null, Tarboton (2015), PLOS ONE: 2-day-step Fortran mass balance, 1966-2012 hindcast, NSE 0.99 on annual level when driven by USGS Bear/Jordan/Weber flows and PRISM precipitation with evaporation by closure.
- Dunn, Crookston, Dutta, Neilson (2025), J. Hydrology: Regional Studies, doi:10.1016/j.ejrh.2025.102768: open-source monthly multi-layer water and salt balance with causeway breach exchange, 2017-2023. Code on HydroShare.
- Bigalke, Loikith, Siler (2025), GRL: water-balance attribution of the 2022 record low using USGS gauges, station precipitation, ERA5-Land evaporation.
- Merck and Tarboton (2024, preprint): 1847-2023 inflow reconstruction; consumptive use up to 2.3 km3/yr explains up to 4.6 m of decline. Implies non-stationarity that 1990s-2000s time-series models ignore.

Takeaway: given observed inflow and precipitation, the level is essentially deterministic. All forecast uncertainty lives in the inflow forecast.

## 4.1 Bayesian and state-space water balance

This branch was absent from the survey until 2026-09-03. It is the methodological gap
between section 3, which is regression on the lake record, and section 4, which is a
deterministic mass balance with no account of measurement error.

- Smith and Gronewold (2018), Advances in Water Resources, arXiv:1710.10161: the Large Lake
  Statistical Water Balance Model (L2SWBM), a Bayesian water balance for Lakes Superior and
  Michigan-Huron. It reconciles inflow, over-lake precipitation, evaporation and level, and
  it states the bias and the uncertainty of each measurement instead of treating an input as
  exact. The authors compare 26 model forms and choose on both closure and run time. This is
  the closest published precedent for a probabilistic water balance of a large lake.
- NOAA GLERL (2018), summary report on the same model, gives the operational form.
- Durbin and Koopman, "Time Series Analysis by State Space Methods": the standard reference
  for the filter and the smoother. `statsmodels` implements both, and this project already
  depends on `statsmodels` for the exponential smoothing models.
- Slater and Villarini, and the wider data-assimilation literature on snow (for example the
  ensemble Kalman filter and particle filter comparison of Slater and Clark, 2006, for snow
  water equivalent): a filter is the standard way to carry a snow or storage state forward
  under uncertainty.
- Quaedvlieg (2021), J. Business and Economic Statistics, doi:10.1080/07350015.2019.1620074:
  tests that compare 2 forecasters over a complete path of horizons instead of 1 horizon at
  a time. The relevant reported pattern is that an iterated state-space method trails a
  direct method at short leads and improves with the horizon.

Takeaway, and the reason it matters here: the models in this repository are all direct. Each
lead has its own fit, so nothing links the 24 months of a path, and the interval comes from
past errors rather than from the model. `state_space` (section 4.1 of `docs/model-spec.md`)
is the first iterated model here. It is better than `inflow_chain`, the recursion it
replaces, at every lead and on the spring peak, and its CRPS at lead 12 is better than
`blend`. Its point forecast does not yet beat `blend` at any lead.

## 5. Area and volume

- No paper forecasts surface area or volume as a primary target beyond deriving them from level via hypsometry. Radwin and Bowen (2024) provide Landsat/Sentinel water area 1984-2023 (error under 1 percent deep water, about 4 percent in shallow bays); Root (2023) provides the USGS elevation-area-volume tables. Our level forecasts convert to area and volume through those tables.

## 6. Basin inflow predictors (inputs for multivariate work)

- Neisary et al. (2025), Environmental Modelling and Software: ML post-processing of National Water Model flows with reservoir storage and SNOTEL SWE across 30 GSL-basin gauges; median KGE +65 percent, RMSE -25 percent.
- Morovati (2025), USU dissertation: SWE, January baseflow, and soil moisture improve seasonal streamflow forecasts; soil moisture gives the largest gain.
- CBRFC ESP and NRCS regression forecasts (section 2) are the operational versions of the same information.

## 7. Best existing forecasts and their inputs

| Rank | Forecast | Horizon | Inputs | Data sources | Reported skill |
|---|---|---|---|---|---|
| 1 | NRCS Utah WSOR GSL outlook | 1-6 months, spring peak | SNOTEL SWE, accumulated precipitation, current stage, hypsometry | NRCS AWDB (SNOTEL), USGS 10010000, USGS area-volume tables | April outlook 0.1-0.4 ft on peak, 2024-2026 (3 seasons) |
| 2 | Gillies et al. 2011 / USU Climate Center | 1-8 years, annual | Lagged watershed precipitation, NINO4 SST PC, lagged level | Utah Climate Center COOP precipitation, Kaplan SST | 0.9-1.5 ft RMSE in-sample; 3.0 ft mean RMSE over 8-year forecasts |
| 3 | Gillies et al. 2015 TreeARX | 5 years, annual | Observed dGSL plus tree-ring dGSL reconstruction | USGS level, DeRose 2014 chronologies (NOAA paleo) | 32.1 cm/yr RMSE vs 40.2 for AR-only (5-point hold-out) |
| 4 | Moon, Lall, Kwon 2007 | 1-24 months, monthly | Lagged volume, SOI, PNA, CNP | USGS level via hypsometry, NOAA CPC indices | Improvement over univariate; magnitude not in abstract |
| 5 | Mass-balance models (White 2015, Dunn 2025) | Any, given inflow | Streamflow, precipitation, evaporation, salinity, causeway exchange | USGS gauges, PRISM, ERA5-Land or Penman | NSE 0.99 hindcast; no forecast skill |

## 8. Comparison with our current results

The README "Current results" section holds the one maintained set of numbers: walk-forward MAE by lead for the univariate and snowpack models, the spring-peak and water-year-end errors by issue month, and the refit comparison against the NRCS record. At 12 months the univariate models sit at rough parity with the published statistical work and barely beat persistence. From a January issue the snowpack models roughly halve the univariate spring-peak error.

## 9. Implications for the multivariate implementation

1. Priority predictor: SNOTEL SWE and accumulated precipitation for GSL-contributing sites, aggregated to a basin index (NRCS AWDB API; the NRCS GSL page lists the site set). This is what resolves the December-February cutoff problem documented in the README.
2. Second: tributary streamflow (USGS 10126000 Bear near Corinne, 10141000 Weber near Plain City, 10171000 Jordan at 1700 S) as lagged regressors, and the CBRFC ESP or NRCS seasonal volume forecasts as forward-looking regressors available at issue time.
3. Third, for horizons beyond 12 months: NINO4 SST and Pacific QDO index (Kaplan or ERSST via NOAA PSL), Utah Climate Division precipitation, and the DeRose 2014 tree-ring dGSL reconstruction. Expect small gains; the literature shows about 1 ft over 8 years.
4. Structure: a water-balance skeleton (level change = f(inflow, precipitation, evaporation) through hypsometry) with statistically forecast inflow will likely beat a free-form regression, because the process models show level is deterministic given inflow. Evaporation can be approximated by a seasonal climatology at first (Strike Team average 2.7 Maf/yr). Implemented twice: `inflow_chain` as a deterministic recursion and `state_space` as a filter (section 4.1 above). Neither beats the direct regression at short leads, which is where the peak is decided.
5. Evaluation: keep the walk-forward harness, but add a spring-peak metric (June level from Dec-Mar cutoffs) so results are comparable to NRCS, and report errors with and without 2022-2023.
6. Caution from Zhu et al. (2022) and our own naive_last results: for slowly varying lakes persistence is a strong baseline at short horizons; any multivariate model must be scored against naive_last at every horizon, as the harness already does.

## 10. Papers still worth retrieving in full

- Lall, Moon, Kwon, Bosworth (2006), WRR: locally weighted polynomial regression for short-term GSL forecasts; probably the cleanest sub-annual metrics. Cited but not returned.
- Moon, Lall, Kwon (2007) full text for the SOI/PNA/CNP gain magnitude.
- Gillies et al. (2015) full text for the ARX specification and lag orders.
- Smith and Gronewold (2018) full text for the L2SWBM priors and the sampler, before any
  Bayesian phase of `state_space`.
- NRCS WSOR PDFs (Jan-May 2024-2026) for the exact GSL inflow regression sites; curl to nrcs.usda.gov is blocked from this sandbox.

## 11. Where this project can add value

- The NRCS outlook stops in May, so no operational product forecasts the water-year-end (autumn) low, which is where salinity and brine-shrimp stress peak and what the Strike Team and Commissioner's statements target.
- Nothing dated and versioned exists between the spring outlook (1-6 months) and the 30-year scenario models. A 6-24 month probabilistic elevation forecast that combines NRCS or CBRFC inflow exceedances with an explicit water-balance evaporation term, runs year-round, and is scored against the USGS gauge would be the first of its kind rather than merely competitive.
- Policy thresholds are all in south-arm elevation (4,198 ft healthy minimum, 4,192 ft serious adverse effects, 4,188.5 ft 2022 record low), so elevation stays the primary target and area and volume remain lookups.
