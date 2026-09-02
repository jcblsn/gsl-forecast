# Autoresearch program for gsl-forecast

This file is the research strategy for an agent-driven improvement loop, in the pattern of
karpathy/autoresearch (see `autoresearch-memo.md`).

## The three primitives

- Editable asset: files under `src/forecasting/multivariate/` and the entries that register
  them in `src/forecasting/registry.py`. Nothing else may be edited by the loop.
- Metric: `peak_mae_feb` for the model under study, as logged by `gsl-cv` (spring-peak MAE
  in feet from cutoffs whose data end in January, i.e. the February 1 outlook). Secondary
  metrics to report but not optimise: `mae_h6`, `crps_h6`, `wyend_mae_apr`.
- Harness: `uv run gsl-cv --no-plots --models <model>,ets_damped_s12` with config defaults
  (24-month horizon, cutoffs from the last 15 years, training from 1960). The harness, the
  data pipeline, the univariate baselines and the scoring code are fixed.

## Loop

1. Read the latest experiment's metrics: `uv run gsl-results <experiment_id>`.
2. Propose one change to one multivariate model (features, lags, regularisation, fallback
   rules, the stage-two step in `inflow_chain`, a new model file).
3. Run the harness. Compare `peak_mae_feb` and `mae_h6` against the previous run and
   against `ets_damped_s12`.
4. Keep the change (commit) if `peak_mae_feb` improves without `mae_h6` getting worse by
   more than 0.05 ft; otherwise revert.
5. Log one line per experiment in `outputs/autoresearch.log`: date, model, change, metrics.

## Guardrails

- A model may only use columns dated at or before the cutoff. The leakage test in
  `tests/test_multivariate.py` must keep passing; add a case for any new feature.
- Do not touch `train_start`, the cutoff policy, or the horizon.
- Do not add models that need data outside `monthly_elevation` and `monthly_covariates`
  without first extending the pipeline in a separate, reviewed change.

## Ideas queue

- Percent-of-median SWE per basin instead of raw inches (site roster changes over time).
- Separate Bear/Weber/Provo SWE terms instead of the basin-weighted mean.
- Soil moisture and water-year precipitation from SNOTEL as runoff-efficiency terms.
- Issued NRCS/CBRFC inflow exceedance forecasts (`data/external/`) as a covariate in season.
- Bathymetry (USGS elevation-area-volume table) so `inflow_chain` uses real area for
  evaporation instead of the elevation proxy.
- Stacking: regress `ets_damped_s12` residuals at each horizon on snowpack anomalies.
