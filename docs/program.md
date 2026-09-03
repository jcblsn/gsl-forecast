# Autoresearch program for gsl-forecast

This file is the research strategy for an agent-driven improvement loop, in the pattern of
karpathy/autoresearch: one editable asset, one scalar metric, one fixed harness. The loop
fits the multivariate layer and not the univariate baselines, whose design space is too
small to search. The data pipeline is not part of the loop: a new source is a reviewed
pipeline change first, and only then a feature the loop may use.

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

1. Read the latest experiment's metrics: `uv run gsl-results <experiment_id>`. The
   committed record of the same run is `data/results/cv_summary.csv`.
2. Propose one change to one multivariate model (features, lags, regularisation, fallback
   rules, the stage-two step in `inflow_chain`, a new model file).
3. Run the harness. Compare `peak_mae_feb` and `mae_h6` against the previous run and
   against `ets_damped_s12`.
4. Keep the change (commit) if `peak_mae_feb` improves without `mae_h6` getting worse by
   more than 0.05 ft; otherwise revert.
5. Log one line per experiment in `docs/autoresearch.log`: date, model, change, metrics.

## Guardrails

- A model may only use columns dated at or before the cutoff. The leakage test in
  `tests/test_multivariate.py` must keep passing; add a case for any new feature.
- Do not touch `train_start`, the cutoff policy, or the horizon.
- Do not add models that need data outside `monthly_elevation` and `monthly_covariates`
  without first extending the pipeline in a separate, reviewed change.

## Ideas queue

Results so far are in `docs/autoresearch.log`. Reservoir storage as a plain extra regressor
overfits (about 13 training rows per calendar month); it needs a lower-dimensional form,
such as storage deficit as a fraction of capacity, or a longer training window. The head
difference between the arms helps the peak and is registered as `swe_head`.

Columns already in `monthly_covariates` (see the README data section):

- `swe_pct_median_*` and `prec_pct_median_*` in place of raw inches (the site roster grew
  from 18 to 55 sites, so raw means drift).
- Separate Bear/Weber/Provo-Jordan terms instead of the pooled index.
- Reservoir storage in a form that does not add a free coefficient per month: deficit
  below the long-run maximum, or a single pooled regression across calendar months.
- `sms_eom_gsl` (8-inch soil moisture, from 1999) for autumn cutoffs.
- `head_diff_ft` in the `inflow_chain` bucket step (it already helps `swe_regression`).
- Ridge `alpha` well above 1e-3 for any model with three or more features.
- Stacking: regress `ets_damped_s12` residuals at each horizon on snowpack anomalies.
- A blend that follows `swe_regression` at leads 1-12 and moves to `ets_damped_s12` by
  lead 24, registered as its own model so it is exported and verified.

Needs pipeline work first:

- Bathymetry so `inflow_chain` uses lake area for evaporation instead of the level proxy.
- The issued NRCS inflow forecast (`nrcs_inflow_forecasts`, from 2024) as an in-season
  anchor for `inflow_chain` stage one once there are enough seasons to fit.
- Climate division temperature and precipitation (NOAA nClimDiv) for the evaporation term.
- Climate indices (NINO4, PDO, PNA, SOI) for leads beyond 12 months.
